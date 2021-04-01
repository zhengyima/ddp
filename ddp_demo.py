from __future__ import division
from __future__ import print_function

import os
import pickle
import logging
import argparse
import random
from tqdm import tqdm, trange
import json
import moxing as mox
import torch.multiprocessing as mp
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    主进程
    """
    parser = argparse.ArgumentParser()

    ## 用户设置的模型参数
    parser.add_argument('--aa', action='store_true')
    parser.add_argument('--bb', action='store_true')
    parser.add_argument('--cc', action='store_true')

    ## DDP的前四要素
    parser.add_argument('--rank', type=int, default=0, help='Index of current task')  # 表示当前是第几个节点
    parser.add_argument('--world_size', type=int, default=1, help='Total number of tasks')  # 表示一共有几个节点
    parser.add_argument('--dist_backend', type=str, default='nccl',
                        help='distributed backend')
    parser.add_argument('--init_method', default=None,
                        help='print process when training')  # 表示PyTorch分布式作业中master的地址，modelarts已经完成设置

    args, unparsed = parser.parse_known_args()
    logger.info('args: %s' % args)

    ## 获取每个节点的GPU个数并赋给ngpus_per_node
    print("available GPU numbers:", torch.cuda.device_count())
    ngpus_per_node = torch.cuda.device_count()

    ## 计算新的进程总数，假设使用12个节点，world_size初始为12,12*8=96,为新的world_size
    args.world_size = ngpus_per_node * args.world_size

    ## 在主进程中通过torch多进程spawn启动多个子进程，nprocs为每个节点启动的进程数
    ## args为向子进程函数main_worker中传入的参数，这里传入的是每个节点的卡数及初始化的全部args
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    ## 多进程结束后的操作，用户可以仅在0号节点(rank=0)上进行一些后续操作，比如将子进程运行的中间结果拷贝至云道
    if args.rank == 0:
        logger.info('start to copy model to {}!'.format(args.s3_output_dir))
        mox.file.copy_parallel(args.output_dir, args.s3_output_dir)
        logger.info('finish copying model to {}!'.format(args.s3_output_dir))



def main_worker(local_rank, ngpus_per_node, args):
    """
    子进程
    """
    ## local rank参数为主进程中mp.spawn自动传入的，需要放在第一个位置接收，范围从0~8
    ## 先计算global_rank，如果是12个节点，那么global rank的范围从0~95
    global_rank = args.rank * (ngpus_per_node) + local_rank

    ## 打印local_rank，这里认为每个节点有几张卡就启动几个进程
    if local_rank is not None:
        print("Use GPU: {} for training".format(local_rank))

    ## 初始化进程组，需要使用DDP的六要素
    print('backend:', args.dist_backend)
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method,
                            world_size=args.world_size, rank=global_rank)

    ## 计算每个进程的batch size，这里默认train_batch_size为所有进程的，比如你想让每个进程的batch size为1，而你有12个8卡的机器，那么你的初始batch size需要设为96
    args.train_batch_size = int(args.train_batch_size / args.world_size)

    ## 锁定模型随机种子，保证每个进程的模型初始化相同
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ## 获取训练数据及特征
    train_examples = get_train_examples() # 用户自定义
    train_features = convert_examples_to_features(train_examples) # 转为特征，用户自定义

    ## 对数据进行distributed sampler,保证每个进程采出的数据不一样
    train_sampler = DistributedSampler(train_features)
    train_dataloader = DataLoader(train_features, sampler=train_sampler, batch_size=args.train_batch_size)

    ## 初始化DDP模型，模型分布在不同GPU上
    model = MyModel.from_pretrained() ## 用户自定义加载模型
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    ## 初始化优化器
    optimizer = MyOptimizer() ## 用户自定义初始化

    ## 训练
    model.train()
    for e in range(int(args.num_train_epochs)):
        train_sampler.set_epoch(e) # 保证每个epoch启动相同的random
        for step, batch in enumerate(train_dataloader):
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if global_rank == 0: ## 仅在global_rank=0时打印loss，否则每个进程会打印一份，由于数据输入不同，每个进程的loss不同
                logger.info('loss: %s' % (sum(tr_loss) / len(tr_loss)))
                ## 可以打印每个进程的梯度，梯度和模型参数是完全相同
                if global_rank == 0 or global_rank == 1 or global_rank == 2:
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            print('-->global rank:', global_rank, '-->grad:', p.grad.data)
                            break
                        break
        ## 在global_rank = 0处保存模型即可，如果每个进程都在保存模型较危险
        if global_rank == 0:
            save(model, args.output_dir, str(e + 1))

if __name__ == "__main__":
    main()
