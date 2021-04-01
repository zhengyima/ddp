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
    ngpus_per_node = torch.cuda.device_count() // n ## n可以控制一个节点内部开多少进程 

    ## 计算新的进程总数，假设使用12个节点，world_size初始为12,12*8=96,为新的world_size
    args.world_size = ngpus_per_node * args.world_size

    ## 获取原生训练数据集
    train_examples = get_train_examples()

    ## 将原生数据集进行切片
    logger.info('making chunk')
    num_replicas = args.world_size  # world_size为进程数，也是切片的个数，这里假设为16
    num_samples = int(math.ceil(len(train_examples) * 1.0 / num_replicas))  # 假设原生数据集为100，由于共16个进程，因此ceil 100/16 = 7，即num_samples为每个进程应该分配多个样本，这里为7
    total_size = num_samples * num_replicas  # total_size为采用向上取整的情况下，需要的全部样本个数，7*16 = 112
    indices = list(range(len(train_examples))) # 提取原生训练数据集的索引列表
    padding_size = total_size - len(indices)  # 计算全部进程理应获得的全部样本数与原生数据集样本个数的差值 112-100=12
    ## 对原生数据集的索引进行填补，填补为total_size
    if padding_size <= len(indices):
        indices += indices[:padding_size] # 如果需要填补的样本个数比原生数据集小，则提取原生样本的前padding_size个原生数据补充到indices后面
    else:
        indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size] # 如果需要补充的样本个数比原生数据集大，则需要将原生数据集进行复制后再提取前padding_size个原生数据补充到indices后面
    ## 将切片后的数据按照global_rank的顺序存放到list中
    train_examples_list = []
    for i in range(num_replicas):
        train_features.append([train_examples[item] for item in indices[i * num_samples:(i + 1) * num_samples]])
    ## check切片的维度是否正确
    for ele in train_examples_list:
        assert len(ele) == num_samples
    assert len(train_examples_list) == num_replicas

    ## 以python的多进程方式启动每个节点的子进程，这种方式的目的是可以将切片数据作为入参输入到子进程中
    daemon = False
    mp = multiprocessing.get_context('spawn')
    error_queues = []
    processes = []
    for i in range(ngpus_per_node):
        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_wrap,
            args=(main_worker, i, (ngpus_per_node, train_features[i + args.rank * ngpus_per_node], args), error_queue), # 通过args.rank和ngpus_per_node确定应该输入至每个子进程的切片索引
            daemon=daemon,
        )
    ## python多进程方法程序化步骤,follow即可
        process.start()
        error_queues.append(error_queue)
        processes.append(process)
    ## python多进程方法程序化步骤,follow即可
    for p in processes:
        p.join()

    ## 多进程结束后的操作，用户可以仅在0号节点(rank=0)上进行一些后续操作，比如将子进程运行的中间结果拷贝至云道
    if args.rank == 0:
        logger.info('start to copy model to {}!'.format(args.s3_output_dir))
        mox.file.copy_parallel(args.output_dir, args.s3_output_dir)
        logger.info('finish copying model to {}!'.format(args.s3_output_dir))



def main_worker(local_rank, ngpus_per_node, train_examples, args):
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
    train_features = convert_examples_to_features(train_examples) # train_examples从主进程切片而来， 转为特征，用户自定义

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
