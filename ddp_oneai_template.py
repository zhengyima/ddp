import os
import torch
import argparse
import logging
import moxing as mox

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


"""
    oneAI在通用框架下与modelarts没有区别，平台可自动传入下面几个要素参数
    但通用框架版本较低，很多用户会使用自定义框架，在自定义框架下这几个参数就
    无法通过平台传入了，需要读取框架的环境变量，然后改写为博客中的几个要素，
    其余操作与modelarts没有差别，具体参数含义可参考本教程
"""

INIT_METHOD = "tcp://" + os.environ.get('BATCH_CUSTOM0_HOSTS') # BATCH_CUSTOM0_HOSTS 为选定的init机器地址和端口号，需手动加入tcp://即可等价于init_method
WORLD_SIZE = os.environ.get('DLS_TASK_NUMBER') # DLS_TASK_NUMBER 等价于初始 world_size
RANK = os.environ.get('DLS_TASK_INDEX') # DLS_TASK_INDEX 等价于初始 rank
DATA_URL = os.environ.get('DLS_DATA_URL') # DLS_DATA_URL 等价于 modelarts上的data_url
TRAIN_URL = os.environ.get('DLS_TRAIN_URL') # DLS_TRAIN_URL 等价于 modelarts上的train_url


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_url', type=str, default=DATA_URL, help='data_url')
    parser.add_argument('--train_url', type=str, default=TRAIN_URL,help='train_url')
    parser.add_argument('--init_method', default=None,
                        help='init_method for yundao')
    parser.add_argument('--dist_backend', type=str, default='nccl',
                        help='distributed backend')
    parser.add_argument('--rank', type=int, default=0, help='Index of current task')  # 表示当前是第几个节点
    parser.add_argument('--world_size', type=int, default=1, help='Total number of tasks')  # 表示一共有几个节点
    parser.add_argument("--s3_model_dir", type=str,
                        default='',
                        help='define path directory')
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("available GPU numbers: %s" % (torch.cuda.device_count()))
    args.ngpus_per_node = torch.cuda.device_count()
    # 将参数赋值给原来的表达方式
    args.rank = args.ngpus_per_node * int(RANK)
    args.world_size = args.ngpus_per_node * int(WORLD_SIZE)
    args.init_method = INIT_METHOD
    logger.info('args: %s' % args)


if __name__ == '__main__':
    main()
