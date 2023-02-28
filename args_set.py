# difine the parameters in model
import argparse  # 自定义参数
import torch

parser = argparse.ArgumentParser(description='parameters in model')  # 创建参数解析器
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available(), help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during traing pass')
parser.add_argument('--seed', type=int, default=42, help='Random seed')  # 随机数种子
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train')  # 循环次数
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')  # 学习率

parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay(L2 loss)')  # 权重衰退
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units')  # 隐藏层数量
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')  # Dropout
