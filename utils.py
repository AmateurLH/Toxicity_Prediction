import os
import random
import numpy as np
import torch
import torch_geometric
import args_set
args = args_set.parser.parse_args()  # 获取设置的参


def set_random_seed(seed=args.seed):
	torch_geometric.seed_everything(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)


def show_Hyperparameter( args ):
	argsDict = args.__dict__  # 获取类中包含的属性
	# print(argsDict)
	for key in argsDict:
		print(key, ':', argsDict[key])
