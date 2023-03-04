import os

import networkx as nx
import numpy as np
import torch
import torch_geometric
from matplotlib import pyplot as plt

import args_set

args = args_set.parser.parse_args()  # 获取设置的参


# set the random seed
def set_random_seed( seed=args.seed ):
	torch_geometric.seed_everything(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)


# show the hyperparameter of model
def show_Hyperparameter( args ):
	argsDict = args.__dict__  # 获取类中包含的属性
	# print(argsDict)
	for key in argsDict:
		print(key, ':', argsDict[key])


# document path
def generate_raw_processed_dir():
	root = os.getcwd()
	data_dir = os.path.join(root, 'data')

	[os.mkdir(data_dir + "/" + i) for i in ['raw', 'processed'] if not os.path.exists(data_dir + "/" + i)]
	raw_dir = os.path.join(data_dir, 'raw')
	processed_dir = os.path.join(data_dir, 'processed')
	return root, data_dir, raw_dir, processed_dir


# show the graph
def graph_showing( data ):
	G = nx.Graph()
	edge_index = data['edge_index'].t()
	edge_index = np.array(edge_index.cpu())
	G.add_edges_from(edge_index)

	nx.draw(G, with_labels=True)
	plt.show()
