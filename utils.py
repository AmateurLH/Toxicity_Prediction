import os

import networkx as nx
import numpy as np
import torch
import torch_geometric
from matplotlib import pyplot as plt

import args_set

args = args_set.parser.parse_args()  # gain the hyperparameter of model


# set the random seed
def set_random_seed( seed=args.seed ):
	torch_geometric.seed_everything(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)


# show the hyperparameter of model
def show_Hyperparameter( args ):
	argsDict = args.__dict__
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


def generate_processed_data_dir( animal, route, test=False ):
	_, _, _, processed_dir = generate_raw_processed_dir()
	if not os.path.exists(f"{processed_dir}/{animal}_{route}"):
		os.mkdir(f"{processed_dir}/{animal}_{route}")

	if not test:
		if not os.path.exists(processed_dir + f"/{animal}_{route}" + '/test'):
			os.mkdir(processed_dir + f"/{animal}_{route}" + '/test')
	else:
		if not os.path.exists(processed_dir + f"/{animal}_{route}" + '/train'):
			os.mkdir(processed_dir + f"/{animal}_{route}" + '/train')

	return processed_dir + f"/{animal}_{route}/{'train' if not test else 'test'}"


# show the graph
def graph_showing( data ):
	G = nx.Graph()
	edge_index = data['edge_index'].t()
	edge_index = np.array(edge_index.cpu())
	G.add_edges_from(edge_index)

	nx.draw(G, with_labels=True)
	plt.show()


def normalize_pDose():
	pass
