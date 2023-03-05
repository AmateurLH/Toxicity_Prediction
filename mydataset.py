import os

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from deepchem.feat import MolGraphConvFeaturizer

import torch
from torch_geometric.data import Dataset, Data

import utils

utils.set_random_seed()
root, data_dir, raw_dir, processed_dir = utils.generate_raw_processed_dir()


def raw_label_processed( animal, route ):
	# 1. read the raw data
	# 2. process the raw data label
	# 3. save as the raw data with label

	raw_df = pd.read_excel(raw_dir + '/Toxicity_SMILES.xlsx', sheet_name=f'{animal}_{route}', engine='openpyxl',
	                       header=None, names=['SMILES', 'Dose', 'pDose', 'Class'], skiprows=1)
	raw_df['pDose'] = np.log10(raw_df['Dose']).round(3)
	if route == 'oral':  # oral dose
		raw_df['Class'] = raw_df['Dose'].apply(lambda x: 0 if x <= 5
		else (1 if 5 < x <= 50
		      else (2 if 50 < x <= 300
		            else (3 if 300 < x <= 2000
		                  else 4))))
	else:  # subcutaneous dose
		raw_df['Class'] = raw_df['Dose'].apply(lambda x: 0 if x <= 50
		else (1 if 50 < x <= 200
		      else (2 if 200 < x <= 1000
		            else (3 if 1000 < x <= 2000
		                  else 4))))
	# counts = raw_df['class'].value_counts(sort=True)
	# print(counts)
	with open(raw_dir + f"/{animal}_{route}" + '.csv', 'w', newline='') as f:
		f.write(raw_df.to_csv(index=False, header=True))
	print(f'{animal}_{route} is done!')


def generate_raw_train_data( animal='rat', route='oral' ):
	raw_label_processed(animal, route)

	if not os.path.exists(raw_dir + f"/{animal}_{route}"):
		os.mkdir(raw_dir + f"/{animal}_{route}")

	df = pd.read_csv(raw_dir + f'/{animal}_{route}.csv', usecols=['SMILES', 'Class'])
	X = df['SMILES']  # 取出smiles和name列
	y = df['Class']
	X_train, X_test, y_train, y_test = \
		train_test_split(X, y, test_size=0.2, shuffle=True, stratify=df['Class'])  # 划分训练集和测试集
	train_df = pd.concat([X_train, y_train], axis=1)
	test_df = pd.concat([X_test, y_test], axis=1)
	test_df.to_csv(raw_dir + f"/{animal}_{route}/test.csv", index=False)
	train_df.to_csv(raw_dir + f"/{animal}_{route}/train.csv", index=False)
	print(f"raw_{animal}_{route}_data is done!")
	return raw_dir + f"/{animal}_{route}/train.csv", raw_dir + f"/{animal}_{route}/test.csv"


class MyDataSets(Dataset):
	def __init__( self, root, csv_file, transform=None, pre_transform=None, test=False ):
		self.csv_file = csv_file  # The name of the csv file containing SMILES and labels
		self.animal = csv_file.split('/')[-2].split('_')[0]
		self.route = csv_file.split('/')[-2].split('_')[1]
		self.test = test
		self.data = pd.read_csv(self.csv_file).reset_index()

		self.processed_dir_train_data = utils.generate_processed_data_dir(self.animal, self.route, test=False)
		self.processed_dir_test_data = utils.generate_processed_data_dir(self.animal, self.route, test=True)
		super(MyDataSets, self).__init__(root, transform, pre_transform)

	@property
	def raw_file_names( self ):
		return self.csv_file  # The directory of raw file names

	@property
	def processed_file_names( self ):
		if not self.test:
			return [f'data_{i}.pt' for i in range(self.len())]  # the name of processed train file names
		else:
			return [f'data_{i}.pt' for i in range(self.len())]  # the name of processed test file names

	def download(self):
		pass  # No need to download anything

	def process( self ):
		data_list = []  # The list of graph objects
		featurizer = MolGraphConvFeaturizer(use_edges=True)  # The featurizer for converting SMILES into graphs
		for idx, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):  # Iterate over each row
			try:
				f = featurizer.featurize(mol['SMILES'])  # Featurize the SMILES string
				mol_graph = f[0]
				x = torch.tensor(mol_graph.node_features, dtype=torch.float)  # Extract node features
				edge_index = torch.tensor(mol_graph.edge_index, dtype=torch.long).contiguous()  # Extract edge indices
				data = Data(x=x, edge_index=edge_index)  # Create a Data object
			except:
				continue
			data.y = mol["Class"]  # Add binary classification label
			data.smiles = mol["SMILES"]  # Add SMILES string
			data_list.append(data)  # Append to the list
		if not self.test:
			for i in range(len(data_list)):
				torch.save(data_list[i], self.processed_dir_train_data + f"/data_{i}.pt")
		else:
			for i in range(len(data_list)):
				torch.save(data_list[i], self.processed_dir_test_data + f"/data_{i}.pt")

	def len( self ):
		if not self.test:
			folder_path = self.processed_dir_train_data
			num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
		else:
			folder_path = self.processed_dir_test_data
			num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
		return num_files  # Return the number of graphs

	def get( self, idx ):
		if not self.test:
			data = torch.load(f"{self.processed_dir_train_data}/data_{idx}.pt")
		else:
			data = torch.load(f"{self.processed_dir_test_data}/data_{idx}.pt")

		return data
