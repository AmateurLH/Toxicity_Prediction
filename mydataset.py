import os

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from deepchem.feat import MolGraphConvFeaturizer

import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

from utils import set_random_seed
from utils import generate_raw_processed_dir

set_random_seed()
root, data_dir, raw_dir, processed_dir = generate_raw_processed_dir()


def raw_label_processed( dir=raw_dir + '/Toxicity_SMILES.xlsx', animal='rat', route='oral' ):
	# 1. read raw data
	# 2. 对原始数据的标签进行处理
	# 3. 保存为原始训练数据

	raw_df = pd.read_excel(dir, sheet_name=f'{animal}_{route}', engine='openpyxl',
	                       header=None, names=['SMILES', 'Dose', 'pDose', 'Class'], skiprows=1)
	raw_df['pDose'] = np.log10(raw_df['Dose']).round(3)
	if route == 'oral':  # oral dose
		raw_df['Class'] = raw_df['Dose'].apply(lambda x: 1 if x <= 5
		else (2 if 5 < x <= 50
		      else (3 if 50 < x <= 300
		            else (4 if 300 < x <= 2000
		                  else 5))))
	else:  # subcutaneous dose
		raw_df['Class'] = raw_df['Dose'].apply(lambda x: 1 if x <= 50
		else (2 if 50 < x <= 200
		      else (3 if 200 < x <= 1000
		            else (4 if 1000 < x <= 2000
		                  else 5))))
	# counts = raw_df['class'].value_counts(sort=True)
	# print(counts)
	with open(raw_dir + f"/{animal}_{route}" + '.csv', 'w', newline='') as f:
		f.write(raw_df.to_csv(index=False, header=True))
	print(f'{animal}_{route} is done!')


def generate_raw_train_data( animal='rat', route='oral' ):
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


# raw_label_processed()
train_data_dir, test_data_dir = generate_raw_train_data('rat', 'oral')

# generate_raw_train_data('rat', 'subcutaneous')
# generate_raw_train_data('mouse', 'oral')
# generate_raw_train_data('mouse', 'subcutaneous')
print(len(processed_dir))


class MyDataSets(Dataset):
	def __init__( self, root, csv_file, transform=None, pre_transform=None, test=False ):
		self.csv_file = csv_file  # The name of the csv file containing SMILES and labels
		self.animal = csv_file.split('/')[-2].split('_')[0]
		self.route = csv_file.split('/')[-2].split('_')[1]
		self.test = test
		self.data = pd.read_csv(self.csv_file).reset_index()

		if not self.test:
			if not os.path.exists(processed_dir + f"/{self.animal}_{self.route}" + '/train'):
				os.mkdir(processed_dir + f"/{self.animal}_{self.route}" + '/train')
		else:
			if not os.path.exists(processed_dir + f"/{self.animal}_{self.route}" + '/test'):
				os.mkdir(processed_dir + f"/{self.animal}_{self.route}" + '/test')
		self.processed_dir_train_data = processed_dir + f"/{self.animal}_{self.route}" + '/train'
		self.processed_dir_test_data = processed_dir + f"/{self.animal}_{self.route}" + '/test'
		super(MyDataSets, self).__init__(root, transform, pre_transform)

	@property
	def raw_file_names( self ):
		return self.csv_file  # The list of raw file names

	@property
	def processed_file_names( self ):
		if not self.test:
			return [f'data_{i}.pt' for i in range(self.len())]
		else:
			return [f'data_{i}.pt' for i in range(self.len())]

	def download( self ):
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
				torch.save(data_list[i], self.processed_dir_train_data + f"/data_{i}.pt")  # Save each graph object as a separate file in the processed directory
		else:
			for i in range(len(data_list)):
				torch.save(data_list[i], self.processed_dir_test_data + f"/data_{i}.pt")  # Save each graph object as a separate file in the processed directory

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
			data = torch.load(processed_dir + f"/{self.animal}_{self.route}/train/data_{idx}.pt")
		else:
			data = torch.load(processed_dir + f"/{self.animal}_{self.route}/test/data_{idx}.pt")

		return data


train_dataset = MyDataSets(root, train_data_dir, test=False)
test_dataset = MyDataSets(root, test_data_dir, test=True)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
count = 0
for batch in test_dataloader:
	count += 1
	print(batch)
print(count)


def normalize_pDose():
	pass
