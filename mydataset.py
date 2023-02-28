import os

import numpy as np
import torch
from deepchem.feat import MolGraphConvFeaturizer
from sklearn.model_selection import train_test_split
from torch.functional import F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm

from utils import set_random_seed
import pandas as pd

set_random_seed()

# document path
root = os.getcwd()
data_dir = os.path.join(root, 'data')
[os.mkdir(data_dir + "/" + i) for i in ['raw', 'processed'] if not os.path.exists(data_dir + "/" + i)]
raw_dir = os.path.join(data_dir, 'raw')
processed_dir = os.path.join(data_dir, 'processed')


def raw_label_processed( dir=data_dir + '/Toxicity_SMILES.xlsx' ):
	# 1. read raw data
	# 2. process raw data
	# 3. save processed data
	for i in ['rat', 'mouse']:
		for j in ['oral', 'subcutaneous']:
			raw_df = pd.read_excel(dir, sheet_name=f'{i}_{j}', engine='openpyxl',
			                       header=None, names=['SMILES', 'Dose', 'pDose', 'Class'], skiprows=1)
			raw_df['pDose'] = np.log10(raw_df['Dose']).round(3)
			if j == 'oral':  # oral dose
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
			with open(data_dir + '/' + i + '_' + j + '.csv', 'w', newline='') as f:
				f.write(raw_df.to_csv(index=False, header=True))
			print(f'{i}_{j} is done!')


# raw_label_processed()
df_rat_oral = pd.read_csv(data_dir + '/rat_oral.csv', usecols=['SMILES', 'Class'])
print(df_rat_oral.head())
X = df_rat_oral['SMILES']  # 取出smiles和name列
y = df_rat_oral['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True,
                                                    stratify=df_rat_oral['Class'])  # 划分训练集和测试集

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

test_df.to_csv(raw_dir + "/test.csv", index=False)
train_df.to_csv(raw_dir + "/train.csv", index=False)

train_dir = os.path.join(raw_dir, 'train.csv')
test_dir = os.path.join(raw_dir, 'test.csv')

data = pd.read_csv(train_dir).reset_index()
featurizer = MolGraphConvFeaturizer(use_edges=True)  # 提取分子特征,转化为图数据use_edges=True表示使用边
data_list = []
for idx, mol in tqdm(data.iterrows(), total=data.shape[0]):  # 读取csv文件中的每一行
	try:
		f = featurizer.featurize(mol['SMILES'])
		mol_graph = f[0]
		x = torch.tensor(mol_graph.node_features, dtype=torch.float)  # 提取原子特征
		edge_index = torch.tensor(mol_graph.edge_index, dtype=torch.long).contiguous()  # 提取边
		data = Data(x=x, edge_index=edge_index)
	except:
		continue
	data.y = mol["Class"]  # binary classification label
	data.smiles = mol["SMILES"]

	data_list.append(data)
# cls = mol['label']
# data, slices = collate(Dataset, data_list)  # 将数据列表转换为一个大的数据集
# torch.save((data, slices), processed_dir + 'data.pt')
# print(data_list)
train_dataloader = DataLoader(data_list, batch_size=1, shuffle=True)

data = pd.read_csv(test_dir).reset_index()
featurizer = MolGraphConvFeaturizer(use_edges=True)  # 提取分子特征,转化为图数据use_edges=True表示使用边
data_list_test = []
for idx, mol in tqdm(data.iterrows(), total=data.shape[0]):  # 读取csv文件中的每一行
	try:
		f = featurizer.featurize(mol['SMILES'])
		mol_graph = f[0]
		x = torch.tensor(mol_graph.node_features, dtype=torch.float)  # 提取原子特征
		edge_index = torch.tensor(mol_graph.edge_index, dtype=torch.long).contiguous()  # 提取边

		data = Data(x=x, edge_index=edge_index)
	except:
		continue
	data.y = mol["Class"]  # binary classification label
	data.smiles = mol["SMILES"]
	data_list_test.append(data)
test_dataloader = DataLoader(data_list_test, batch_size=1, shuffle=True)


class GCN(torch.nn.Module):
	def __init__( self, hidden_channels ):
		super(GCN, self).__init__()
		self.conv1 = GCNConv(data_list[0].x.shape[1], hidden_channels)
		self.conv2 = GCNConv(hidden_channels, hidden_channels)
		self.conv3 = GCNConv(hidden_channels, hidden_channels)
		self.lin = Linear(hidden_channels, 5)

	def forward( self, x, edge_index, batch ):
		# 1. Obtain node embeddings
		# print(np.array(x).shape)
		edge_index = torch.tensor(np.squeeze(np.array(edge_index)))
		x = self.conv1(x, edge_index)
		x = x.relu()
		x = self.conv2(x, edge_index)
		x = x.relu()
		x = self.conv3(x, edge_index)

		# 2. Readout layer
		x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

		# 3. Apply a final classifier
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.lin(x)

		return x


model = GCN(hidden_channels=64)

optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
criterion = torch.nn.CrossEntropyLoss()


def train():
	model.train()

	for data in train_dataloader:  # Iterate in batches over the training dataset.
		out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
		y = torch.tensor([[0., 0., 0., 0., 0.]])
		out = torch.flatten(out, start_dim=1)
		#  print(y, out)
		loss = criterion(out, y)  # Compute the loss.
		loss.backward()  # Derive gradients.
		optimizer.step()  # Update parameters based on gradients.
		optimizer.zero_grad()  # Clear gradients.
	return loss


def test( loader ):
	model.eval()

	correct = 0
	for data in loader:  # Iterate in batches over the training/test dataset.
		out = model(data.x, data.edge_index, data.batch)
		out = torch.flatten(out, start_dim=1)
		pred = out.argmax(dim=1)  # Use the class with highest probability.
		if data.y == pred:
			correct += 1
	return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 30):
	loss = train()
	train_acc = test(train_dataloader)
	test_acc = test(test_dataloader)
	print(f'Epoch: {epoch:03d}, Loss: {loss:.5f} Train Acc: {train_acc:.5f}, Test Acc: {test_acc:.5f}')


def normalize_pDose():
	pass
