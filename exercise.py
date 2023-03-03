

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