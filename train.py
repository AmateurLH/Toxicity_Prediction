import torch
from torch_geometric.loader import DataLoader
import utils
from mydataset import MyDataSets, generate_raw_train_data
from model import GCN
from torch.nn import functional as F

root, data_dir, raw_dir, processed_dir = utils.generate_raw_processed_dir()

train_data_dir, test_data_dir = generate_raw_train_data('rat', 'oral')

train_dataset = MyDataSets(root, train_data_dir, test=False)
test_dataset = MyDataSets(root, test_data_dir, test=True)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

print(train_dataset.get(0))

model = GCN(num_feature=train_dataset[0].x.shape[1], hidden_channels=64, num_class=5)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)


def train( model, data_loader, optimizer ):
	model.train()
	total_loss = 0
	for data in data_loader:
		optimizer.zero_grad()
		out = model(data)
		loss = F.cross_entropy(out, data.y)
		loss.backward()
		optimizer.step()
		total_loss += loss.item() * data.num_graphs
	return total_loss / len(data_loader.dataset)


train(model, train_dataloader, optimizer)