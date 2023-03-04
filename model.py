import numpy as np
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import functional as F


class GCN(torch.nn.Module):
	def __init__( self, dataset, hidden_channels ):
		super(GCN, self).__init__()
		self.conv1 = GCNConv(dataset.get(0).x.shape[1], hidden_channels)
		self.conv2 = GCNConv(hidden_channels, hidden_channels)
		self.conv3 = GCNConv(hidden_channels, hidden_channels)
		self.lin = Linear(hidden_channels, 5)

	def forward(self, x, edge_index, batch):
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
