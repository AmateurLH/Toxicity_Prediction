import numpy as np
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import functional as F


class GCN(torch.nn.Module):
	def __init__( self, num_feature, hidden_channels, num_class ):
		super(GCN, self).__init__()
		self.conv1 = GCNConv(num_feature, hidden_channels)
		self.conv2 = GCNConv(hidden_channels, num_class)
		self.lin = Linear(num_class, num_class)

	def forward( self, data ):
		# 1. Obtain node embeddings
		# print(np.array(x).shape)
		x, edge_index = data.x, data.edge_index
		x = self.conv1(x, edge_index)
		x = x.relu()
		x = self.conv2(x, edge_index)

		# 2. Readout layer
		x = global_mean_pool(x, data.batch)  # [batch_size, hidden_channels]

		# 3. Apply a final classifier
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.lin(x)

		return x
