import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch_scatter import scatter_mean
from torch_geometric.data import Data, Batch
from tqdm import tqdm

# GNN Layer Definition
class GNNLayer(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GNNLayer, self).__init__()
        self.conv = GCNConv(in_feats, out_feats)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return F.relu(x), x

# Bi-Directional GCN Model
class BiGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BiGCN, self).__init__()
        self.TD_layer = GNNLayer(in_feats, hid_feats)
        self.BU_layer = GNNLayer(in_feats, hid_feats)
        self.fc = torch.nn.Linear(hid_feats * 4, out_feats)

    def forward(self, data):
        td_x, td_x_copy = self.TD_layer(data.x, data.edge_index)
        bu_x, bu_x_copy = self.BU_layer(data.x, data.edge_index)

        td_x = torch.cat((td_x, td_x_copy), 1)
        bu_x = torch.cat((bu_x, bu_x_copy), 1)
        x = torch.cat((td_x, bu_x), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return F.log_softmax(self.fc(x), dim=1)

# Sample Dataset
class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(16, 3)  # Node features
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

# Parameters
class Args:
    batch_size = 128
    epochs = 45
    nhid = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = Args()

# Dataset and DataLoader Setup
dataset = SampleDataset()
num_training = int(len(dataset) * 0.8)
num_test = len(dataset) - num_training
training_set, test_set = random_split(dataset, [num_training, num_test])

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

# Model Setup
model = BiGCN(in_feats=3, hid_feats=args.nhid, out_feats=2)
model = model.to(args.device)

# Training Loop
model.train()
for epoch in tqdm(range(args.epochs)):
    for batch in train_loader:
        batch = batch.to(args.device)
        out = model(batch)
        # Add your loss and optimizer steps here

# Note: Add your loss function, optimizer, and evaluation steps according to your task.
