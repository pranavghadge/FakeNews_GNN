import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, Data
import torch_geometric
from tqdm import tqdm

# Corrected Sample Dataset Class
class SampleDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, size=100, features=3, classes=2, transform=None, pre_transform=None):
        super(SampleDataset, self).__init__(root, transform, pre_transform)
        self.size = size
        self.features = features  # Renamed attribute
        self.classes = classes    # Renamed attribute
        self.data_list = self.generate_data()

    def generate_data(self):
        data_list = []
        for _ in range(self.size):
            data = Data()
            data.x = torch.randn(10, self.features)  # Node features
            data.edge_index = torch.randint(0, 10, (2, 20))  # Edges
            data.y = torch.randint(0, self.classes, (1,))  # Graph label
            data_list.append(data)
        return data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

# Define the GNN Model
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)  # Global pooling

        return F.log_softmax(x, dim=1)

# Settings
features = 3   # Number of node features
classes = 2    # Number of classes
hidden_channels = 16

# Dataset and DataLoader
dataset = SampleDataset(root='/tmp/SampleDataset', features=features, classes=classes)
train_set, test_set = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(features, hidden_channels, classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training Loop
model.train()
for epoch in tqdm(range(10)):  # Set the number of epochs
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
correct = 0
for data in test_loader:
    data = data.to(device)
    out = model(data)
    pred = out.argmax(dim=1)
    correct += int((pred == data.y).sum())
accuracy = correct / len(test_set)
print(f'Test Accuracy: {accuracy:.4f}')
