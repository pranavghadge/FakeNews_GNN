import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv, global_mean_pool, Linear
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

# Sample Dataset
class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, size=100, num_features=310, num_classes=2):
        self.size = size
        self.num_features = num_features
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = torch_geometric.data.Data()
        data.x = torch.randn(16, self.num_features)  # Node features
        data.edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # Edge indices
        data.y = torch.randint(0, self.num_classes, (1,))  # Label
        return data

# GCNFN Model
class GCNFN(torch.nn.Module):
    def __init__(self, num_features, nhid, num_classes, concat=False):
        super(GCNFN, self).__init__()
        self.concat = concat
        self.conv1 = GATConv(num_features, nhid * 2)
        self.conv2 = GATConv(nhid * 2, nhid * 2)

        if concat:
            self.fc0 = Linear(num_features, nhid)
            self.fc1 = Linear(nhid * 2 + nhid, nhid)  # Adjusted input size after concatenation
        else:
            self.fc1 = Linear(nhid * 2, nhid)  # Only nhid * 2 from conv2

        self.fc2 = Linear(nhid, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.selu(self.conv1(x, edge_index))
        x = F.selu(self.conv2(x, edge_index))
        x = F.selu(global_mean_pool(x, batch))

        if self.concat:
            news = torch.stack([torch.mean(data.x[data.batch == idx], dim=0) for idx in torch.unique(data.batch)])
            news = F.relu(self.fc0(news))
            x = torch.cat([x, news], dim=1)

        x = F.selu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.fc2(x), dim=-1)

        return x

# Settings
num_features = 310
nhid = 128
num_classes = 2
batch_size = 128
epochs = 60
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset and DataLoader
dataset = SampleDataset(num_features=num_features, num_classes=num_classes)
train_set, test_set = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Model
model = GCNFN(num_features, nhid, num_classes, concat=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Training Loop
model.train()
for epoch in tqdm(range(epochs)):
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