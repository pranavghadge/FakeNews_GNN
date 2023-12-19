import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_max_pool as gmp
from torch_geometric.data import Data, DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

# Sample Dataset Class (replace with your actual dataset)
class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, size=100, num_features=3, num_classes=2):
        self.size = size
        self.num_features = num_features
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = Data()
        data.x = torch.randn(10, self.num_features)  # Node features
        data.edge_index = torch.randint(0, 10, (2, 20))  # Edges
        data.y = torch.randint(0, self.num_classes, (1,))  # Graph label
        return data

# Define the Model
class Model(torch.nn.Module):
    def __init__(self, num_features, nhid, num_classes, model_type='gcn', concat=False):
        super(Model, self).__init__()
        self.num_features = num_features  # Define num_features as an instance attribute
        self.concat = concat

        if model_type == 'gcn':
            self.conv1 = GCNConv(num_features, nhid)
        elif model_type == 'sage':
            self.conv1 = SAGEConv(num_features, nhid)
        elif model_type == 'gat':
            self.conv1 = GATConv(num_features, nhid)

        if concat:
            self.lin0 = torch.nn.Linear(num_features, nhid)
            self.lin1 = torch.nn.Linear(nhid + nhid, nhid)
        else:
            self.lin1 = torch.nn.Linear(nhid, nhid)

        self.lin2 = torch.nn.Linear(nhid, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = gmp(x, data.batch)  # Global max pooling

        if self.concat:
            news = torch.cat([data.x[data.batch == idx][0] for idx in torch.unique(data.batch)], dim=0)
            news = news.view(-1, self.num_features)  # Reshape to match features
            news = F.relu(self.lin0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.lin1(x))

        x = F.log_softmax(self.lin2(x), dim=-1)

        return x

# Settings
num_features = 3
nhid = 128
num_classes = 2
batch_size = 128
epochs = 35
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset and DataLoader
dataset = SampleDataset(num_features=num_features, num_classes=num_classes)
train_set, test_set = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Model
model = Model(num_features, nhid, num_classes, model_type='sage', concat=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training and Evaluation
model.train()
for epoch in tqdm(range(epochs)):
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

model.eval()
correct = 0
for data in test_loader:
    data = data.to(device)
    out = model(data)
    pred = out.argmax(dim=1)
    correct += int((pred == data.y).sum())
accuracy = correct / len(test_set)
print(f'Test Accuracy: {accuracy:.4f}')
