import os
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array
import numpy as np
import scipy.sparse as sp
import random

# Function to read a file from disk
def read_file(folder, name, dtype=None):
    path = os.path.join(folder, f'{name}.txt')
    return read_txt_array(path, sep=',', dtype=dtype)

# Function to split data into graph batches
def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = torch.bincount(batch).tolist()

    slices = {'edge_index': edge_slice}
    for attr in ['x', 'edge_attr', 'y']:
        item = getattr(data, attr, None)
        if item is not None:
            if item.size(0) == batch.size(0):
                slices[attr] = node_slice
            else:
                slices[attr] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    return data, slices

# Function to read and process graph data
def read_graph_data(folder, feature):
    node_attributes = sp.load_npz(os.path.join(folder, f'new_{feature}_feature.npz'))
    edge_index = read_file(folder, 'A', torch.long).t()
    node_graph_id = np.load(os.path.join(folder, 'node_graph_id.npy'))
    graph_labels = np.load(os.path.join(folder, 'graph_labels.npy'))

    x = torch.from_numpy(node_attributes.todense()).to(torch.float)
    y = torch.from_numpy(graph_labels).to(torch.long)
    _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    edge_index, _ = add_self_loops(edge_index, None)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

    data = Data(x=x, edge_index=edge_index, y=y)
    data, slices = split(data, torch.from_numpy(node_graph_id).to(torch.long))

    return data, slices

# Custom dataset class
class FNNDataset(InMemoryDataset):
    def __init__(self, root, feature='spacy', empty=False, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.feature = feature
        super(FNNDataset, self).__init__(self.root, transform, pre_transform, pre_filter)
        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'new_{self.feature}_feature.npz', 'A.txt', 'node_graph_id.npy', 'graph_labels.npy']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Data should be pre-downloaded in Google Drive
        pass

    def process(self):
        self.data, self.slices = read_graph_data(self.raw_dir, self.feature)
        torch.save((self.data, self.slices), self.processed_paths[0])

# Example of using the dataset
dataset = FNNDataset(root='/content/drive/My Drive/GNN/gossipcop', feature='spacy')
