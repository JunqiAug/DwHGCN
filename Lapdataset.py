import numpy as np
import os
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch_geometric.utils as ut
np.random.seed(42)

from scipy.io import loadmat
datapth = r'D:\projects\laplacian_learning_HGCN\data'
high_raw = loadmat(os.path.join(datapth, 'young_hgcn.mat'))
low_raw = loadmat(os.path.join(datapth, 'old_hgcn.mat'))

def Lap2wudgraph(L):
    """
    convert laplacian to weighted undirected graph
    L: n*n matrix
    node_feature = None
    return:
    node_num, edge_num, edge_index, edge_weigts
    """
    source_nodes = []
    target_nodes = []
    edge_weight = []
    edge_num = 0
    if L.shape[0] != L.shape[1]:
        raise RuntimeError('Wrong Laplacian shape!')
    else:
        node_num =L.shape[0]
    for raw in range(L.shape[0]):
        L[raw, raw] = 1
        for col in range(L.shape[1]):
            if L[raw, col] != 0:
                edge_num += 1
                source_nodes.append(raw)
                target_nodes.append(col)
                edge_weight.append(np.abs(L[raw, col]))
    edge_index = np.array([source_nodes, target_nodes])
    edge_weights = np.array(edge_weight)
    return node_num, edge_num, edge_index, edge_weights

def Lap2budgraph(X, L):
    """
    convert laplacian to binary undirected graph
    L: n*n matrix
    node_feature = n*n matrix
    return:
    node_num, edge_num, edge_index
    """
    source_nodes = []
    target_nodes = []
    edge_weights = []
    edge_num = 0
    if L.shape[0] != L.shape[1]:
        raise RuntimeError('Wrong Laplacian shape!')
    else:
        node_num =L.shape[0]
    for raw in range(L.shape[0]):
        for col in range(L.shape[1]):
            if L[raw, col] != 0:
                edge_num += 1
                source_nodes.append(raw)
                target_nodes.append(col)
    x = np.abs(X)
    edge_index = np.array([source_nodes, target_nodes])
    # edge_weights = np.array([edge_weights])
    return node_num, edge_num, edge_index, x


class PNCDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(PNCDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [r'D:\projects\laplacian_learning_HGCN\Lap.dataset']

    def download(self):
        pass

    def process(self):
        k = 1  # 1:k=10   2:k=20   3:k=30   4:k=40
        data_list = []
        # high pwmt
        # L = high_raw['young'][1, k]
        for i in range(len(high_raw['young'])):
            X = np.abs(high_raw['young'][i, 0])
            # X = torch.diag(torch.tensor(np.ones(264)))
            # L = high_raw['young'][i, 1]
            H = high_raw['young'][i, 1]
            D = high_raw['young'][i, 2]
            De = high_raw['young'][i, 3]
            Gs = high_raw['young'][i, 4]
            # node_num, edge_num, edge_ind, edge_weight = Lap2wudgraph(L)
            # edge_index = torch.tensor(edge_ind, dtype=torch.long)
            # edge_weight = torch.tensor(edge_weight, dtype=torch.float)
            x = torch.tensor(X, dtype=torch.float)
            H = torch.tensor(H, dtype=torch.float)
            D = torch.tensor(D, dtype=torch.float)
            De = torch.tensor(De, dtype=torch.float)
            Gs = torch.tensor(Gs, dtype=torch.float)
            y = torch.LongTensor([1])
            data = Data(x=x, D=D, y=y, H=H, De=De, Gs=Gs)
            data_list.append(data)

        for i in range(len(low_raw['old'])):
            X = np.abs(low_raw['old'][i, 0])
            # X = torch.diag(torch.tensor(np.ones(264)))
            # L = low_raw['old'][i, 1]
            H = low_raw['old'][i, 1]
            D = low_raw['old'][i, 2]
            De = low_raw['old'][i, 3]
            Gs = low_raw['old'][i, 4]
            # node_num, edge_num, edge_ind, edge_weight = Lap2wudgraph(L)
            # edge_index = torch.tensor(edge_ind, dtype=torch.long)
            # edge_weight = torch.tensor(edge_weight, dtype=torch.float)
            x = torch.tensor(X, dtype=torch.float)
            y = torch.LongTensor([0])
            H = torch.tensor(H, dtype=torch.float)
            D = torch.tensor(D, dtype=torch.float)
            De = torch.tensor(De, dtype=torch.float)
            Gs = torch.tensor(Gs, dtype=torch.float)
            # data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)
            data = Data(x=x, D=D, y=y, H=H, De=De, Gs=Gs)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])