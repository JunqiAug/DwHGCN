import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import time
import torch
import torch.optim as optim
import torch_geometric as pyg
import utils as U
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  roc_curve, auc
import scipy.linalg
from torch_geometric.data import DataLoader
import config


class Laplacian_GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim, task='graph'):
        super(Laplacian_GCN, self).__init__()
        self.task = task
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim1))
        self.weight_lap = nn.Parameter(torch.Tensor(config.weight_dim))
        self.reset_parameters()
        # self.weight_lap = torch.Tensor(nn.Linear(264, 1))
        self.convs.append(self.build_conv_model(hidden_dim1, hidden_dim2))
        self.convs.append(self.build_conv_model(hidden_dim2, hidden_dim3))
        # for l in range(2):
        #     self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim3, hidden_dim3), nn.Dropout(0.25),
            nn.Linear(hidden_dim3, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        # self.dropout = 0
        self.num_layers = 3

    def reset_parameters(self):
        # torch.nn.init.kaiming_uniform_(self.weight_lap)
        torch.nn.init.uniform_(self.weight_lap)
        # torch.nn.init.constant_(self.weight_lap, 1/config.weight_dim)

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg_nn.GCNConv(input_dim, hidden_dim)

    def forward(self, data):
        # x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        x, H, De, batch, batchsize = data.x, data.H, data.De, data.batch, data.y
        # weight = torch.tensor([213, 83, 239, 127, 255])
        #
        # for i in range(5):
        #     self.weight_lap.data[weight[i]] = 0

        mask = torch.abs(torch.diag(self.weight_lap))

        H=torch.reshape(H, (batchsize.shape[0], config.weight_dim, config.weight_dim))
        # D=torch.reshape(D, (batchsize.shape[0], 264, 264))
        De = torch.reshape(De, (batchsize.shape[0], config.weight_dim, config.weight_dim))
        L = torch.matmul(torch.matmul(U.mat_inverse(torch.diag_embed(torch.matmul(H, torch.abs(self.weight_lap))**0.5)), torch.matmul(H, mask)), torch.matmul(torch.matmul(De, H.transpose(-1, -2)), U.mat_inverse(torch.diag_embed(torch.matmul(H, torch.abs(self.weight_lap))**0.5))))
        # L = torch.matmul(torch.matmul(D, torch.matmul(H, mask)),torch.matmul(torch.matmul(De, H.transpose(-1, -2)),D))
        # print(L.shape)
        # print(self.weight_lap.grad)
        # print(torch.sum(torch.abs(self.weight_lap)))
        edge_index, edge_weight = pyg.utils.dense_to_sparse(L)

        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):

            x = self.convs[i](x, edge_index, edge_weight)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # if not i == self.num_layers - 1:
            #     x = self.lns[i](x)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1), self.weight_lap, L

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
        # return F.nll_loss(pred, label, weight=torch.tensor((0.6, 0.4)))

def train(dataset, test_set, val_set, writer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_size = len(dataset.dataset)
    print('Use {} subjects for training, {} subjects for validation'.format(int(data_size * 0.9), data_size - int(data_size * 0.9)))
    # loader = DataLoader(dataset[:int(data_size * 0.9)], batch_size=16, shuffle=True)
    # val_loader = DataLoader(dataset[int(data_size * 0.9):], batch_size=8, shuffle=True)
    loader = dataset
    val_loader = val_set
    test_loader = test_set

    # build model
    model = Laplacian_GCN(config.weight_dim, config.hidden_dim1, config.hidden_dim2, config.hidden_dim3, int(2))
    model = model
    print(model)
    opt = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_norm)# weight_decay: L2 regularization/Ridge norm

    val_acc = 0
    thres = 0.5
    # train
    for epoch in range(config.max_num_epoch):
        starttime = time.time()
        total_loss = 0
        model.train()
        for batch in loader:

            opt.zero_grad()
            embedding, pred, weight_vector, L = model(batch)
            label = batch.y
            X_org = torch.reshape(batch.Gs, (label.shape[0], 231, config.weight_dim))
            # X_org = torch.reshape(batch.Gs, (label.shape[0], 565, 264))
            loss = model.loss(pred, label) + U.weight_laplacian_penalty(torch.abs(weight_vector))/batch.num_graphs + U.manifold_hyper_reg(X_org, L)/batch.num_graphs
            # loss = model.loss(pred, label)
            # loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        endtime = time.time()
        total_loss /= len(loader.dataset)
        print('Epoch {}.  Loss: {:.4f}. Training time: {:.3f}'.format(epoch, total_loss, endtime - starttime))
        writer.add_scalar("loss", total_loss, epoch)

        if epoch % 2 == 0:
            test_acc, yprobs, ytrue, sen, spe = test(val_loader, model)
            tt, _, _, _, _ = test(test_loader, model)
            fpr, tpr, thresholds = roc_curve(ytrue, yprobs)

            # print("Epoch {}. Loss: {:.4f}. val_auc: {:.4f}. ".format(epoch, total_loss, auc(fpr, tpr)))
            print("Epoch {}. Loss: {:.4f}. val_accuracy: {:.4f}. AUC: {:.4f}. ".format(epoch,  total_loss, test_acc, auc(fpr, tpr)))
            # writer.add_scalar("test accuracy", test_acc, epoch)

            if auc(fpr, tpr) > val_acc:

                # val_acc = auc(fpr, tpr)
                val_acc = auc(fpr, tpr)
                # optimal_idx = np.argmax(tpr - fpr)
                # thres = thresholds[optimal_idx]
                torch.save(model, '.\saved_model\modelt.pkl')

            if epoch % 20 == 0:
                weight_vector = model.weight_lap
                np.savetxt('.\saved_model\weights'+str(epoch)+'.txt', weight_vector.detach().numpy())

    model = torch.load('.\saved_model\modelt.pkl')

    return model, val_acc

def test(loader, model):
    model.eval()
    correct = 0
    y_pre = []
    y_true = []
    y_probs = []
    for data in loader:
        with torch.no_grad():
            emb, pred, weight_vector, L = model(data)
            label = data.y
            probs = torch.softmax(pred, dim=1)
            pred = probs.argmax(dim=1)

            y_true.extend(label.tolist())
            y_pre.extend(pred.tolist())
            y_probs.extend(probs[:, 1].tolist())

            correct += pred.eq(label).sum().item()
    # print(y_true, y_pre)
    total = len(loader.dataset)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pre).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    return correct / total, y_probs, y_true, sen, spe
    # return correct / total, sen, spe
