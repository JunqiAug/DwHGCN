# from Lapdataset_multi_hyper import PNCDataset
# from Lap_nback import PNCDataset
from Lapdataset import PNCDataset
# from Lap_emoid import PNCDataset
# from Lap_total import PNCDataset
import numpy as np
import torch
from torch_geometric.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

dataset = PNCDataset(root='../')
data_size = len(dataset)

loader = DataLoader(dataset, batch_size=1, shuffle=False)
embs = []
# forward hook
total_feat_out = []
# total_feat_in = []

#forward hook function
def hook_fn_forward(module, input, output):
    # print(module)
    # print('input', input)
    # print('output', output)
    total_feat_out.append(output)
    # total_feat_in.append(input)

total_grad_out = []
# total_grad_in = []
# backward hook_function
def hook_fn_backward(module, grad_input, grad_output):
    # print(module)
    # print('grad_output', grad_output)
    # print('grad_input', grad_input)
    # total_grad_in.append(grad_input)
    total_grad_out.append(grad_output[0].detach())

def Relu(x):
    """
    :param x: input matrix
    :return: relu(x)
    """
    x[x<0] = 0
    return x

cam = []
one_hot = torch.tensor([[0.0, 1.0]], requires_grad=False)
model = torch.load('.\saved_model\modelt.pkl')
# change to 4 classes
# one_hot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
# for i, batch in enumerate(loader):
#     model.eval()
#     model.convs[-1].register_forward_hook(hook_fn_forward)
#     model.convs[-1].register_backward_hook(hook_fn_backward)
#     total_grad_out = []
#     total_feat_out = []
#     emb, pred, weigh, L = model(batch)
#     class_vec = torch.sum(one_hot * pred)
#     # backward
#     model.zero_grad()
#     class_loss = class_vec
#     class_loss.backward()
#     grads_val = total_grad_out[0].cpu().data.numpy().squeeze()
#     fmap = total_feat_out[0].cpu().data.numpy().squeeze()
#
#     cam.append(Relu(np.mean(grads_val @ fmap.T, axis=1)))
#     # cam.append((np.mean(grads_val @ fmap.T, axis=1)))
#
# cam = np.array(cam)
# import scipy.io as sio
# sio.savemat('.\saved_model\cam0.mat', {'cam0':cam})

# visualize use tsne
colors = []
color_list = ["red", "green"]
# color_list = ["red", "green", "blue", "yellow"]

for i, batch in enumerate(loader):
    with torch.no_grad():
        emb, pred, weigh, L = model(batch)
        embs.append(emb.detach().numpy())
        colors += [color_list[y] for y in batch.y]
embs = np.array(embs)
embs = np.reshape(embs, (embs.shape[0], embs.shape[1]*embs.shape[2]))
print(embs)
print(embs.shape)
xs, ys = zip(*TSNE(n_components=2).fit_transform(embs))
plt.scatter(xs, ys, color=colors)
plt.show()