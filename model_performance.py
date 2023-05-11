from datetime import datetime

#from Lapdataset import PNCDataset
# from Lapdataset_multi_hyper import PNCDataset
from Lap_nback import PNCDataset
from torch_geometric.datasets import TUDataset
import torch
from models import train
from models import test
from tensorboardX import SummaryWriter
from torch_geometric.data import DataLoader
from sklearn.metrics import classification_report

writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
#dataset = TUDataset(root='../', name='ENZYMES', use_edge_attr=True)
dataset = PNCDataset(root='../')
dataset = dataset.shuffle()
data_size = len(dataset)
train_dataset = dataset[:int(data_size * 0.8)]
test_dataset = dataset[int(data_size * 0.8):]
train_dataset = train_dataset.shuffle()
# model = train(train_dataset, writer)
# torch.save(model, '.\saved_model\model.pkl')
model = torch.load(r'D:\projects\laplacian_learning\saved_model\model9.pkl')
print('Use {} subjects for testing'.format(len(test_dataset)))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
test_acc, y_true, y_pre = test(test_loader, model)
print('test accuracy{:.4f}'.format(test_acc))
print('classification report{: .4f}\n', classification_report(y_true, y_pre))

