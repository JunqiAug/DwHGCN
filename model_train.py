from datetime import datetime
from torch_geometric.data import Data
# from Lapdataset import PNCDataset
# from Lap_cog import PNCDataset
# from Lap_cog_withoutAUG import PNCDataset
from Lap_nback import PNCDataset
# from Lap_total import PNCDataset
# from Lap_emoid import PNCDataset
from torch_geometric.datasets import TUDataset
import torch
import numpy as np
from models import train
from models import test
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import scipy.io as sio



def main():
	writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
	#dataset = TUDataset(root='../', name='ENZYMES', use_edge_attr=True)
	dataset = PNCDataset(root='../')
	# labels = dataset.data.y
	# n_fold = 10
	# skf = StratifiedKFold(n_splits=n_fold, shuffle=True)
	# y_probs = []
	# y_true = []
	# acc = []
	# sen = []
	# spe = []
	# for Ind_train, Ind_test in skf.split(labels, labels):

	N = len(dataset)
	Ind_all = list(range(N))
	Ind_train, Ind_test = train_test_split(Ind_all, test_size=0.1)
	Ind_train, Ind_val = train_test_split(Ind_train, test_size=0.1)

	print(Ind_test)

	train_data = torch.utils.data.Subset(dataset, Ind_train)
	test_data = torch.utils.data.Subset(dataset, Ind_test)
	val_data = torch.utils.data.Subset(dataset, Ind_val)


	print("Training samples: ", len(train_data))
	print("Validation samples: ", len(val_data))
	print("Test samples: ", len(test_data))
	#
	train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

	val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

	test_loader = DataLoader(test_data, batch_size=16, shuffle=False)


	model, val_acc = train(train_loader, test_loader, val_loader, writer)
	print(model)

	# torch.save(model, '.\saved_model\modelrest1.pkl')
	# model = torch.load(r'D:\projects\laplacian_learning_HGCN\saved_model\modelt.pkl')
	print('Use {} subjects for testing'.format(len(test_loader)))
	# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
	# test_acc, score, ytrue, sen1, spe1 = test(test_loader, model)
	# #
	# weight_vector = model.weight_lap
	# weight_value = weight_vector.data
	# weight_index = torch.argsort(weight_value)
	# for i in range(5):
	# 	weight_value[weight_index[i]] = 0
	#
	# model.weight_lap.data = weight_value

	test_acc1, score1, ytrue1, sen1, spe1 = test(test_loader, model)
	#
	# weight_vector = torch.tensor(weight_vector)


	# fpr, tpr, temp = roc_curve(ytrue, score)
	# optimal_idx = np.argmax(tpr - fpr)
	# thres1 = temp[optimal_idx]
	# model_auc = auc(fpr, tpr)
	print('test accuracy{:.4f}'.format(test_acc1))
	# print('Auc {:.4f}' .format(model_auc))
	# print('sensitivity {:.4f}' .format(sen1))
	# print('specificity {:.4f}'.format(spe1))
	# print('threshold {:.4f}'.format(thres))
	# print('threshold1 {:.4f}'.format(thres1))

	# sio.savemat('.\saved_model\cinepsl1.mat', {'score': score})
	# sio.savemat('.\saved_model\weight_1.mat', {'weight1': weight_vector})

	# np.savetxt('.\saved_model\weightemoind_10.txt', weight_vector.numpy())
	total_num_parameters = 0
	for parameters in model.parameters():
		total_num_parameters += parameters.reshape(-1).size()[0]
	print('Total number of parameters is {:.2f} k'.format(total_num_parameters / 1000))

	# sio.savemat('.\saved_model\srest_with_top5.mat', {'score': score})
	# sio.savemat('.\saved_model\yrest_with_top5.mat', {'label': ytrue})

	sio.savemat('.\saved_model\srest_remove_top51.mat', {'score1': score1})
	sio.savemat('.\saved_model\yrest_remove_top51.mat', {'label1': ytrue1})
	return test_acc1, sen1, spe1, ytrue1, score1
	# return val_acc, weight_vector
	# return test_acc, classification_report(y_true, y_pre)


if __name__ == "__main__":
	main()
