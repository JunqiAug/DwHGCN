import time
import numpy as np

from model_train import main
from sklearn.model_selection import train_test_split

test_acc_list = []
# classification_report_list =[]
y_true_list = []
y_pred_list = []
sen_list = []
spe_list = []
thres_list = []
# Ind_all = list(range(389))
# Ind_train, Ind_test = train_test_split(Ind_all, test_size=0.2)
# Ind_train, Ind_val = train_test_split(Ind_train, test_size=0.1)

for i in range(10):
    # print("This is the {}th time training.......".format(i+1))
    # val, weight = main()
    # np.savetxt('.\saved_model\weightrest' + str(i) + '.txt', weight.numpy())
    # print(val)
    test_acc, sen, spe, y_true, y_pred =main()
    test_acc_list.append(test_acc)
    y_true_list.append(y_true)
    y_pred_list.append(y_pred)
    sen_list.append(sen)
    spe_list.append(spe)
    # thres_list.append(thres)

time.sleep(10)

import scipy.io as sio
sio.savemat('.\saved_model\Acc_rest1.mat', {'accuracy': test_acc_list})
sio.savemat('.\saved_model\ytrue_rest1.mat', {'y_true': y_true_list})
sio.savemat('.\saved_model\ypred_rest1.mat', {'y_probs': y_pred_list})
sio.savemat('.\saved_model\Senrest1.mat', {'sen': sen_list})
sio.savemat('.\saved_model\Spe_rest1.mat', {'spe': spe_list})
# sio.savemat('.\saved_model\hres_cog.mat', {'thres': thres_list})

# print("test_acc_list is", test_acc_list)
# print("the report list is", classification_report_list)