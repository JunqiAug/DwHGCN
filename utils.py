import numpy as np
import os
import torch


def weight_laplacian_penalty(weight_vector):
    """
    :param X_org: original signal (N, 142, 90)
    :param E: laplacian matrix (N, 90, 90)
    """
    loss=torch.sum(torch.abs(weight_vector))

    return 100*torch.abs(loss-1)

def manifold_hyper_reg(X_org, E):
    """
    :param X_org: original signal (N, 142, 90)
    :param E: laplacian matrix (N, 90, 90)
    """
    # X_org = X_org.to(device)
    # E = E.to(device)
    return 0.01*torch.sum(torch.diagonal(torch.matmul(torch.matmul(X_org, E), X_org.transpose(-1, -2))))/X_org.shape[0]

def mat_inverse(A):
    """
    :param A: a N by N matrix
    :Output a matrix of A' inverse where the 0s in A remain 0.
    """
    # print(A.shape)
    # b = A.view(32,-1)
    b = torch.where(A!=0, 1/A, A)
    # (ind_x,ind_y) = torch.nonzero(b, as_tuple=True)
    # ind = torch.nonzero(A, as_tuple=True)
    # print(ind.shape)
    # B = A
    # b[ind_x, ind_x]=1/b[ind_x,ind_x]

    return b