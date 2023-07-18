# DwHGCN
The input data is the series of matrix in .mat file generated from MatLab. It is in a cell format in M x 4, where M is the number of subjects in each catagory (i.e. higher IQ group or lower IQ group). 
The 5 columns of matrix are:
                   1. N x N matrix, the original hypergraph similarity matrix, N is the number of nodes (ROIs).
                   2. N x N matrix, the incidence matrix.
                   3. diagonal matrix, the edge degree with exponential (-1/2).
                   4. N x P matrix, the original graph signals.
            
