import pdb
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
# import numpy.linalg as np_lg
# import numpy.matlib as np_mat
from scipy.io import loadmat,savemat
# import sklearn
# from scipy import linalg
# from scipy.sparse import *
# from sklearn import mixture
# from sklearn.cluster import *
# from sklearn.manifold import *
from scipy.sparse import csr_matrix


from DataPathclass import *
DataPathobj = DataPath(dataSource,VideoIndex)
from parameterClass import *
Parameterobj = parameter(dataSource,VideoIndex)


from subspace_clustering_merge import prepare_input_data


for matidx, matfile in enumerate(adjmatfiles):
    labelsave = {} #for the save in the end
    adjfile = loadmat(matfile)
    DirName = ['upup','updown','downup','downdown']
    for dirii in range(4):
        try:
            trjAdj  = adjfile['adj_'+DirName[dirii]]
            CClabel = adjfile['c_'+DirName[dirii]]  # labels from connected Component
            trjID   = adjfile['trjID_'+DirName[dirii]]
        except:
            continue




















