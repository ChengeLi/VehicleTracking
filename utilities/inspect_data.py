# This is a program for inspecting data files
import cv2
import os
import sys
import pdb
import pickle
import numpy as np
import glob as glob
from scipy.io import loadmat,savemat
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from DataPathclass import *
DataPathobj = DataPath(dataSource,VideoIndex)
from parameterClass import *
Parameterobj = parameter(dataSource,VideoIndex)


smoothFiles = sorted(glob.glob(os.path.join(DataPathobj.smoothpath,'klt*.mat')))
kltfiles = sorted(glob.glob(os.path.join(DataPathobj.kltpath,'klt*.mat')))
clustered_result = sorted(glob.glob(os.path.join(DataPathobj.unifiedLabelpath,'Complete*'+Parameterobj.clustering_choice+'*.mat')))
clustered_result = clustered_result[0]

trjID   = np.uint32(loadmat(clustered_result)['trjID'][0]) # labeled trjs' indexes
mlabels = np.int32(np.ones(max(trjID)+1)*-1)  #initial to be -1
labels  = loadmat(clustered_result)['label'][0]
for idx,ID in enumerate(trjID):  # ID=trjID[idx], the content, trj ID
    mlabels[int(ID)] = np.int(labels[int(idx)])

matidx = 0
trunkTrjFile = loadmat(smoothFiles[matidx])
xtrj = csr_matrix(trunkTrjFile['xtracks'], shape=trunkTrjFile['xtracks'].shape).toarray()
ytrj = csr_matrix(trunkTrjFile['ytracks'], shape=trunkTrjFile['ytracks'].shape).toarray()
ttrj = csr_matrix(trunkTrjFile['Ttracks'], shape=trunkTrjFile['Ttracks'].shape).toarray()
ttrj[ttrj ==np.max(ttrj[:])]=np.nan
chunklabels = mlabels[trunkTrjFile['trjID'][0]]

plt.scatter(xtrj[chunklabels==22,:][xtrj[chunklabels==22,:]!=0],ytrj[chunklabels==22,:][xtrj[chunklabels==22,:]!=0],color = 'g')
plt.scatter(xtrj[chunklabels==24,:][xtrj[chunklabels==24,:]!=0],ytrj[chunklabels==24,:][xtrj[chunklabels==24,:]!=0],color = 'r')
plt.scatter(xtrj[chunklabels==26,:][xtrj[chunklabels==26,:]!=0],ytrj[chunklabels==26,:][xtrj[chunklabels==26,:]!=0],color = 'b')


labelTarget = 50
labelTarget = 45
labelTarget = 46


plt.scatter(xtrj[chunklabels==labelTarget,:][xtrj[chunklabels==labelTarget,:]!=0],ytrj[chunklabels==labelTarget,:][xtrj[chunklabels==labelTarget,:]!=0])
print 'number of points in this class:', sum(chunklabels==labelTarget)

for frmNum in xrange(int(np.min(np.nanmin(ttrj[chunklabels==labelTarget,:],1))),int(np.max(np.nanmax(ttrj[chunklabels==labelTarget,:],1))),1):
	plt.scatter(xtrj[chunklabels==labelTarget,frmNum],ytrj[chunklabels==labelTarget,frmNum])
	plt.draw()
	pdb.set_trace()
















