import pdb
import os
from scipy.io import loadmat,savemat
import glob

import pickle as pickle
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
from DataPathclass import *
DataPathobj = DataPath(dataSource,VideoIndex)
from parameterClass import *
Parameterobj = parameter(dataSource,VideoIndex)

from ssc_with_Adj import ssc_with_Adj_CC  #, sscConstructedAdj_CC, sscAdj_inNeighbour

isSave      = True
isVisualize = False
# Nameprefix = 'Aug12'
Nameprefix = 'Aug10'

class trjClusteringFromAdj:
    """With constructed adjacency matrix """
    def __init__(self):
        if Parameterobj.useWarpped:      
            self.adjmatfiles = sorted(glob.glob(os.path.join(DataPathobj.adjpath,'usewarpped*'+Nameprefix+'*.mat')))
        else:
            self.adjmatfiles = sorted(glob.glob(os.path.join(DataPathobj.adjpath,Nameprefix+'*.mat')))

        self.savePath = DataPathobj.sscpath
        self.trjmatfiles = sorted(glob.glob(os.path.join(DataPathobj.smoothpath,'*.mat')))
        self.DirName = ['upup','updown','downup','downdown']

    def trjclustering(self,matidx):
        """clustering trj in each truncation"""
        adjfile = loadmat(self.adjmatfiles[matidx])
        self.labelsave = {} 
        for dirii in range(4):
            try:
                self.trjAdj  = adjfile['adj_'+self.DirName[dirii]]
                self.CClabel = adjfile['c_'+self.DirName[dirii]]  # labels from connected Component
                self.trjID   = adjfile['trjID_'+self.DirName[dirii]]
                self.non_isolatedCC_ind = adjfile['non_isolatedCC'+self.DirName[dirii]][0] #location index in the truncation
            except:
                continue
            """ andy's method, not real sparse sc, just spectral clustering"""
            self.labels_DPGMM,self.labels_spectral, self.labels_affini_prop, self.small_CC_oneCls = ssc_with_Adj_CC(self.trjAdj,self.CClabel,self.trjID,Parameterobj)
            """ construct adj use ssc"""
            # self.trjID,labels, adj = sscConstructedAdj_CC(adjfile)
            """ construct adj use ssc, with Neighbour adj as constraint"""
            # trjID,labels = sscAdj_inNeighbour(adjfile)

            self.labelsave['labels_DPGMM_'+self.DirName[dirii]] = self.labels_DPGMM
            self.labelsave['labels_spectral_'+self.DirName[dirii]] = self.labels_spectral
            self.labelsave['labels_affinity_'+self.DirName[dirii]] = self.labels_affini_prop

            self.labelsave['trjID_'+self.DirName[dirii]] = self.trjID

    def saveLabel(self,matidx):
        print "saving the labels..."
        if Parameterobj.useWarpped:
            savename = os.path.join(self.savePath,'usewarpped_'+str(matidx+1).zfill(3))
        else:
            savename = os.path.join(self.savePath,'Aug12'+str(matidx+1).zfill(3))
        savemat(savename, self.labelsave)


    def visLabel(self,matidx):
        labels = labels_DPGMM
        # labels = labels_spectral
        # visualize different classes for each Connected Component
        """use the x_re and y_re from adj mat files  """
        trjfile = loadmat(trjmatfiles[matidx])
        if Parameterobj.useWarpped:
            xtrj = csr_matrix(trjfile['xtracks_warpped'], shape=trjfile['xtracks_warpped'].shape).toarray()
            ytrj = csr_matrix(trjfile['ytracks_warpped'], shape=trjfile['ytracks_warpped'].shape).toarray()
        else:
            xtrj = csr_matrix(trjfile['xtracks'], shape=trjfile['xtracks'].shape).toarray()
            ytrj = csr_matrix(trjfile['ytracks'], shape=trjfile['ytracks'].shape).toarray()

            # xtrj = csr_matrix(trjfile['xtracks'], shape=trjfile['xtracks'].shape).toarray()[self.non_isolatedCC_ind,:]
            # ytrj = csr_matrix(trjfile['ytracks'], shape=trjfile['ytracks'].shape).toarray()[self.non_isolatedCC_ind,:]

        color = np.array([np.random.randint(0, 255) for _ in range(3 * int(max(labels) + 1))]).reshape(int(max(labels) + 1), 3)
        fig999 = plt.figure()
        plt.ion()
        ax = plt.subplot(1, 1, 1)

        newtrjID  = list(trjID[0])
        newlabels = list(labels)

        label_id = {} #one adj for each predicted class

        # for i in range(int(max(labels)) + 1):
        for i in labels:
            # pdb.set_trace()
            trjind = np.where(labels == i)[0]
            label_id[i] = trjind
            # print "trjind = ", str(trjind)
            if len(trjind) <= 5:
                # newlabels.remove(i)
                newlabels = [x for x in newlabels if x!=i]
                newtrjID = [x for x in newtrjID if x not in trjID[0][trjind]]
                # pdb.set_trace()
                continue ## skip the rest, don't draw too short trjs

            for jj in range(len(trjind)):
                startlimit = np.min(np.where(xtrj[trjind[jj],:]!=0))
                endlimit   = np.max(np.where(xtrj[trjind[jj],:]!=0))
                # lines = ax.plot(x_re[trjind[jj],startlimit:endlimit], y_re[trjind[jj],startlimit:endlimit],color = (0,1,0),linewidth=2)
                lines = ax.plot(xtrj[trjind[jj],startlimit:endlimit], ytrj[trjind[jj],startlimit:endlimit],color = (color[i-1].T)/255.,linewidth=2)
                # plt.annotate(str(trjind[jj]),(xtrj[trjind[jj],endlimit], ytrj[trjind[jj],endlimit] ))
                plt.draw()
            one_class_adj = csr_matrix(adjfile['adj'], shape=adjfile['adj_upup'].shape).toarray()[trjind,:][:,trjind]
            # one_class_adj_color = cv2.applyColorMap(np.hstack((one_class_adj,one_class_adj,one_class_adj)).reshape((one_class_adj.shape[0],-1,3)), cv2.COLORMAP_JET)

            # plt.imshow(one_class_adj,cmap = 'jet')
            # plt.draw()
        pickle.dump(label_id,open(os.path.join(savePath,'label_id_'+str(matidx+1).zfill(3)),'wb'))




if __name__ == '__main__':

    clsObj = trjClusteringFromAdj()

    for matidx in range(len(clsObj.adjmatfiles)):
        print "clustering trj based on adj truncation ", matidx
        clsObj.trjclustering(matidx)

        if isSave:
            clsObj.saveLabel(matidx)
        if isVisualize:
            clsObj.visLabel(matidx)














