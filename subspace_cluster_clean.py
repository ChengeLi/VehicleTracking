import pdb
import os
import copy
import glob
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as np_lg
import numpy.matlib as np_mat
import scipy.io as scipy_io
import sklearn
from scipy import linalg
from scipy.io import savemat
from scipy.sparse import *
from sklearn import (mixture)
from sklearn.cluster import *
from sklearn.manifold import *
from sklearn.utils import check_random_state
import itertools



from DataPathclass import *
DataPathobj = DataPath(VideoIndex)

class sparse_subspace_clustering:
    def __init__(self, lambd=10, dataset=np.random.randn(100), n_dimension=100, random_state=None):
        """dataset :[sample,feature] """
        self.lambd = lambd
        self.dataset = dataset
        self.random_state = random_state
        self.n_dimension = n_dimension

    def construct_adjacency(self):
        self.adjacency = np.zeros([self.dataset.shape[0], self.dataset.shape[0]])
        adjacency = np.zeros([self.dataset.shape[0], self.dataset.shape[0]])
        for i in range(self.dataset.shape[0]):

            clf = sklearn.linear_model.Lasso(self.lambd)
            temp_Y = self.dataset[i, :]
            temp_X = np.zeros(self.dataset.shape)
            for j in range(self.dataset.shape[0]):
                if i != j:
                    temp_X[j, :] = self.dataset[j, :]
            clf.fit(temp_X.T, temp_Y)
            adjacency[i, :] = clf.sparse_coef_.todense()
            print clf.sparse_coef_.indices.size
            print 'set %d' % i
        # self.adjacency = np.abs(adjacency)
        # want it to be symmetric
        self.adjacency = np.abs(adjacency + np.transpose(adjacency))

    def construct_adjacency_non_fix_len(self):
        """samples not dimension aligned"""
        self.adjacency = np.zeros([self.dataset.shape[0], self.dataset.shape[0]])
        adjacency = np.zeros([self.dataset.shape[0], self.dataset.shape[0]])

        for i in range(self.dataset.shape[0]):
            print i
            idx = np.where(self.dataset[i, :] != 0)[0]
            temp_Y = self.dataset[i, idx] / np_lg.norm(self.dataset[i, idx])
            temp_X = np.zeros([self.dataset.shape[0] + idx.size, idx.size])
            temp_X_norm = 1.0 / (np_lg.norm(self.dataset[:, idx], axis=1) + np.power(10, -10))
            temp_X_norm[np.where(np.isinf(temp_X_norm))[0]] = 0
            temp_X[0:self.dataset.shape[0], :] = self.dataset[:, idx] * np.transpose(
                    np_mat.repmat(temp_X_norm, idx.size, 1))
            temp_X[i, :] = np.zeros(idx.size)
            temp_X[self.dataset.shape[0]:self.dataset.shape[0] + idx.size, :] = np.diag(np.ones(idx.size) * 0.1)

            clf = sklearn.linear_model.Lasso(1 / np.power(idx.size, 0.5) / 1000)
            clf.fit(temp_X.T * 100, temp_Y * 100)
            adjacency[i, :] = clf.sparse_coef_.todense()[:, 0:self.dataset.shape[0]]
        self.adjacency = np.abs(adjacency + np.transpose(adjacency))

    def manifold(self):
        random_state = check_random_state(self.random_state)
        self.embedding_ = spectral_embedding(self.adjacency, n_components=self.n_dimension, eigen_solver='arpack',
                                             random_state=random_state) * 1000

    def clustering_DPGMM(self, n_components, alpha):
        model = mixture.DPGMM(n_components=n_components, alpha=alpha, n_iter=1000)
        model.fit(self.embedding_)
        self.label = model.predict(self.embedding_)
        return self.label, model

    def get_adjacency(self, adjacency):
        self.adjacency = adjacency

    def get_embedding(self, embedding_):
        self.embedding_ = embedding_

    def clustering_kmeans(self, num_cluster):
        model = KMeans(num_cluster)
        model.fit(self.embedding_)
        self.label = model.predict(self.embedding_)
        return self.label
    
    def clustering_spectral(self, num_cluster):  #Ncut chenge
        model = SpectralClustering(n_clusters=num_cluster)
        self.label = model.fit_predict(self.adjacency)  # no embedding
        return self.label


def visulize(data, labels, clf, colors):
    
    color_iter =itertools.cycle(['r', 'g', 'b', 'c', 'm'])

    for i, (mean, covar, color) in enumerate(zip(clf.means_, clf._get_covars(), color_iter)):
    # for i, (mean, covar) in enumerate(zip(clf.means_, clf._get_covars())):
        ww, vv = linalg.eigh(covar) # eigen values, vectors
        uu = vv[0] / linalg.norm(vv[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(labels == i):
            continue
        if data.shape[1] >= 3:
            fig = plt.figure()
            plt.scatter(data[labels == i, 1], data[labels == i, 2],8, color=tuple(color))
            # from mpl_toolkits.mplot3d.axes3d import Axes3D #fix me?
            # ax = fig.add_subplot(111,projection='3d')
            # ax = Axes3D(fig) # error!!!!!!! 
            # ax.scatter(data[labels == i, 0], data[labels == i, 1],data[labels == i, 2],color=tuple(color))
            fig.canvas.draw()
        else:
            pass
            # print "projected to one dimension."
            # plt.scatter(range(data[labels == i, 0].shape[0]),data[labels == i, 0], 8, color=color)

            # Plot an ellipse to show the Gaussian component
            #    plt.xlim(-6, 4 * np.pi - 6)
            #    plt.ylim(-5, 5)
            #    plt.title(title)
    # plt.xticks(())
    # plt.yticks(())
    plt.show()


def ssc_with_Adj_CC(file, useBinaryAdj = False):
    small_connected_comp = []
    if useBinaryAdj:
        feature = (file['adj'] > 0).astype('float')  
    else:
        feature = (file['adj']).astype('float')  
    CClabel = file['c']  # labels from connected Component
    trjID = file['trjID']
    labels = np.ones(CClabel.size)*(-222)
    color_choice = np.array([np.random.randint(0, 255) for _ in range(3 * int(CClabel.size))]).reshape(int(CClabel.size), 3)
    print 'np.unique(CClabel):',np.unique(CClabel)
    for i in np.unique(CClabel):
        color = ((color_choice[i].T) / 255.)
        # print "connected component No. " ,str(i)
        sub_index = np.where(CClabel == i)[1]  # noted, after saving to Mat, got appened zeros, should use [1] instead of [0]
        sub_matrix = feature[sub_index][:, sub_index]
        if sub_index.size > 3:
            project_dimension = int(np.floor(sub_index.size / 100) + 1)
            print "project dimension is: ", str(project_dimension)
            ssc = sparse_subspace_clustering(2000000, feature, n_dimension=project_dimension)
            ssc.get_adjacency(sub_matrix)
            ssc.manifold()
            'DPGMM'
            sub_labels_DPGMM, model = ssc.clustering_DPGMM(n_components=int(np.floor(sub_index.size / 4) + 1), alpha=0.001)
            num_cluster_prior = len(np.unique(sub_labels_DPGMM))
            # visulize(ssc.embedding_,sub_labels,model,color)
            'k-means'
            # sub_labels_k_means = ssc.clustering_kmeans(num_cluster_prior)
            'N cut spectral'
            # sub_labels_spectral = ssc.clustering_spectral(num_cluster_prior)

            sub_labels = sub_labels_DPGMM
            labels[sub_index] = max(np.max(labels),0) + (sub_labels + 1)
            print 'number of trajectory in this connected components %s' % sub_labels.size + '  unique labels %s' % np.unique(
                    sub_labels).size
        else:  ## if size small, treat as one group
            sub_labels = np.ones(sub_index.size)
            labels[sub_index] = max(np.max(labels),0) + sub_labels
            small_connected_comp.append(sub_index)
            # print 'number of trajectory %s'%sub_labels.size + '  unique labels %s' % np.unique(sub_labels).size
    j = 0
    labels_new = np.zeros(labels.shape)
    unique_array = np.unique(labels[labels!=-222])
    for i in range(unique_array.size):
        labels_new[np.where(labels == unique_array[i])] = j
        j = j + 1
    labels = labels_new
    # plt.figure()
    # plt.plot(sub_labels_spectral,'g')
    # plt.plot(labels_new,'r')
    # pdb.set_trace()
    return trjID, labels, small_connected_comp


def prepare_input_data():
    if smooth:
        matfiles = sorted(glob.glob(os.path.join(DataPathobj.adjpath,'smooth_*.mat')))
    else:
        matfiles = sorted(glob.glob(os.path.join(DataPathobj.adjpath,'*.mat'))) 
    savePath = DataPathobj.sscpath
    matfiles = matfiles[0:]
    return matfiles, savePath


if __name__ == '__main__':

    dataSource = 'DoT'
    """With constructed adjacency matrix """
    matfiles, savePath = prepare_input_data()
    isSave      = False
    isVisualize = True
    global smooth = True
    for matidx, matfile in enumerate(matfiles):
        pdb.set_trace()
        file = scipy_io.loadmat(matfile)
        """ andy's method, not real sparse sc, just spectral clustering"""
        trjID, labels,small_connected_comp = ssc_with_Adj_CC(file)

        if isVisualize:
            # visualize different classes for each Connected Component
            """  use the x_re and y_re from adj mat files  """
            xtrj = file['xtracks']
            ytrj = file['ytracks']

            color = np.array([np.random.randint(0, 255) for _ in range(3 * int(max(labels) + 1))]).reshape(int(max(labels) + 1), 3)
            fig999 = plt.figure()
            plt.ion()
            ax = plt.subplot(1, 1, 1)

            newtrjID  = list(trjID[0])
            newlabels = list(labels)
            for i in range(int(max(labels)) + 1):
                trjind = np.where(labels == i)[0]
                # print "trjind = ", str(trjind)
                if len(trjind) <= 20:
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
                   
     
        if isSave:
            print "saving the labels..."
            labelsave = {}
            labelsave['label'] = labels
            labelsave['trjID'] = trjID
            labelsave['newlabel'] = newlabels
            labelsave['newtrjID'] = newtrjID
            if smooth:
                savename =os.path.join(savePath,'smooth_'+matfiles[matidx][-7:-4].zfill(3))
            else:
                savename = os.path.join(savePath, + matfiles[matidx][-7:-4].zfill(3))
            savemat(savename, labelsave)

