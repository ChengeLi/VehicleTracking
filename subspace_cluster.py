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
import pdb

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
        # self.adjacency = np.abs( adjacency)
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

    def clustering_connected(self, threshold, min_sample_cluster, alpha):
        temp_mat = copy.copy(self.adjacency)
        temp_mat[np.where(temp_mat < threshold)[0], np.where(temp_mat < threshold)[1]] = 0
        n_components, c_temp = csgraph.connected_components(csr_matrix(temp_mat))
        print n_components
        labels = np.zeros(c_temp.size)
        for i in np.unique(c_temp):
            sub_index = np.where(c_temp == i)[0]
            sub_matrix = temp_mat[sub_index][:, sub_index]
            if sub_index.size > 3:
                project_dimension = int(np.floor(sub_index.size / 20) + 1)
                ssc = sparse_subspace_clustering(2000000, temp_mat, n_dimension=project_dimension)
                ssc.get_adjacency(sub_matrix)
                ssc.manifold()
                sub_labels, model = ssc.clustering_DPGMM(
                        n_components=int(np.floor(sub_index.size / min_sample_cluster) + 1), alpha=alpha)
                print "project dimension is: ", str(project_dimension)
                #        visulize(ssc.embedding_,sub_labels,model)
                labels[sub_index] = np.max(labels) + sub_labels
                print 'number of trajectory %s' % sub_labels.size + '  unique labels %s' % np.unique(sub_labels).size
            else:
                sub_labels = np.ones(sub_index.size)
                labels[sub_index] = np.max(labels) + sub_labels
                print 'number of trajectory %s' % sub_labels.size + '  unique labels %s' % np.unique(sub_labels).size
        j = 0
        labels_new = np.zeros(labels.shape)
        unique_array = np.unique(labels)
        for i in range(unique_array.size):
            labels_new[np.where(labels == unique_array[i])] = j
            j = j + 1
        labels = labels_new
        return labels

    def clustering_kmeans(self, num_cluster):
        model = KMeans(num_cluster)
        model.fit(self.embedding_)
        self.label = model.predict(self.embedding_)
        return self.label


def visulize(data, labels, clf, color):
    # color_iter =itertools.cycle(['r', 'g', 'b', 'c', 'm'])
    # for i, (mean, covar, color) in enumerate(zip(clf.means_, clf._get_covars(), color_iter)):
    for i, (mean, covar) in enumerate(zip(clf.means_, clf._get_covars())):
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(labels == i):
            continue
        if data.shape[1] >= 2:
            plt.scatter(data[labels == i, 0], data[labels == i, 1], 8, color=tuple(color))
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


def ssc_with_Adj_CC(file):
    small_connected_comp = []
    feature = (file['adj'] > 0).astype('float')  ## adj mtx
    CClabel = file['c']  # labels from connected Component
    trjID = file['trjID']
    labels = np.zeros(CClabel.size)
    color_choice = np.array([np.random.randint(0, 255) for _ in range(3 * int(CClabel.size))]).reshape(int(CClabel.size), 3)
    for i in np.unique(CClabel):
        color = ((color_choice[i].T) / 255.)
        # print "connected component No. " ,str(i)
        sub_index = np.where(CClabel == i)[
            1]  # noted, after saving to Mat, got appened zeros, should use [1] instead of [0]
        sub_matrix = feature[sub_index][:, sub_index]
        if sub_index.size > 3:
            project_dimension = int(np.floor(sub_index.size / 100) + 1)
            # project_dimension = int(np.floor(sub_index.size/50)+1)
            # project_dimension = 2
            print "project dimension is: ", str(project_dimension)
            ssc = sparse_subspace_clustering(2000000, feature, n_dimension=project_dimension)
            ssc.get_adjacency(sub_matrix)
            ssc.manifold()
            # sub_labels,model = ssc.clustering_DPGMM(n_components=int(np.floor(sub_index.size/2)+1),alpha=0.1)
            # sub_labels111, model111 = ssc.clustering_DPGMM(n_components=int(np.floor(sub_index.size / 4) + 1), alpha=0.1)
            # sub_labels222, model222 = ssc.clustering_DPGMM(n_components=int(np.floor(sub_index.size / 4) + 1), alpha=1)
            # sub_labels333, model333 = ssc.clustering_DPGMM(n_components=int(np.floor(sub_index.size / 4) + 1), alpha=10)
            # sub_labels444, model444 = ssc.clustering_DPGMM(n_components=int(np.floor(sub_index.size / 4) + 1), alpha=0.01)
            # sub_labels, model = ssc.clustering_DPGMM(n_components=int(np.floor(sub_index.size / 4) + 1), alpha=0.05)
            sub_labels, model = ssc.clustering_DPGMM(n_components=int(np.floor(sub_index.size / 4) + 1), alpha=0.001)


            # pdb.set_trace()
            # visulize(ssc.embedding_,sub_labels,model,color)

            #            sub_labels = ssc.clustering_kmeans(int(np.floor(sub_index.size/4)+1))
            #        visulize(ssc.embedding_,sub_labels,model)
            labels[sub_index] = np.max(labels) + (sub_labels + 1)
            # print sub_labels  ## not always start from 0?? 
            print 'number of trajectory in this connected components %s' % sub_labels.size + '  unique labels %s' % np.unique(
                    sub_labels).size
        else:  ## if size small, treat as one group
            sub_labels = np.ones(sub_index.size)
            labels[sub_index] = np.max(labels) + sub_labels
            small_connected_comp.append(sub_index)
            # print 'number of trajectory %s'%sub_labels.size + '  unique labels %s' % np.unique(sub_labels).size
    j = 0
    labels_new = np.zeros(labels.shape)
    unique_array = np.unique(labels)

    for i in range(unique_array.size):
        labels_new[np.where(labels == unique_array[i])] = j
        j = j + 1
    labels = labels_new
    return trjID, labels, small_connected_comp


def sscConstructedAdj_CC(file):  # use ssc to construct adj, use any samples except the sample itself
    xtrj = file['x_re']
    ytrj = file['y_re']
    xspd = file['xspd']
    yspd = file['yspd']
    trjID = file['trjID']
    dataFeature = np.concatenate((xtrj, xspd, ytrj, yspd), axis=1)
    # dataFeature       = np.concatenate((xtrj,ytrj), axis = 1)
    project_dimension = int(np.floor(dataFeature.shape[0] / 10) + 1)  # assume per group has max 10 trjs????
    ssc = sparse_subspace_clustering(lambd=2000, dataset=dataFeature, n_dimension=project_dimension)
    ssc.construct_adjacency()
    adj = ssc.adjacency
    ssc.manifold()
    labels = ssc.clustering_connected(threshold=0, min_sample_cluster=50, alpha=0.1)
    return trjID, labels, adj


def sscAdj_inNeighbour(file):  ## use neighbour adj as prior, limiting ssc's adj choice to be within neighbours
    xtrj = file['x_re']
    ytrj = file['y_re']
    xspd = file['xspd']
    yspd = file['yspd']
    trjID = file['trjID']
    feature = (file['adj'] > 0).astype('float')  ## adj mtx
    CClabel = file['c']  # labels from connected Component

    # dataFeature = np.concatenate((xtrj,xspd,ytrj,yspd), axis = 1)
    dataFeature = np.concatenate((xtrj, ytrj), axis=1)

    labels = np.zeros(CClabel.size)
    adj = np.zeros((CClabel.size, CClabel.size))
    for i in np.unique(CClabel):
        print i
        sub_index = np.where(CClabel == i)[1]
        if sub_index.size > 3:
            if sub_index.size < 100:  # if connected component too big, just use the binary adj
                project_dimension = int(np.floor(sub_index.size / 100) + 1)
                ssc = sparse_subspace_clustering(lambd=2000, dataset=dataFeature[sub_index, :],
                                                 n_dimension=project_dimension)
                ssc.construct_adjacency()
            else:
                project_dimension = int(np.floor(sub_index.size / 100) + 1)
                sub_matrix = feature[sub_index][:, sub_index]
                ssc = sparse_subspace_clustering(lambd=2000000, dataset=feature, n_dimension=project_dimension)
                ssc.get_adjacency(sub_matrix)

            """adj assignment not successful!!! whyyyyyyyy"""
            # adjInNeighbour = ssc.adjacency
            # adj[sub_index][:,sub_index]= adjInNeighbour[:][:] 


            ssc.manifold()
            sub_labels, model = ssc.clustering(n_components=int(np.floor(sub_index.size / 2) + 1), alpha=0.1)
            # sub_labels = ssc.clustering_kmeans(int(np.floor(sub_index.size/4)+1))
            # visulize(ssc.embedding_,sub_labels,model)
            labels[sub_index] = np.max(labels) + (sub_labels + 1)
            print 'number of trajectory in this connected components%s' % sub_labels.size + '  unique labels %s' % np.unique(
                    sub_labels).size
        else:
            sub_labels = np.ones(sub_index.size)
            labels[sub_index] = np.max(labels) + sub_labels
            print 'number of trajectory %s' % sub_labels.size + '  unique labels %s' % np.unique(sub_labels).size
    j = 0
    labels_new = np.zeros(labels.shape)
    unique_array = np.unique(labels)

    for i in range(unique_array.size):
        labels_new[np.where(labels == unique_array[i])] = j
        j = j + 1
    labels = labels_new

    return trjID, labels


def prepare_input_data(isAfterWarpping, isLeft, dataSource, usingLinux=True):
    global savePath
    if isAfterWarpping:
        if isLeft:
            loadPath = '../DoT/CanalSt@BaxterSt-96.106/leftlane/adj/'
            matfiles = sorted(glob.glob(loadPath + 'warpped_Adj_' + '*.mat'))
            savePath = '../DoT/CanalSt@BaxterSt-96.106/leftlane/sscLabels/'
        else:
            loadPath = '../DoT/CanalSt@BaxterSt-96.106/rightlane/adj/'
            matfiles = sorted(glob.glob(loadPath + 'warpped_Adj_' + '*.mat'))
            savePath = '../DoT/CanalSt@BaxterSt-96.106/rightlane/sscLabels/'
    else:
        # matfiles = sorted(glob.glob('./mat/20150222_Mat/adj/'+'HR'+'*.mat'))
        # matfiles = sorted(glob.glob('./mat/20150222_Mat/adj/'+'HR'+'_adj_withT_'+'*.mat'))
        # savePath = './mat/20150222_Mat/labels/'+'HR'+'_label_'
        # savePath = './mat/20150222_Mat/labels/'+'HR'+'_label_withT_'

        # matfiles = sorted(glob.glob('../DoT/5Ave@42St-96.81/adj/5Ave@42St-96.81_2015-06-16_16h04min40s686ms/' +'*.mat'))
        # savePath = '../DoT/5Ave@42St-96.81/labels/5Ave@42St-96.81_2015-06-16_16h04min40s686ms/' 
        if dataSource == 'DoT':
            # for linux
            matfiles = sorted(glob.glob(
                    '/media/My Book/CUSP/AIG/DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/adj/Adj_' + '*.mat'))
            savePath = '/media/My Book/CUSP/AIG/DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/ssc/ssc_'
            # for mac
            # matfiles = sorted(glob.glob('../DoT/CanalSt@BaxterSt-96.106/adj/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/Adj_' +'*.mat'))
            # savePath = '../DoT/CanalSt@BaxterSt-96.106/labels/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/ssc_'

        if dataSource == 'Johnson':
            # Jay & Johnson
            if usingLinux:
                matfiles = sorted(glob.glob('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/adj/' + '*.mat'))
                savePath = '/media/My Book/CUSP/AIG/Jay&Johnson/roi2/ssc/ssc_'
            else:
                matfiles = sorted(glob.glob('../Jay&Johnson/roi2/adj/' + '*.mat'))
                savePath = '../Jay&Johnson/roi2/ssc/ssc_'

        matfiles = matfiles[0:]
    return matfiles, savePath


if __name__ == '__main__':
    # def ssc_main(dataSource):
    dataSource = 'Johnson'
    """With constructed adjacency matrix """
    isAfterWarpping = False
    isLeft = False
    matfiles, savePath = prepare_input_data(isAfterWarpping, isLeft, dataSource, usingLinux=False)
    isSave = True
    isVisualize = True

    for matidx, matfile in enumerate(matfiles):
        file = scipy_io.loadmat(matfile)
        """ andy's method, not real sparse sc, just spectral clustering"""
        trjID, labels,small_connected_comp = ssc_with_Adj_CC(file)
        """ construct adj use ssc"""
        # trjID,labels, adj = sscConstructedAdj_CC(file)

        """ construct adj use ssc, with Neighbour adj as constraint"""
        # trjID,labels = sscAdj_inNeighbour(file)

        if isVisualize:
            # visualize different classes seperated by SSC for each Connected Component
            """  use the x_re and y_re from adj mat files  """
            xtrj = file['xtracks']
            ytrj = file['ytracks']

            color = np.array([np.random.randint(0, 255) for _ in range(3 * int(max(labels) + 1))]).reshape(int(max(labels) + 1), 3)
            fig999 = plt.figure()
            # plt.ion()
            ax = plt.subplot(1, 1, 1)

            newtrjID   = list(trjID[0])
            newlabels = list(labels)
            for i in range(int(max(labels)) + 1):
                trjind = np.where(labels == i)[0]
                print "trjind = ", str(trjind)
                if len(trjind) <= 20:
					# newlabels.remove(i)
					newlabels = [x for x in newlabels if x!=i]
					newtrjID = [x for x in newtrjID if x not in trjID[0][trjind]]
					# pdb.set_trace()
					continue ## skip the rest, don't draw too short trjs

     #            for jj in range(len(trjind)):
     #                startlimit = np.min(np.where(xtrj[trjind[jj],:]!=0))
     #                endlimit   = np.max(np.where(xtrj[trjind[jj],:]!=0))
     #                # lines = ax.plot(x_re[trjind[jj],startlimit:endlimit], y_re[trjind[jj],startlimit:endlimit],color = (0,1,0),linewidth=2)
     #                lines = ax.plot(xtrj[trjind[jj],startlimit:endlimit], ytrj[trjind[jj],startlimit:endlimit],color = (color[i-1].T)/255.,linewidth=2)
                   
     
					# # fig999.canvas.draw()
					# # plt.pause(0.0001)

     #            # plt.text(np.median(xtrj[trjind[jj],startlimit:endlimit]), np.median(ytrj[trjind[jj],startlimit:endlimit]), str(trjind))

     #        # im = plt.imshow(np.zeros([528,704,3])) 
     #        fig999.canvas.draw()
     #        name = '../Jay&Johnson/roi2/ssc/'+str(matidx).zfill(3)+'.jpg'
     #        plt.savefig(name)

        if isSave:
            print "saving the labels..."
            labelsave = {}
            # labelsave['label']   = np.array(newlabels)
            labelsave['label'] = labels
            labelsave['trjID'] = trjID
            labelsave['newlabel'] = newlabels
            labelsave['newtrjID'] = newtrjID
            savename = savePath + matfiles[matidx][-7:-4].zfill(3)
            savemat(savename, labelsave)

    """SSC clustering Andy Project"""
    #    file = scipy_io.loadmat('ptsTrj_remove_station')
    #    x = file['x']
    #    y = file['y']
    #    trjID = file['A']
    #    feature = np.concatenate([x,y],axis = 1)
    #    project_dimension = 20
    #    ssc = sparse_subspace_clustering(1,feature,n_dimension = project_dimension)
    #    ssc.construct_adjacency_non_fix_len()
    #    #    pdb.set_trace()
    #
    #    scipy_io.savemat('remove_station_adj',{'adjacency':ssc.adjacency})
    #
    #    ssc.manifold()
    #    #    label, = ssc.clustering(n_components=3000,alpha=1)
    #    #    label = ssc.clustering_connected(threshold =0.1,min_sample_cluster=3,alpha=0.1)
    #    label = ssc.clustering_kmeans(300)
    #    pdb.set_trace()
    #    print 1
    """ SSC clustering human seizure dataset"""
##    file = scipy_io.loadmat('/Users/songyilin/Documents/ieeg-matlab-1.8.3/Study_026_analysis/fft_feature')
##    feature = file['fft_feature']
##    project_dimension = 20
##    ssc = sparse_subspace_clustering(np.power(10.0,-4),feature,n_dimension = project_dimension)
##    ssc.construct_adjacency()
##    pdb.set_trace()
##    label = ssc.clustering_connected(threshold=0,min_sample_cluster=10,alpha=1)
##    print 1
#    file = scipy_io.loadmat('/Users/songyilin/Documents/ieeg-matlab-1.8.3/Study_026_analysis/ssc_seizure_adjacency')
#    adjacency = file['adj']
#    project_dimension = 20
#    ssc = sparse_subspace_clustering(1,adjacency,n_dimension = project_dimension)
#    ssc.get_adjacency(adjacency)
#    ssc.manifold()
#    labels,model = ssc.clustering(n_components=50,alpha=1)
#    pdb.set_trace()
#    print 1
