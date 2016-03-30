import pdb
import os
import copy
import glob
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as np_lg
import numpy.matlib as np_mat
from scipy.io import loadmat,savemat
import sklearn
from scipy import linalg
from scipy.sparse import *
from sklearn import mixture
from sklearn.cluster import *
from sklearn.manifold import *
from sklearn.utils import check_random_state
import itertools
import pickle as pickle
from scipy.sparse import csr_matrix


from DataPathclass import *
DataPathobj = DataPath(dataSource,VideoIndex)
from parameterClass import *
Parameterobj = parameter(dataSource,VideoIndex)


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

        """if provide adj"""
        self.embedding_ = sklearn.manifold.spectral_embedding(self.adjacency, n_components=self.n_dimension, eigen_solver='arpack',
                                             random_state=random_state) * 1000  
        
        """use the graph laplacian to calculate embedding"""
        """this is really too slow!!!"""
        # from numpy import linalg as LA
        # graph_lap = csgraph.laplacian(self.adjacency, normed=False)
        # eigVl, eigVc = LA.eigh(csr_matrix(graph_lap).toarray())
        # # np.allclose(self.embedding_,eigVc[:,:2])
        # self.embedding_  = eigVc[:,:self.n_dimension]

        """if provide the raw data"""
        # model = sklearn.manifold.SpectralEmbedding(n_components=2, affinity='rbf', gamma=None, random_state=None, eigen_solver=None, n_neighbors=None)
        # model = sklearn.manifold.SpectralEmbedding(n_components=50, affinity='nearest_neighbors', gamma=None, random_state=None, eigen_solver=None, n_neighbors=None)
        # self.embedding_ = model.fit_transform(data_sampl_*feature_)

        """locally linear embedding_"""
        # model = sklearn.manifold.LocallyLinearEmbedding(n_neighbors=5, n_components=self.n_dimension, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, 
        #     method='standard', hessian_tol=0.0001, modified_tol=1e-12, neighbors_algorithm='auto', random_state=None)
        # self.embedding_ = model.fit_transform(data_sampl_*feature_)

    def clustering_DPGMM(self, n_components, alpha):
        model = mixture.DPGMM(n_components=n_components, alpha=alpha, n_iter=1000)
        model.fit(self.embedding_)
        self.label = model.predict(self.embedding_)
        # pdb.set_trace()
        return self.label

    def get_adjacency(self, adjacency):
        self.adjacency = adjacency

    def get_embedding(self, embedding_):
        self.embedding_ = embedding_

    def clustering_kmeans(self, num_cluster):
        model = KMeans(num_cluster)
        model.fit(self.embedding_)
        # self.label = model.predict(self.embedding_)
        label = model.predict(self.embedding_)
        return label


    def clustering_spectral(self, num_cluster):  #Ncut chenge
        model = sklearn.cluster.SpectralClustering(n_clusters=num_cluster,affinity='precomputed')
        # self.label = model.fit_predict(self.adjacency)  # no embedding
        label = model.fit_predict(self.adjacency)  # no embedding

        # knnmodel = KMeans(num_cluster)
        # label2 = knnmodel.fit_predict(self.embedding_)
        return label

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

def uniqulizeLabel(labels):
    j = 0
    labels_new = np.zeros(labels.shape)
    unique_array = np.unique(labels[labels!=-222])
    for i in range(unique_array.size):
        labels_new[np.where(labels == unique_array[i])] = j
        j = j + 1
    return labels_new



def ssc_with_Adj_CC(trjAdj,CClabel,trjID):
    small_connected_comp = []
    # one_car_trjID = pickle.load(open('./johnson_one_car_trjID','rb'))

    labels_DPGMM = np.ones(CClabel.size)*(-222)
    labels_spectral = np.ones(CClabel.size)*(-222)

    color_choice = np.array([np.random.randint(0, 255) for _ in range(3 * int(CClabel.size))]).reshape(int(CClabel.size), 3)
    print 'connected component number:',len(np.unique(CClabel))

    # FeatureMtx = pickle.load(open('./NGSIM_FeatureMtx', 'rb'))
    FeatureMtx = pickle.load(open('./Johnson00115_FeatureMtx', 'rb'))
    FeatureMtx[np.isnan(FeatureMtx)] = 0
    
    # FeatureMtxLoc = []
    # for aa in one_car_trjID:
    #     FeatureMtxLoc+=list(np.where(trjID[0]==aa)[0])

    for i in np.unique(CClabel):
        color = ((color_choice[i].T) / 255.)
        # print "connected component No. " ,str(i)
        sub_index = np.where(CClabel == i)[1]  # noted, after saving to Mat, got appened zeros, should use [1] instead of [0]
        sub_adjMtx = trjAdj[sub_index][:, sub_index]
        # sub_FeatureMtx = FeatureMtx[sub_index,:]
        sub_FeatureMtx = []
        if sub_index.size > 3:
            project_dimension = int(np.floor(sub_index.size/Parameterobj.embedding_projection_factor) + 1)
            print "CC size:", sub_index.size
            print "project dimension is: ", str(project_dimension)  ## embeded lower dimension
            ssc = sparse_subspace_clustering(2000000, sub_FeatureMtx, n_dimension=project_dimension)
            ssc.get_adjacency(sub_adjMtx)
            ssc.manifold()
            """DPGMM"""
            print 'DPGMM n_components =', int(np.floor(sub_index.size/Parameterobj.DPGMM_num_component_shirink_factor))
            sub_labels_DPGMM = ssc.clustering_DPGMM(n_components=int(np.floor(sub_index.size/Parameterobj.DPGMM_num_component_shirink_factor)), alpha=Parameterobj.DPGMM_alpha)
            sub_adjMtx = csr_matrix(sub_adjMtx, shape=sub_adjMtx.shape).toarray()
            sub_adjMtx_rearrange = np.zeros(sub_adjMtx.shape)
            # arrange_index = []
            # lastLen = 0
            # for ii in np.unique(sub_labels_DPGMM):
            #     arrange_index = arrange_index+ list(np.where(sub_labels_DPGMM==ii)[0])
                # thisLen = np.sum(sub_labels_DPGMM==ii)
                # sub_adjMtx_rearrange[lastLen:lastLen+thisLen,:][:,lastLen:lastLen+thisLen] = sub_adjMtx[sub_labels_DPGMM==ii,:][:,sub_labels_DPGMM==ii]
                # lastLen = lastLen+thisLen

            # if sub_adjMtx.shape[0]>10:
            #     plt.figure();
            #     plt.title('after rearrangement');
            #     plt.imshow(sub_adjMtx[arrange_index,:][:,arrange_index])
            #     plt.draw()

            #     plt.figure()
            #     plt.title('original sub_adjMtx')
            #     plt.imshow(sub_adjMtx)
            #     plt.draw()
            #     pdb.set_trace()


            # num_cluster_prior = len(np.unique(sub_labels_DPGMM))
            num_cluster_prior = int(np.floor(sub_index.size/Parameterobj.DPGMM_num_component_shirink_factor))
            # visulize(ssc.embedding_,sub_labels,model,color)
            """k-means"""
            # sub_labels_k_means = ssc.clustering_kmeans(num_cluster_prior)
            """N cut spectral"""
            # pdb.set_trace()
            print 'spectral clustering num_cluster is', num_cluster_prior
            # sub_labels_spectral = ssc.clustering_spectral(num_cluster_prior)
            sub_labels_spectral = ssc.clustering_spectral(int(np.floor(sub_index.size/Parameterobj.DPGMM_num_component_shirink_factor)))

            # arrange_index = []
            # for ii in np.unique(sub_labels_spectral):
            #     arrange_index = arrange_index+ list(np.where(sub_labels_spectral==ii)[0])
            # if sub_adjMtx.shape[0]>10:
            #     plt.figure();
            #     plt.title('after rearrangement');
            #     plt.imshow(sub_adjMtx[arrange_index,:][:,arrange_index])
            #     plt.draw()

            #     plt.figure()
            #     plt.title('original sub_adjMtx')
            #     plt.imshow(sub_adjMtx)
            #     plt.draw()
            labels_DPGMM[sub_index] = max(np.max(labels_DPGMM),0) + (sub_labels_DPGMM + 1)
            labels_spectral[sub_index] = max(np.max(labels_spectral),0) + (sub_labels_spectral + 1)
            # print 'number of trajectory in this connected components %s' % sub_labels.size + '  unique labels %s' % np.unique(
            #         sub_labels).size
        else:  ## if size small, treat as one group
            sub_labels = np.ones(sub_index.size)
            labels_DPGMM[sub_index] = max(np.max(labels_DPGMM),0) + sub_labels
            labels_spectral[sub_index] = max(np.max(labels_spectral),0) + sub_labels
            small_connected_comp.append(sub_index)
            # print 'number of trajectory %s'%sub_labels.size + '  unique labels %s' % np.unique(sub_labels).size
        
    labels_DPGMM = uniqulizeLabel(labels_DPGMM)
    labels_spectral = uniqulizeLabel(labels_spectral)
    
    # plt.figure()
    # plt.plot(sub_labels_spectral,'g')
    # plt.plot(labels_new,'r')
    # pdb.set_trace()
    return trjID, labels_DPGMM, labels_spectral, small_connected_comp





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
    trjAdj = (file['adj'] > 0).astype('float')  ## adj mtx
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
                sub_adjMtx = trjAdj[sub_index][:, sub_index]
                ssc = sparse_subspace_clustering(lambd=2000000, dataset=trjAdj, n_dimension=project_dimension)
                ssc.get_adjacency(sub_adjMtx)

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


def prepare_input_data():
    global savePath
    if Parameterobj.useWarpped:      
        adjmatfiles = sorted(glob.glob(os.path.join(DataPathobj.adjpath,'usewarpped_*.mat')))
    else:
        adjmatfiles = sorted(glob.glob(os.path.join(DataPathobj.adjpath,'*.mat')))
    savePath = DataPathobj.sscpath
    trjmatfiles = sorted(glob.glob(os.path.join(DataPathobj.smoothpath,'klt*.mat')))

    adjmatfiles = adjmatfiles[0:]
    trjmatfiles = trjmatfiles[0:]
    return adjmatfiles, trjmatfiles, savePath


def eigen_decomp(adj):
    from numpy import linalg as LA
    w, v = LA.eigvalsh(adj)



if __name__ == '__main__':
    """With constructed adjacency matrix """
    adjmatfiles, trjmatfiles, savePath = prepare_input_data()
    isSave      = True
    isVisualize = False

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
            """ andy's method, not real sparse sc, just spectral clustering"""
            trjID, labels_DPGMM,labels_spectral,small_connected_comp = ssc_with_Adj_CC(trjAdj,CClabel,trjID)
            """ construct adj use ssc"""
            # trjID,labels, adj = sscConstructedAdj_CC(adjfile)

            """ construct adj use ssc, with Neighbour adj as constraint"""
            # trjID,labels = sscAdj_inNeighbour(adjfile)

            labelsave['labels_DPGMM'+DirName[dirii]] = labels_DPGMM
            labelsave['labels_spectral'+DirName[dirii]] = labels_spectral
            labelsave['trjID'+DirName[dirii]] = trjID

        if isSave:
            print "saving the labels..."
            if Parameterobj.useWarpped:
                savename = os.path.join(savePath,'usewarpped_'+str(matidx+1).zfill(3))
            else:
                savename = os.path.join(savePath,str(matidx+1).zfill(3))
            savemat(savename, labelsave)


        if isVisualize:
            labels = labels_DPGMM
            # labels = labels_spectral
            # visualize different classes for each Connected Component
            """  use the x_re and y_re from adj mat files  """
            trjfile = loadmat(trjmatfiles[matidx])
            if Parameterobj.useWarpped:
                xtrj = csr_matrix(trjfile['xtracks_warpped'], shape=trjfile['xtracks_warpped'].shape).toarray()
                ytrj = csr_matrix(trjfile['ytracks_warpped'], shape=trjfile['ytracks_warpped'].shape).toarray()
            else:
                xtrj = csr_matrix(trjfile['xtracks'], shape=trjfile['xtracks'].shape).toarray()
                ytrj = csr_matrix(trjfile['ytracks'], shape=trjfile['ytracks'].shape).toarray()


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
            pdb.set_trace()
            pickle.dump(label_id,open(os.path.join(savePath,'label_id_'+str(matidx+1).zfill(3)),'wb'))

        


