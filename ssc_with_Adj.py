import numpy as np
from sparse_subspace_clusteringClass import sparse_subspace_clustering
from numpy import linalg as LA
import pdb

import itertools
from scipy.sparse import csr_matrix

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def eigen_decomp(adj):
    w, v = LA.eigvalsh(adj)


def uniqulizeLabel(labels):
    j = 0
    labels_new = np.zeros(labels.shape)
    unique_array = np.unique(labels[labels!=-222])
    for i in range(unique_array.size):
        labels_new[np.where(labels == unique_array[i])] = j
        j = j + 1
    return labels_new

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
    # plt.xticks(())ls 
    # plt.yticks(())
    plt.show()


def visAdj_rearrange(sub_adjMtx,sub_labels):
    plt.figure();
    sub_adjMtx_rearrange = np.zeros(sub_adjMtx.shape)
    arrange_index = []
    lastLen = 0
    for ii in np.unique(sub_labels):
        arrange_index = arrange_index+ list(np.where(sub_labels==ii)[0])
        thisLen = np.sum(sub_labels==ii)
        sub_adjMtx_rearrange[lastLen:lastLen+thisLen,:][:,lastLen:lastLen+thisLen] = sub_adjMtx[sub_labels==ii,:][:,sub_labels==ii]
        lastLen = lastLen+thisLen

    if sub_adjMtx.shape[0]>10:
        plt.subplot(121)
        plt.title('original sub_adjMtx')
        plt.imshow(sub_adjMtx)
        plt.draw()
        plt.subplot(122)
        plt.title('after rearrangement');
        plt.imshow(sub_adjMtx[arrange_index,:][:,arrange_index])
        plt.draw()

def ssc_with_Adj_CC(trjAdj,CClabel,trjID,Parameterobj):
    small_CC_oneCls = []

    labels = np.ones(CClabel.size)*(-222)

    print 'connected component number:',len(np.unique(CClabel))

    """data is featureMtx or adj????"""
    # FeatureMtx = pickle.load(open('./NGSIM_FeatureMtx', 'rb'))
    # FeatureMtx = pickle.load(open('./Johnson00115_FeatureMtx', 'rb'))
    # FeatureMtx[np.isnan(FeatureMtx)] = 0
    
    # color_choice = np.array([np.random.randint(0, 255) for _ in range(3 * int(np.unique(CClabel).size))]).reshape(int(np.unique(CClabel).size), 3)
    # for i in np.unique(CClabel):
    for ind in range(len(np.unique(CClabel))):
        i = np.unique(CClabel)[ind]
        # color = ((color_choice[ind].T) / 255.)
        sub_index = np.where(CClabel == i)[1]  # noted, after saving to Mat, got appened zeros, should use [1] instead of [0]
        sub_adjMtx = trjAdj[sub_index][:, sub_index]
        # sub_FeatureMtx = FeatureMtx[sub_index,:]
        sub_FeatureMtx = []

        if sub_index.size > Parameterobj.smallclssize and sub_index.size < 50:
            # project_dimension = int(np.floor(sub_index.size/Parameterobj.embedding_projection_factor) + 1)
            project_dimension = 10

            print "CC size:", sub_index.size
            """restrict the prj dim <=200, otherwise too slow"""
            print "project dimension is: ", min(int(np.sqrt(sub_index.size)), project_dimension)  ## embeded lower dimension
            ssc = sparse_subspace_clustering(2000000, sub_FeatureMtx, n_dimension=min(int(np.sqrt(sub_index.size)), project_dimension))
            sub_adjMtx = csr_matrix(sub_adjMtx, shape=sub_adjMtx.shape).toarray()
            ssc.get_adjacency(sub_adjMtx)
            
            if Parameterobj.clustering_choice == 'labels_DPGMM_':
                ssc.manifold()
                """DPGMM"""
                # n_components_DPGMM = int(np.floor(sub_index.size/Parameterobj.DPGMM_num_component_shirink_factor))
                # n_components_DPGMM = max(1,n_components_DPGMM)
                n_components_DPGMM = sub_index.size
                sub_labels_DPGMM = ssc.clustering_DPGMM(n_components=n_components_DPGMM, alpha=Parameterobj.DPGMM_alpha)
                # visulize(ssc.embedding_,sub_labels,model,color)
                """vis rearrangement of the adj"""
                # visAdj_rearrange(sub_adjMtx,sub_labels_DPGMM)
                labels[sub_index] = max(np.max(labels),0) + (sub_labels_DPGMM + 1)

            elif Parameterobj.clustering_choice == 'labels_spectral_':
                """k-means"""
                # num_cluster_prior = len(np.unique(sub_labels_DPGMM))
                # num_cluster_prior = n_components_DPGMM
                # sub_labels_k_means = ssc.clustering_kmeans(num_cluster_prior)

                ###################################################################################
                """N cut spectral"""
                # n_components_spectral = int(np.floor(sub_index.size/Parameterobj.spectral_num_component_shirink_factor))
                # n_components_spectral = max(2,n_components_spectral)
                # n_components_spectral = min(sub_index.size,n_components_spectral)
                n_components_spectral = 2
                print 'spectral clustering num_cluster is', n_components_spectral
                sub_labels_spectral = ssc.clustering_spectral(n_components_spectral)
                # visAdj_rearrange(sub_adjMtx,sub_labels_spectral)                
                labels[sub_index] = max(np.max(labels),0) + (sub_labels_spectral + 1)

            elif Parameterobj.clustering_choice == 'labels_affinity_':
                """affinity propogation"""
                sub_labels_affinity = ssc.clustering_Affini_prpoga()
                labels[sub_index] = max(np.max(labels),0) + (sub_labels_affinity + 1)

        else:  ## if size small, treat as one group
            sub_labels = np.ones(sub_index.size)
            labels[sub_index] = max(np.max(labels),0) + sub_labels
            small_CC_oneCls.append(sub_index)
        
    labels = uniqulizeLabel(labels)
    return labels, small_CC_oneCls




def sscConstructedAdj_CC(DataPathobj):  # use ssc to construct adj, use concatenated raw data????
    """fix me....??? :/"""
    # if len(sorted(glob.glob(DataPathobj.adjpath+'extremeValue'+self.DirName[directionInd]+str(matidx+1).zfill(3))))>0:
    #     print "mean_std_ForKernel(ignore) and extremeValue already stored, load..."
    #     self.extremeValue = pickle.load(open(DataPathobj.adjpath+'extremeValue'+self.DirName[directionInd]+str(matidx+1).zfill(3),'rb'))


    if len(sorted(glob.glob(self.savePath+feaName+self.DirName[directionInd]+str(matidx+1).zfill(3))))>0:
        print "distance diff already stored, load..."
        self.feature_diff_tensor = pickle.load(open(self.savePath+feaName+self.DirName[directionInd]+str(matidx+1).zfill(3),'rb'))

    xtrj = self.feature_diff_tensor[:,:,0]
    ytrj = file['y_re']
    xspd = file['xspd']
    yspd = file['yspd']
    trjID = file['trjID']
    dataFeature = np.concatenate((xtrj, xspd, ytrj, yspd), axis=1)
    project_dimension = int(np.floor(dataFeature.shape[0] / 10) + 1)  # assume per group has max 10 trjs????
    ssc = sparse_subspace_clustering(lambd=2000, dataset=dataFeature, n_dimension=project_dimension)
    ssc.construct_adjacency()
    adj = ssc.adjacency
    ssc.manifold()
    labels = ssc.clustering_connected(threshold=0, min_sample_cluster=50, alpha=0.1)
    return trjID, labels, adj


def sscAdj_inNeighbour(file):  ## use neighbour adj as prior, limiting ssc's adj choice to be within neighbours
    """fixe me..... :/"""

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

