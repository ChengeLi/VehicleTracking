import numpy as np
import scipy as sp
import scipy.io as scipy_io
import sklearn,copy,glob,pdb,itertools
import matplotlib.pyplot as plt
from sklearn import (manifold, datasets, decomposition, ensemble, lda,
                     random_projection,mixture)
from sklearn.manifold import *
from sklearn.utils import check_random_state
from scipy import linalg
import matplotlib as mpl
import numpy.linalg as np_lg
import numpy.matlib as np_mat
from scipy.sparse import *
from sklearn.cluster import *
from scipy.io import savemat

class sparse_subspace_clustering:
    def __init__(self,lambd = 10,dataset = np.random.randn(100),n_dimension = 100,random_state= None):
        """dataset :[sample,feature] """
        self.lambd = lambd
        self.dataset = dataset
        self.random_state = random_state
        self.n_dimension = n_dimension
    def construct_adjacency(self):
        self.adjacency = np.zeros([self.dataset.shape[0],self.dataset.shape[0]])
        adjacency = np.zeros([self.dataset.shape[0],self.dataset.shape[0]])
        for i in range(self.dataset.shape[0]):
            
            clf = sklearn.linear_model.Lasso(self.lambd)
#            pdb.set_trace()
            temp_Y = self.dataset[i,:]
            temp_X = np.zeros(self.dataset.shape)
            for j in range(self.dataset.shape[0]):
                if i !=j:
                    temp_X[j,:] = self.dataset[j,:]
            clf.fit(temp_X.T,temp_Y)
            adjacency[i,:]= clf.sparse_coef_.todense()
            print clf.sparse_coef_.indices.size
            print 'set %d'%i
        self.adjacency = np.abs( adjacency)

    def construct_adjacency_non_fix_len(self):
        """samples not dimension aligned"""
        self.adjacency = np.zeros([self.dataset.shape[0],self.dataset.shape[0]])
        adjacency = np.zeros([self.dataset.shape[0],self.dataset.shape[0]])

        for i in range(self.dataset.shape[0]):
            print i
            idx = np.where(self.dataset[i,:]!=0)[0]
            temp_Y = self.dataset[i,idx]/np_lg.norm(self.dataset[i,idx])
            temp_X = np.zeros([self.dataset.shape[0]+idx.size,idx.size])
            temp_X_norm = 1.0/(np_lg.norm(self.dataset[:,idx],axis = 1)+np.power(10,-10))
            temp_X_norm[np.where(np.isinf(temp_X_norm))[0]] = 0
            temp_X[0:self.dataset.shape[0],:] = self.dataset[:,idx]*np.transpose(np_mat.repmat(temp_X_norm,idx.size,1))
            temp_X[i,:] = np.zeros(idx.size)
            temp_X[self.dataset.shape[0]:self.dataset.shape[0]+idx.size,:]= np.diag(np.ones(idx.size)*0.1)

            clf = sklearn.linear_model.Lasso(1/np.power(idx.size,0.5)/1000)
            clf.fit(temp_X.T*100,temp_Y*100)
#            pdb.set_trace()
            adjacency[i,:]= clf.sparse_coef_.todense()[:,0:self.dataset.shape[0]]
        self.adjacency = np.abs( adjacency +np.transpose(adjacency))
    
    
    
    
    def manifold(self):
        random_state = check_random_state(self.random_state)
#        pdb.set_trace()
        self.embedding_ = spectral_embedding(self.adjacency,n_components=self.n_dimension,eigen_solver='arpack',random_state=random_state)*1000
    def clustering(self,n_components,alpha):
        model = mixture.DPGMM(n_components=n_components,alpha=alpha,n_iter = 1000)
#        pdb.set_trace()

        model.fit(self.embedding_)
        self.label = model.predict(self.embedding_)
        return self.label, model
    def get_adjacency(self,adjacency):
        self.adjacency = adjacency
    def get_embedding(self,embedding_):
        self.embedding_ = embedding_
    def clustering_connected(self,threshold,min_sample_cluster,alpha):
        temp_mat = copy.copy(self.adjacency)
        temp_mat[np.where(temp_mat<threshold)[0],np.where(temp_mat<threshold)[1]]= 0
        pdb.set_trace()
        n_components, c_temp =csgraph.connected_components(csr_matrix(temp_mat))
        print n_components
        labels = np.zeros(c_temp.size)
        for i in np.unique(c_temp):
            sub_index = np.where(c_temp==i)[0]
            sub_matrix = temp_mat[sub_index][:,sub_index]
            if sub_index.size >3:
                project_dimension = int(np.floor(sub_index.size/20)+1)
                ssc = sparse_subspace_clustering(2000000,temp_mat,n_dimension = project_dimension)
                ssc.get_adjacency(sub_matrix)
                ssc.manifold()
                sub_labels,model = ssc.clustering(n_components=int(np.floor(sub_index.size/min_sample_cluster)+1),alpha= alpha)
            #        visulize(ssc.embedding_,sub_labels,model)
                labels[sub_index] = np.max(labels) + sub_labels
                print 'number of trajectory %s'%sub_labels.size + '  unique labels %s' % np.unique(sub_labels).size
            else:
                sub_labels = np.ones(sub_index.size)
                labels[sub_index] = np.max(labels) + sub_labels
                print 'number of trajectory %s'%sub_labels.size + '  unique labels %s' % np.unique(sub_labels).size
        j = 0
        labels_new = np.zeros(labels.shape)
        unique_array =np.unique(labels)
        for i in range(unique_array.size):
            labels_new[np.where(labels == unique_array[i])] = j
            j = j+1
        labels = labels_new
        return labels
    def clustering_kmeans(self,num_cluster):
        model = KMeans(num_cluster)
        model.fit(self.embedding_)
        self.label = model.predict(self.embedding_)
        return self.label
    
def visulize(data,labels,clf):
    color_iter =itertools.cycle(['r', 'g', 'b', 'c', 'm'])
    for i, (mean, covar, color) in enumerate(zip(clf.means_, clf._get_covars(), color_iter)):
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(labels == i):
            continue
        plt.scatter(data[labels == i, 0], data[labels== i, 1], .8, color=color)
        
        # Plot an ellipse to show the Gaussian component
        #    plt.xlim(-6, 4 * np.pi - 6)
#    plt.ylim(-5, 5)
#    plt.title(title)
    plt.xticks(())
    plt.yticks(())
    plt.show()



if __name__ == '__main__':
#    
#    """With constructed adjacency matrix """

    inifilename = 'HR'
    matfiles = sorted(glob.glob('./mat/adj/'+inifilename+'*.mat'))

    for matidx,matfile in enumerate(matfiles):

        file = scipy_io.loadmat(matfile)
        feature =(file['adj'] > 0).astype('float')
        #pdb.set_trace()
        c = file['c']
        mask = file['mask']
        labels = np.zeros(c.size)
        for i in np.unique(c):
            print i
            sub_index = np.where(c==i)[1]
            sub_matrix = feature[sub_index][:,sub_index]
            if sub_index.size >3:
                project_dimension = int(np.floor(sub_index.size/20)+1)
                ssc = sparse_subspace_clustering(2000000,feature,n_dimension = project_dimension)
                ssc.get_adjacency(sub_matrix)
                ssc.manifold()
                sub_labels,model = ssc.clustering(n_components=int(np.floor(sub_index.size/2)+1),alpha= 0.1)
                #            sub_labels = ssc.clustering_kmeans(int(np.floor(sub_index.size/4)+1))
                #        visulize(ssc.embedding_,sub_labels,model)
                labels[sub_index] = np.max(labels) + (sub_labels+1)
                print 'number of trajectory %s'%sub_labels.size + '  unique labels %s' % np.unique(sub_labels).size
            else:
                sub_labels = np.ones(sub_index.size)
                labels[sub_index] = np.max(labels) + sub_labels
                print 'number of trajectory %s'%sub_labels.size + '  unique labels %s' % np.unique(sub_labels).size
        j = 0
        labels_new = np.zeros(labels.shape)
        unique_array =np.unique(labels)

        for i in range(unique_array.size):
            labels_new[np.where(labels == unique_array[i])] = j
            j = j+1
        labels = labels_new
            #pdb.set_trace()
            #print 1
    
        labelsave ={}
        labelsave['label']=labels
        labelsave['mask']=mask
        savename = './mat/labels/'+inifilename+'_label_'+str(matidx+1).zfill(3)

        savemat(savename,labelsave)


    """SSC clustering Andy Project"""
#    file = scipy_io.loadmat('ptsTrj_remove_station')
#    x = file['x']
#    y = file['y']
#    mask = file['A']
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
