import numpy as np
from sklearn.utils import check_random_state
# import copy
import numpy.linalg as np_lg
import numpy.matlib as np_mat
import sklearn
from scipy import linalg
from scipy.sparse import *
from sklearn import mixture
from sklearn.cluster import *
from sklearn.manifold import *
from sklearn.cluster import AffinityPropagation
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

        """if provide adj"""
        self.embedding_ = sklearn.manifold.spectral_embedding(adjacency = self.adjacency, n_components=self.n_dimension, eigen_solver='arpack',
                                             random_state=random_state) * 1000  
        
        """use the graph laplacian to calculate embedding"""
        """this is really too slow!!!"""
        # from numpy import linalg as LA
        # graph_lap = csgraph.laplacian(self.adjacency, normed=False)
        # eigVl, eigVc = LA.eigh(csr_matrix(graph_lap).toarray())
        # # np.allclose(self.embedding_,eigVc[:,:2])
        # self.embedding_  = eigVc[:,:self.n_dimension]

        """use the newer version of SpectralEmbedding"""
        # model = sklearn.manifold.SpectralEmbedding(n_components=self.n_dimension, affinity ='precomputed', gamma=None, random_state=None, eigen_solver=None, n_neighbors=None)
        # self.embedding_ = model.fit_transform(self.adjacency)

        """if provide the raw data"""
        # model = sklearn.manifold.SpectralEmbedding(n_components=50, affinity='nearest_neighbors', gamma=None, random_state=None, eigen_solver=None, n_neighbors=None)
        # self.embedding_ = model.fit_transform(data_sampl_*feature_)

        """locally linear embedding_"""
        # model = sklearn.manifold.LocallyLinearEmbedding(n_neighbors=5, n_components=self.n_dimension, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, 
        #     method='standard', hessian_tol=0.0001, modified_tol=1e-12, neighbors_algorithm='auto', random_state=None)
        # self.embedding_ = model.fit_transform(data_sampl_*feature_)

    def clustering_DPGMM(self, n_components, alpha):
        model = mixture.DPGMM(n_components=n_components, alpha=alpha, n_iter=1000)
        model.fit(self.embedding_)
        label = model.predict(self.embedding_)
        return label


    def clustering_Affini_prpoga(self):
        # af = AffinityPropagation(preference=-50, affinity ='precomputed').fit(self.adjacency)
        af = AffinityPropagation(affinity ='precomputed').fit(self.adjacency)
        pdb.set_trace()
        assert af.affinity_matrix_ == self.adjacency

        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        return labels


    def clustering_spectral(self, num_cluster):  #Ncut chenge
        model = sklearn.cluster.SpectralClustering(n_clusters=num_cluster,affinity='precomputed')
        label = model.fit_predict(self.adjacency)  # no embedding
        # knnmodel = KMeans(num_cluster)
        # label2 = knnmodel.fit_predict(self.embedding_)
        return label

    def get_adjacency(self, adjacency):
        self.adjacency = adjacency

    def get_embedding(self, embedding_):
        self.embedding_ = embedding_

    def clustering_kmeans(self, num_cluster):
        model = KMeans(num_cluster)
        model.fit(self.embedding_)
        label = model.predict(self.embedding_)
        return label



