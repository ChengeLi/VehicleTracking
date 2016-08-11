#### embedding 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from mpl_toolkits.mplot3d import Axes3D

class embeddings(obj):
    def __init__(self, model,data):
        self.modelChoice = model
        self.data = data
        # self.data = FeatureMtx_norm

    def PCA_embedding(self,n_components):
        print 'PCA projecting...'
        self.pca = PCA(n_components= n_components,whiten=False)
        self.embedding_ = self.model.fit(data)

        # self.pca = PCAembedding(self.data,50)
        # FeatureAfterPCA = self.pca.transform(self.data)

    def TSNE_embedding(self,n_components):
        # tsne = TSNE(n_components=2, perplexity=30.0)
        tsne3 = TSNE(n_components=n_components, perplexity=30.0)

        # tsne_data = tsne.fit_transform(FeatureAfterPCA50) 
        tsne3_data = tsne3.fit_transform(FeatureAfterPCA50) 
        # pickle.dump(tsne_data,open(DataPathobj.DataPath+'/tsne_data.p','wb'))
        # tsne_data = pickle.load(open(DataPathobj.DataPath+'/tsne_data.p','rb'))
        self.embedding_ = tsne3_data

    def MDS_embedding(self,n_components):
        self.mds = MDS(n_components=n_components, max_iter=100, n_init=1)
        MDS_data = self.mds.fit_transform(FeatureAfterPCA50)

    def LLE_embedding(self):
        """locally linear embedding_"""
        # self.lle = sklearn.manifold.LocallyLinearEmbedding(n_neighbors=5, n_components=self.n_dimension, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, 
        #     method='standard', hessian_tol=0.0001, modified_tol=1e-12, neighbors_algorithm='auto', random_state=None)
        # self.embedding_ = self.lle.fit_transform(data_sampl_*feature_)

        """use DPGMM or Spectral labels"""
        sscfile = loadmat(DataPathobj.sscpath+'001.mat')
        labels_DPGMM = csr_matrix(sscfile['labels_DPGMM_upup'], shape=sscfile['labels_DPGMM_upup'].shape).toarray()
        labels_spectral = csr_matrix(sscfile['labels_spectral_upup'], shape=sscfile['labels_spectral_upup'].shape).toarray()
        trjID = csr_matrix(sscfile['trjID_upup'], shape=sscfile['trjID_upup'].shape).toarray()

        """use connected_components labels"""
        adjfile = loadmat(DataPathobj.adjpath+'20knn&thresh_Gaussian_diff_dir_001.mat')
        labels_CC = csr_matrix(adjfile['c_upup'], shape=adjfile['c_upup'].shape).toarray()

        """use fake ground truth labels"""
        arrange_index = pickle.load(open(DataPathobj.DataPath+'/arrange_index.p','rb'))

        # labels_fakeGT = labels_CC[arrange_index]
        labels_fakeGT = np.zeros_like(labels_CC)
        for ii in range(0,int(labels_fakeGT.shape[1]/20),1):
            labels_fakeGT[0,arrange_index[20*ii:min(20*(ii+1),labels_fakeGT.shape[1])]] = ii
            # labels_fakeGT[0,5*ii:min(5*(ii+1),labels_fakeGT.shape[1])] = ii


    def visEmbedding(self):
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # labels = labels_DPGMM
        # labels = labels_spectral
        # labels = labels_CC
        labels = labels_fakeGT
        # data = MDS_data
        data = tsne_data

        clustered_color = np.array([np.random.randint(0,255) for _ in range(3*int(len(np.unique(labels))))]).reshape(len(np.unique(labels)),3)
        plt.figure()
        for ii in range(labels.shape[1]):
            plt.scatter(data[ii,0],data[ii,1],color=(clustered_color[int(labels[0,ii])].T/255.0))
        plt.draw()








