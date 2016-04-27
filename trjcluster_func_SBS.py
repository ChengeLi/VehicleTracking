import os
import cv2
import math
import pdb
import pickle
import numpy as np
import glob as glob
from scipy.sparse import csr_matrix
from scipy.io import loadmat,savemat
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from DataPathclass import *
DataPathobj = DataPath(dataSource,VideoIndex)
from parameterClass import *
Parameterobj = parameter(dataSource,VideoIndex)

def adj_GTind(fully_adj,trjID):
    rearrange_adj = np.zeros_like(fully_adj)
    matidx = 0
    global_annolist = pickle.load(open(DataPathobj.DataPath+'/'+str(matidx).zfill(3)+'global_annolist.p','rb'))
    _, idx = np.unique(global_annolist, return_index=True)
    unique_anno = np.array(global_annolist)[np.sort(idx)]

    arrange_index = []
    for ii in range(fully_adj.shape[0]):
        arrange_index = arrange_index+ list(np.where(trjID==unique_anno[ii])[0])

    rearrange_adj =  fully_adj[arrange_index,:][:,arrange_index]
    plt.figure()
    plt.imshow(rearrange_adj)
    pdb.set_trace()
    pickle.dump(arrange_index,open(DataPathobj.DataPath+'/arrange_index.p','wb'))

    return rearrange_adj


def knn_graph(fully_adj, knn = 8):
    """construct KNN graph from the fully connected graph"""
    # knn_adj = np.zeros_like(fully_adj)
    knn_adj = []
    if knn < fully_adj.shape[0]:
        for i in range(0, len(fully_adj)):
            knn_adj.append([])
            row = fully_adj[i][:]
            row = sorted(row,reverse=True)
            row = row[:knn]
            for j in range(0, len(fully_adj[i])):
                if fully_adj[i][j] in row:
                    knn_adj[i].append(fully_adj[i][j])
                else:
                    knn_adj[i].append(0)
    else:
        knn_adj = fully_adj
    return np.array(knn_adj)



def standard_scaler_normalization(data_feature_mtx):
    """normalize across all samples for each feature"""
    scaler = StandardScaler().fit(data_feature_mtx)
    # pickle.dump(scaler,open('./clf/scaler','wb'))
    data_feature_mtx = scaler.transform(data_feature_mtx)

    """normalize across all features for each sample"""
    # scaler = StandardScaler().fit(data_feature_mtx.T)
    # pickle.dump(scaler,open('./clf/scaler_across_all_fea','wb'))
    # data_feature_mtx = scaler.transform(data_feature_mtx.T).T
    return data_feature_mtx


"""test for PCA"""
from sklearn.decomposition import PCA
def PCAembedding(data,n_components):
    print 'PCA projecting...'
    pca = PCA(n_components= n_components,whiten=False)
    pca.fit(data)
    return pca

def TwoD_Emedding(FeatureMtx_norm):
    # pca = PCAembedding(FeatureMtx_norm,200)
    # FeatureAfterPCA = pca.transform(FeatureMtx_norm)

    # pca3 = PCAembedding(FeatureMtx_norm,3)
    # FeatureAfterPCA3 = pca3.transform(FeatureMtx_norm)

    # pca50 = PCAembedding(FeatureMtx_norm,50)
    # FeatureAfterPCA50 = pca50.transform(FeatureMtx_norm)
    # pickle.dump(FeatureAfterPCA50,open(DataPathobj.DataPath+'/FeatureAfterPCA50.p','wb'))
    FeatureAfterPCA50 = pickle.load(open(DataPathobj.DataPath+'/FeatureAfterPCA50.p','rb'))

    pdb.set_trace()
    from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    from sklearn.manifold import TSNE, MDS
    # tsne = TSNE(n_components=2, perplexity=30.0)
    tsne3 = TSNE(n_components=3, perplexity=30.0)

    # tsne_data = tsne.fit_transform(FeatureAfterPCA50) 
    tsne3_data = tsne3.fit_transform(FeatureAfterPCA50) 
    # pickle.dump(tsne_data,open(DataPathobj.DataPath+'/tsne_data.p','wb'))
    # tsne_data = pickle.load(open(DataPathobj.DataPath+'/tsne_data.p','rb'))

    # mds = MDS(n_components=2, max_iter=100, n_init=1)
    # MDS_data = mds.fit_transform(FeatureAfterPCA50)


    """locally linear embedding_"""
    # model = sklearn.manifold.LocallyLinearEmbedding(n_neighbors=5, n_components=self.n_dimension, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, 
    #     method='standard', hessian_tol=0.0001, modified_tol=1e-12, neighbors_algorithm='auto', random_state=None)
    # self.embedding_ = model.fit_transform(data_sampl_*feature_)

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

    pdb.set_trace()
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


    pdb.set_trace()


def getRawDataFeatureMtx(dataForKernel):
    if len(dataForKernel)==7:
        [x,y,xspd,yspd,hue,fg_blob_center_X,fg_blob_center_Y] = dataForKernel
    FeatureMtx = np.hstack((x,y,xspd,yspd))
    for ii in range(4, dataForKernel.shape[0]):
        if len(dataForKernel[ii])>0:
            FeatureMtx = np.hstack((FeatureMtx,dataForKernel[ii]))
    FeatureMtx[FeatureMtx==0]=np.nan
    FeatureMtx_norm = np.zeros(FeatureMtx.shape)
    for ii in range(len(dataForKernel)):
        maxV = np.nanmax(FeatureMtx[:,ii*Parameterobj.trunclen:(ii+1)*Parameterobj.trunclen])
        minV = np.nanmin(FeatureMtx[:,ii*Parameterobj.trunclen:(ii+1)*Parameterobj.trunclen])
        FeatureMtx_norm[:,ii*Parameterobj.trunclen:(ii+1)*Parameterobj.trunclen] = (FeatureMtx[:,ii*Parameterobj.trunclen:(ii+1)*Parameterobj.trunclen]-minV)/(maxV-minV)*1

    FeatureMtx_norm[np.isnan(FeatureMtx_norm)] = 0
    # pickle.dump(FeatureMtx, open('./Johnson00115_FeatureMtx','wb'))
    # pickle.dump(FeatureMtx_norm, open('./Johnson00115_FeatureMtx_normalized','wb'))  
    return FeatureMtx, FeatureMtx_norm



def getMuSigma(dataForKernel):
    if len(dataForKernel)==7:
        [x,y,xspd,yspd,hue,fg_blob_center_X,fg_blob_center_Y] = dataForKernel

    sxdiffAll    = []
    sydiffAll    = []
    mdisAll      = []
    huedisAll    = []
    centerDisAll = []

    for i in range(x.shape[0]):
        # print "i", i
        # for j in range(i, min(x.shape[0],i+1000)):
        for j in range(i, x.shape[0]):
            tmp1 = x[i,:]!=0
            tmp2 = x[j,:]!=0
            idx  = num[tmp1&tmp2]
            
            if len(idx)>5: # has overlapping
                # print len(idx)
            # if len(idx)>=30: # at least overlap for 100 frames
                # sidx   = idx[0:-1] # for speed
                # sxdiff = np.mean(np.abs(xspd[i,sidx]-xspd[j,sidx]))
                # sydiff = np.mean(np.abs(yspd[i,sidx]-yspd[j,sidx]))                    
                # # mdis   = np.mean(np.abs(x[i,idx]-x[j,idx])+np.abs(y[i,idx]-y[j,idx])) #mahhattan distance
                # mdis = np.mean(np.sqrt((x[i,idx]-x[j,idx])**2+(y[i,idx]-y[j,idx])**2))
                sidx   = idx[0:-1] # for speed
                sxdiff = np.max(np.abs(xspd[i,sidx]-xspd[j,sidx]))
                sydiff = np.max(np.abs(yspd[i,sidx]-yspd[j,sidx]))                    
                # mdis   = np.max(np.abs(x[i,idx]-x[j,idx])+np.abs(y[i,idx]-y[j,idx])) #mahhattan distance
                mdis = np.max(np.sqrt((x[i,idx]-x[j,idx])**2+(y[i,idx]-y[j,idx])**2))


                if Parameterobj.useSBS:
                    cxi = np.nanmedian(fg_blob_center_X[i,idx])
                    cyi = np.nanmedian(fg_blob_center_Y[i,idx])
                    cxj = np.nanmedian(fg_blob_center_X[j,idx])
                    cyj = np.nanmedian(fg_blob_center_Y[j,idx])

                    if np.isnan(cxi) or np.isnan(cyi): # if not inside a blob, use trj's own center
                        cxi = np.nanmedian(x[i,idx])
                        cyi = np.nanmedian(y[i,idx])
                    if np.isnan(cxj) or np.isnan(cyj):
                        cxj = np.nanmedian(x[j,idx])
                        cyj = np.nanmedian(y[j,idx])

                    centerDis = np.sqrt((cxi-cxj)**2+(cyi-cyj)**2)
                    # huedis = np.mean(np.abs(hue[i,sidx]-hue[j,sidx]))
                    huedis = np.max(np.abs(hue[i,sidx]-hue[j,sidx]))

                else:
                    centerDis = np.nan
                    huedis = np.nan
                
                sxdiffAll.append(sxdiff)
                sydiffAll.append(sydiff)
                mdisAll.append(mdis)
                centerDisAll.append(centerDis)
                huedisAll.append(huedis)

    # n, bins, patches = plt.hist(sxdiffAll, 100, normed=1, facecolor='green', alpha=0.75)
    # n, bins, patches = plt.hist(sydiffAll, 100, normed=1, facecolor='green', alpha=0.75)
    # n, bins, patches = plt.hist(mdisAll, 100, normed=1, facecolor='green', alpha=0.75)
    # n, bins, patches = plt.hist(centerDisAll, 100, normed=1, facecolor='green', alpha=0.75)
    # n, bins, patches = plt.hist(huedisAll, 100, normed=1, facecolor='green', alpha=0.75)
    # plt.draw()
    # plt.show()

    # pickle.dump(sxdiffAll,open('./sxdiffAll_johnson','wb'))
    # pickle.dump(sydiffAll,open('./sydiffAll_johnson','wb'))
    # pickle.dump(mdisAll,open('./mdisAll_johnson','wb'))
    # pickle.dump(centerDisAll, open('./centerDisAll_johnson','wb'))
    # pickle.dump(huedisAll,open('./huedisAll_johnson','wb'))
    # pdb.set_trace()

        # plt.plot(fg_blob_center_X[i,:][fg_blob_center_X[i,:]!=0],fg_blob_center_Y[i,:][fg_blob_center_X[i,:]!=0],'b')
        # plt.plot(fg_blob_center_X[i,idx],fg_blob_center_Y[i,idx],'g')
        # plt.plot(cxi,cyi,'r')
        # plt.draw()

    """fit Gaussian to find mu and sigma"""
    # mu_xspd_diff,sigma_xspd_diff = fitGaussian(sxdiffAll)
    # mu_yspd_diff,sigma_yspd_diff = fitGaussian(sydiffAll)
    # mu_spatial_distance,sigma_spatial_distance = fitGaussian(mdisAll)
    # try:
    #     mu_hue_distance,sigma_hue_distance = fitGaussian(huedisAll)
    #     mu_center_distance,sigma_center_distance = fitGaussian(centerDisAll)
    # except: #if empty
    #     (mu_hue_distance,sigma_hue_distance) = (np.nan, np.nan)
    #     (mu_center_distance,sigma_center_distance) = (np.nan, np.nan)

    # mean_std_ForKernel = np.array([mu_xspd_diff,sigma_xspd_diff,mu_yspd_diff,sigma_yspd_diff,mu_spatial_distance,sigma_spatial_distance,mu_hue_distance,sigma_hue_distance,mu_center_distance,sigma_center_distance])
    mean_std_ForKernel = []

    extremeValue = np.array([min(sxdiffAll[:]),max(sxdiffAll[:]), min(sydiffAll[:]),max(sydiffAll[:]),min(mdisAll[:]),max(mdisAll[:]),min(huedisAll[:]),max(huedisAll[:]),min(centerDisAll[:]),max(centerDisAll[:])])
    return mean_std_ForKernel, extremeValue




def normalize_features(dataForKernel_ele,mean_std_ForKernel,extremeValue,useSBS):

    [sxdiff,sydiff,mdis,huedis,centerDis] = dataForKernel_ele


    """normalize"""
    """use extreame values to normalize"""
    [min_sx,max_sx,min_sy,max_sy,min_mdis,max_mdis,min_hue,max_hue,min_center,max_center]=extremeValue
    # sxdiff_normalized    = (sxdiff-min_sx)/(max_sx-min_sx)
    # sydiff_normalized    = (sydiff-min_sy)/(max_sy-min_sy)
    # mdis_normalized      = (mdis-min_mdis)/(max_mdis-min_mdis)
    # huedis_normalized    = (huedis-min_hue)/(max_hue-min_hue)
   
    """within each thresholded CC, using the threshold to normalize"""
    sxdiff_normalized = (sxdiff)/Parameterobj.nullXspd_for_adj
    sydiff_normalized = (sydiff)/Parameterobj.nullYspd_for_adj
    mdis_normalized   = (mdis)/Parameterobj.nullDist_for_adj
    huedis_normalized = (huedis)  # hue can be ignored, weight = 0 anyway

    if useSBS:
        # centerDis_normalized = (centerDis-min_center)/(max_center-min_center)
        centerDis_normalized = (centerDis)/Parameterobj.nullBlob_for_adj
    else:
        centerDis_normalized = 0
    return sxdiff_normalized,sydiff_normalized,mdis_normalized,huedis_normalized,centerDis_normalized


def get_thresholding_adj(adj,feature_diff_tensor):
    """fix me....?"""
    [sxdiff_normalized,sydiff_normalized,mdis_normalized,huedis_normalized,centerDis_normalized]=feature_diff_tensor
    """before normalization"""
    # try:
    #     yspdth = mean_std_ForKernel[2]+mean_std_ForKernel[3] #mean+sigma
    #     xspdth = mean_std_ForKernel[0]+mean_std_ForKernel[1]
    # except:
    #     yspdth = 0.4*extremeValue[3] #if mean is empty
    #     xspdth = 0.4*extremeValue[1]

    """after normalization"""
    xspdth = 0.5;
    yspdth = 0.5;
    adj = np.ones((Nsample,Nsample))
    adj = adj*(sxdiff_normalized <xspdth )*(sydiff_normalized<yspdth)
    adj[np.isnan(adj)] = 0
    adj = adj + adj.transpose() 
    return adj

def get_gaussian_adj(adj,feature_diff_tensor,sameDirTrjID, afterNormalize = True):
    
    # bigValuePlaceHolder = 1e10
    # feature_diff_tensor[np.isnan(feature_diff_tensor)] = bigValuePlaceHolder
    # feature_diff_tensor_symmetric = np.zeros_like(feature_diff_tensor)
    # feature_diff_tensor_symmetric[:,:,0] = feature_diff_tensor[:,:,0]+feature_diff_tensor[:,:,0].T
    # feature_diff_tensor_symmetric[:,:,1] = feature_diff_tensor[:,:,1]+feature_diff_tensor[:,:,1].T
    # feature_diff_tensor_symmetric[:,:,2] = feature_diff_tensor[:,:,2]+feature_diff_tensor[:,:,2].T
    # feature_diff_tensor_symmetric[:,:,3] = feature_diff_tensor[:,:,3]+feature_diff_tensor[:,:,3].T
    # feature_diff_tensor_symmetric[:,:,4] = feature_diff_tensor[:,:,4]+feature_diff_tensor[:,:,4].T


    """assign different weights to different features"""
    weight = Parameterobj.adj_weight
    if afterNormalize:
        """if loaded from normalized feature tensor:"""
        sxdiff_normalized    = feature_diff_tensor[:,:,0]
        sydiff_normalized    = feature_diff_tensor[:,:,1]
        mdis_normalized      = feature_diff_tensor[:,:,2]
        huedis_normalized    = feature_diff_tensor[:,:,3]
        centerDis_normalized = feature_diff_tensor[:,:,4]
    else:
        """if loaded from unnormalized feature tensor:"""
        sxdiff_normalized    = feature_diff_tensor[:,:,0]/Parameterobj.nullXspd_for_adj
        sydiff_normalized    = feature_diff_tensor[:,:,1]/Parameterobj.nullYspd_for_adj
        mdis_normalized      = feature_diff_tensor[:,:,2]/Parameterobj.nullDist_for_adj
        huedis_normalized    = feature_diff_tensor[:,:,3] ##ignore
        centerDis_normalized = feature_diff_tensor[:,:,4]/Parameterobj.nullBlob_for_adj

    # sigma_xspd_diff = 0.7
    # sigma_yspd_diff = 0.7
    # sigma_spatial_distance = 200
    # [mu_xspd_diff,sigma_xspd_diff,mu_yspd_diff,sigma_yspd_diff,mu_spatial_distance,sigma_spatial_distance,mu_hue_distance,sigma_hue_distance, mu_center_distance,sigma_center_distance] =     mean_std_ForKernel 

    # adj_element = np.exp((-sxdiff**2/(2*sigma_xspd_diff**2)+(-sydiff**2/(2*sigma_yspd_diff**2))+(-mdis**2/(2*sigma_spatial_distance**2)) + (-huedis**2/(2*sigma_hue_distance**2)) +(-centerDis**2/(2*sigma_center_distance**2)) ))
    # adj_element = np.exp((-sxdiff**2/(2*sigma_xspd_diff**2)+(-sydiff**2/(2*sigma_yspd_diff**2))+(-mdis**2/(2*sigma_spatial_distance**2)) + (-huedis/100) +(-centerDis/100)))
    # adj_element = np.exp(-sxdiff_normalized-sydiff_normalized-mdis_normalized-huedis_normalized-centerDis_normalized)

    fully_adj = np.exp(- (weight[0]*sxdiff_normalized+weight[1]*sydiff_normalized+weight[2]*mdis_normalized+weight[3]*huedis_normalized+weight[4]*centerDis_normalized))
    fully_adj[np.isnan(fully_adj)] = 0

    """construct KNN graph from fully connected graph"""
    # adj_knn = knn_graph((fully_adj),knn = 20)
    fully_adj = fully_adj + fully_adj.transpose() 
    adj = fully_adj.copy()

    """inspect the adj elements for the two vehicles"""
    # FeatureMtxLoc = pickle.load(open(os.path.join(DataPathobj.adjpath,'veryZigWhiteCar_FeatureMtxLoc'),'rb'))
    # vehicle1ind = 0
    # vehicle2ind = 1
    # """adj element between two vehicles:"""
    # print np.unique((fully_adj+fully_adj.T)[np.array(FeatureMtxLoc[vehicle1ind]),:][:,np.array(FeatureMtxLoc[vehicle2ind])])

    # print np.unique((fully_adj+fully_adj.T)[np.array(FeatureMtxLoc[vehicle1ind]),:][:,np.array(FeatureMtxLoc[vehicle1ind])])
    # print np.unique((fully_adj+fully_adj.T)[np.array(FeatureMtxLoc[vehicle2ind]),:][:,np.array(FeatureMtxLoc[vehicle2ind])])


    if afterNormalize:
        # """spatial distance"""
        # adj = adj*(feature_diff_tensor[:,:,2]< Parameterobj.nullDist_for_adj/(extremeValue[5]-extremeValue[4])*1)
        # """velocities"""
        # # adj = adj*(feature__diff_tensor[:,:,0]< Parameterobj.nullXspd_for_adj/(extremeValue[1]-extremeValue[0])*1)
        # # adj = adj*(feature_diff_tensor[:,:,1]< Parameterobj.nullYspd_for_adj/(extremeValue[3]-extremeValue[2])*1)
        # adj = adj*(feature_diff_tensor[:,:,0]< Parameterobj.nullXspd_for_adj_norm)
        # adj = adj*(feature_diff_tensor[:,:,1]< Parameterobj.nullYspd_for_adj_norm)
        # """blob"""
        # # # adj = adj*(feature_diff_tensor[:,:,4]< Parameterobj.nullBlob_for_adj/(extremeValue[9]-extremeValue[8])*1)

        normalized_feature_diff_tensor2 = feature_diff_tensor
        normalized_feature_diff_tensor2[np.isnan(normalized_feature_diff_tensor2)] = 0
        normalized_feature_diff_tensor2[:,:,0] = normalized_feature_diff_tensor2[:,:,0]+normalized_feature_diff_tensor2[:,:,0].T
        normalized_feature_diff_tensor2[:,:,1] = normalized_feature_diff_tensor2[:,:,1]+normalized_feature_diff_tensor2[:,:,1].T
        normalized_feature_diff_tensor2[:,:,2] = normalized_feature_diff_tensor2[:,:,2]+normalized_feature_diff_tensor2[:,:,2].T

        """used the new adj_thresholding threshold to normalize"""
        """spatial distance"""
        adj = adj*(normalized_feature_diff_tensor2[:,:,2]< 1)
        """velocities"""
        adj = adj*(normalized_feature_diff_tensor2[:,:,0]< 1)
        adj = adj*(normalized_feature_diff_tensor2[:,:,1]< 1)
        # """blob"""
        # # adj = adj*(feature_diff_tensor[:,:,4]< Parameterobj.nullBlob_for_adj/(extremeValue[9]-extremeValue[8])*1)


    else:
        """Hard thresholding adj based on spatial distance"""
        adj = adj*(feature_diff_tensor[:,:,2]< Parameterobj.nullDist_for_adj)

        """Hard thresholding adj based on velocities"""
        adj = adj*(feature_diff_tensor[:,:,0]< Parameterobj.nullXspd_for_adj)
        adj = adj*(feature_diff_tensor[:,:,1]< Parameterobj.nullYspd_for_adj)

        # """Hard thresholding adj based on blob center dist"""
        # adj = adj*(feature_diff_tensor[:,:,4]< Parameterobj.nullBlob_for_adj)

        """Hard thresholding adj based on hue dist"""
        """Hue info is very very weak, even with 0.001, still almost fully connected"""
        # adj = adj*(feature_diff_tensor[:,:,3]< 0.001)

    adj = adj + adj.transpose() 
    if np.sum(adj==adj.T)!=adj.shape[0]*adj.shape[1]:
        pdb.set_trace()
    # adj_GTind(fully_adj,sameDirTrjID)


    return adj,fully_adj


def CCoverseg(matidx,adj,fully_adj,sameDirTrjID,x,y,non_isolatedCC):
    # if matidx==1:
    #     big_CC_trjID = pickle.load(open(os.path.join(DataPathobj.DataPath,'NGSIM_bigCC_trjID_2ndTrunc'),'rb'))
    # if matidx==2:
    #     big_CC_trjID = pickle.load(open(os.path.join(DataPathobj.DataPath,'NGSIM_bigCC_trjID_3rdTrunc'),'rb'))

    # CC_overseg_trjID = pickle.load(open(os.path.join(DataPathobj.DataPath,'CC_overseg_trjID'),'rb'))
    # interesting_trjIDdic = big_CC_trjID
    # interesting_trjIDdic = CC_overseg_trjID


    # interesting_trjID = [ 921, 1754, 1903, 2032, 2120, 2325, 2584]
    # interesting_trjID =  [  77,  104,  295,  330,  367,  445,  518,  606,  723,  840,  855,
    #     865,  891, 1138, 1362, 1724, 1745]


    interesting_trjID = [2887, 2896, 3000, 3399, 3609,       2714, 2735, 2755, 2764, 2844, 2976, 3192, 4004]



    FeatureMtxLoc_CC = {}
    """interested in connected components, every CC has many trj IDs"""
    # for key in interesting_trjIDdic.keys():
    #     FeatureMtxLoc_CC [key] = []
    #     for aa in interesting_trjIDdic[key]:
    #         # print list(np.where(sameDirTrjID==aa)[0])
    #         FeatureMtxLoc_CC[key]+= list(np.where(sameDirTrjID==aa)[0])

    """interested in trj directly"""
    for key in interesting_trjID:
        location = list(np.where(sameDirTrjID==key)[0])
        if location:
            FeatureMtxLoc_CC [key] = []
            FeatureMtxLoc_CC[key]+= location


    np.where(c==56)[0]


    """distance within one connected component:"""
    # for key in interesting_trjIDdic.keys():
    #     interesting_adj_part = (adj[np.array(FeatureMtxLoc_CC[key]),:][:,np.array(FeatureMtxLoc_CC[key])]>0).astype(int)
    #     interesting_fulladj_part = ((fully_adj+fully_adj.T)[np.array(FeatureMtxLoc_CC[key]),:][:,np.array(FeatureMtxLoc_CC[key])]>0).astype(int)
    #     temp_dist = feature_diff_tensor[:,:,2]
    #     temp_dist[np.isnan(temp_dist)] = 0
    #     dist_these_CCs = temp_dist[np.array(FeatureMtxLoc_CC[key]),:][:,np.array(FeatureMtxLoc_CC[key])].astype(int)
    #     x_these_CCs = x[np.array(FeatureMtxLoc_CC[key]),:]
    #     y_these_CCs = y[np.array(FeatureMtxLoc_CC[key]),:]

    """interested in trj directly"""
    interesting_loc = np.reshape(np.array(FeatureMtxLoc_CC.values()),(-1,))
    interesting_adj_part = (adj[interesting_loc,:][:,interesting_loc]>0).astype(int)
    interesting_fulladj_part = (fully_adj[interesting_loc,:][:,interesting_loc]>0).astype(int)
    
    normalized_feature_diff_tensor2 = normalized_feature_diff_tensor[non_isolatedCC,:][:,non_isolatedCC] ## use the newly normalized tensor
    pdb.set_trace()
    normalized_feature_diff_tensor2[np.isnan(normalized_feature_diff_tensor2)] = 0
    temp_dist = normalized_feature_diff_tensor2[:,:,2]+normalized_feature_diff_tensor2[:,:,2].T
    dist_these_CCs = temp_dist[interesting_loc,:][:,interesting_loc].astype(int)
    Vx_dist_these_CCs = (normalized_feature_diff_tensor2[:,:,0]+normalized_feature_diff_tensor2[:,:,0].T)[interesting_loc,:][:,interesting_loc].astype(int)
    Vy_dist_these_CCs = (normalized_feature_diff_tensor2[:,:,1]+normalized_feature_diff_tensor2[:,:,1].T)[interesting_loc,:][:,interesting_loc].astype(int)

    x_these_CCs = x[non_isolatedCC,:][interesting_loc,:]
    y_these_CCs = y[non_isolatedCC,:][interesting_loc,:]


    CClabelforinterestingTrjID = c[interesting_loc]

    if np.sum(interesting_adj_part!= interesting_fulladj_part*(Vx_dist_these_CCs< 1))>1:
        pdb.set_trace()


    # plt.figure()
    # for hh in range(len(FeatureMtxLoc_CC[key])):
    #     featureloc = FeatureMtxLoc_CC[key][hh]
    #     plt.scatter(x[non_isolatedCC,:][featureloc,:],y[non_isolatedCC,:][featureloc,:])
    #     plt.draw()
    #     plt.show()
    plt.figure()
    for hh in range(len(interesting_loc)):
        plt.scatter(x[non_isolatedCC,:][interesting_loc,:],y[non_isolatedCC,:][interesting_loc,:])
        plt.draw()
        plt.show()

    pdb.set_trace()



# same blob score (SBS) for two trajectories
def sameBlobScore(fgBlobInd1,fgBlobInd2):
    SBS = np.sum(fgBlobInd1 == fgBlobInd2)
    return SBS

def prepare_input_data():
    # if smooth:
    matfilepath = DataPathobj.smoothpath
    matfiles = sorted(glob.glob(matfilepath + 'klt*.mat'))
    # else:
    #     matfilepath = DataPathobj.filteredKltPath
    #     matfiles = sorted(glob.glob(matfilepath + 'len*.mat'))
    
    savePath = DataPathobj.adjpath
    start_position_offset = 0
    matfiles = matfiles[start_position_offset:]
    return matfiles,savePath, start_position_offset

def get_spd_dis_diff(xspd_i,xspd_j,yspd_i,yspd_j,xi,xj,yi,yj):
    """use mean of the spd diff"""
    # sxdiff = np.mean(np.abs(xspd[i,sidx]-xspd[j,sidx]))
    # sydiff = np.mean(np.abs(yspd[i,sidx]-yspd[j,sidx]))                    
    """use MAX of the spd diff!"""
    sxdiff = np.max(np.abs(xspd_i-xspd_j)[:])
    sydiff = np.max(np.abs(yspd_i-yspd_j)[:])                    

    """use MAX of the dist diff!"""
    # mdis   = np.mean(np.abs(x[i,idx]-x[j,idx])+np.abs(y[i,idx]-y[j,idx])) #mahhattan distance
    # mdis = np.mean(np.sqrt((x[i,idx]-x[j,idx])**2+(y[i,idx]-y[j,idx])**2))  #euclidean distance
    mdis = np.max(np.sqrt((xi-xj)**2+(yi-yj)**2))  #euclidean distance
    return sxdiff,sydiff,mdis

def get_hue_diff(hue_i,hue_j):
    huedis = np.abs(np.nanmean(hue_i)-np.nanmean(hue_j))
    return huedis

def diff_feature_on_one_car(dataForKernel,feature_diff_tensor, trjID):
    # one_car_trjID = pickle.load(open('./johnson_one_car_trjID','rb'))
    # one_car_trjID = pickle.load(open(os.path.join(DataPathobj.adjpath,'one_car_trjID'),'rb'))
    # one_car_trjID = pickle.load(open(os.path.join(DataPathobj.adjpath,'veryZigWhiteCar'),'rb'))

    bus_and_car_trjID = [[0,3,204,764,1428,1218],[1092,138,921,934,54,65]]
    one_car_trjID = bus_and_car_trjID

    FeatureMtxLoc    = {}
    dist_one_car     = {}
    vxdist_one_car   = {}
    vydist_one_car   = {}
    blobdist_one_car = {}
    for ii in range(np.array(one_car_trjID).shape[0]):
        FeatureMtxLoc[ii]    = []
        dist_one_car[ii]     = []
        vxdist_one_car[ii]   = []
        vydist_one_car[ii]   = []
        blobdist_one_car[ii] = []
        feature_diff_tensor2 = feature_diff_tensor
        feature_diff_tensor2[np.isnan(feature_diff_tensor2)]=0
        for aa in np.array(one_car_trjID)[ii]:
            FeatureMtxLoc[ii]+=list(np.where(trjID==aa)[0])
            dist_one_car[ii]     = np.max((feature_diff_tensor2[:,:,2]+feature_diff_tensor2[:,:,2].T)[np.array(FeatureMtxLoc[ii]),:][:,np.array(FeatureMtxLoc[ii])])
            vxdist_one_car[ii]   = np.max((feature_diff_tensor2[:,:,0]+feature_diff_tensor2[:,:,0].T)[np.array(FeatureMtxLoc[ii]),:][:,np.array(FeatureMtxLoc[ii])])
            vydist_one_car[ii]   = np.max((feature_diff_tensor2[:,:,1]+feature_diff_tensor2[:,:,1].T)[np.array(FeatureMtxLoc[ii]),:][:,np.array(FeatureMtxLoc[ii])])
            blobdist_one_car[ii] = np.max((feature_diff_tensor2[:,:,4]+feature_diff_tensor2[:,:,4].T)[np.array(FeatureMtxLoc[ii]),:][:,np.array(FeatureMtxLoc[ii])])
    

    # interesting_adj_part     = (adj[np.array(FeatureMtxLoc[ii]),:][:,np.array(FeatureMtxLoc[ii])]>0).astype(int)
    # interesting_fulladj_part = ((fully_adj+fully_adj.T)[np.array(FeatureMtxLoc[ii]),:][:,np.array(FeatureMtxLoc[ii])]>0).astype(int)





    # pickle.dump(FeatureMtxLoc,open(os.path.join(DataPathobj.adjpath,'veryZigWhiteCar_FeatureMtxLoc'),'wb'))
    np.max(dist_one_car.values())
    np.max(vxdist_one_car.values())
    np.max(vydist_one_car.values())
    np.max(blobdist_one_car.values())

    feature_dim = 2
    vehicle1ind = 0
    vehicle2ind = 1
    """distance between two vehicles:"""
    print np.unique((feature_diff_tensor[:,:,feature_dim]+feature_diff_tensor[:,:,feature_dim].T)\
        [np.array(FeatureMtxLoc[vehicle1ind]),:][:,np.array(FeatureMtxLoc[vehicle2ind])])

    """distance within one vehicle:"""
    print np.unique((feature_diff_tensor[:,:,feature_dim]+feature_diff_tensor[:,:,feature_dim].T)\
        [np.array(FeatureMtxLoc[vehicle1ind]),:][:,np.array(FeatureMtxLoc[vehicle1ind])])

    print np.unique((feature_diff_tensor[:,:,feature_dim]+feature_diff_tensor[:,:,feature_dim].T)\
        [np.array(FeatureMtxLoc[vehicle2ind]),:][:,np.array(FeatureMtxLoc[vehicle2ind])])





    pdb.set_trace()



if __name__ == '__main__':
    isVisualize = False
    matfiles,savePath,start_position_offset = prepare_input_data()
    # adj_methods = np.nan
    # adj_methods = "Thresholding"
    adj_methods = "Gaussian"
    # adj_methods = "Cosine"

    # """to visualize the neighbours"""
    if isVisualize:
    	fig888 = plt.figure()
    	ax     = plt.subplot(1,1,1)

    for matidx,matfile in enumerate(matfiles):
    # for matidx in range(5,len(matfiles)):
    # for matidx in range(3,4,1):
        # matfile = matfiles[matidx]
        result = {} #for the save in the end
        print "Processing truncation...", str(matidx+1)
        ptstrj = loadmat(matfile)
        """if no trj in this file, just continue"""
        try:
            print 'total number of trjs in this trunk', len(ptstrj['trjID'])
        except:
            continue
        if len(ptstrj['trjID'])==0:
            continue
        trjID = ptstrj['trjID'][0]
        hue = csr_matrix(ptstrj['Huetracks'], shape=ptstrj['Huetracks'].shape).toarray()

        if not Parameterobj.useWarpped:            
            x    = csr_matrix(ptstrj['xtracks'], shape=ptstrj['xtracks'].shape).toarray()
            y    = csr_matrix(ptstrj['ytracks'], shape=ptstrj['ytracks'].shape).toarray()
            t    = csr_matrix(ptstrj['Ttracks'], shape=ptstrj['Ttracks'].shape).toarray()
            xspd = csr_matrix(ptstrj['xspd'], shape=ptstrj['xspd'].shape).toarray()
            yspd = csr_matrix(ptstrj['yspd'], shape=ptstrj['yspd'].shape).toarray()
            Xdir = csr_matrix(ptstrj['Xdir'], shape=ptstrj['Xdir'].shape).toarray()
            Ydir = csr_matrix(ptstrj['Ydir'], shape=ptstrj['Ydir'].shape).toarray()
        else:
            x    = csr_matrix(ptstrj['xtracks_warpped'],shape=ptstrj['xtracks'].shape).toarray()
            y    = csr_matrix(ptstrj['ytracks_warpped'],shape=ptstrj['ytracks'].shape).toarray()
            t    = csr_matrix(ptstrj['Ttracks'],shape=ptstrj['Ttracks'].shape).toarray()
            xspd = csr_matrix(ptstrj['xspd_warpped'], shape=ptstrj['xspd'].shape).toarray()
            yspd = csr_matrix(ptstrj['yspd_warpped'], shape=ptstrj['yspd'].shape).toarray()
            Xdir = csr_matrix(ptstrj['Xdir_warpped'], shape=ptstrj['Xdir_warpped'].shape).toarray()
            Ydir = csr_matrix(ptstrj['Ydir_warpped'], shape=ptstrj['Ydir_warpped'].shape).toarray()



        if Parameterobj.useSBS:
            FgBlobIndex = csr_matrix(ptstrj['fg_blob_index'], shape=ptstrj['fg_blob_index'].shape).toarray()
            fg_blob_center_X = csr_matrix(ptstrj['fg_blob_center_X'], shape=ptstrj['fg_blob_center_X'].shape).toarray()
            fg_blob_center_Y = csr_matrix(ptstrj['fg_blob_center_Y'], shape=ptstrj['fg_blob_center_Y'].shape).toarray()

            FgBlobIndex[FgBlobIndex==0]=np.nan
            fg_blob_center_X[FgBlobIndex==0]=np.nan
            fg_blob_center_Y[FgBlobIndex==0]=np.nan
        else:
            fg_blob_center_X = np.ones(x.shape)*np.nan
            fg_blob_center_Y = np.ones(x.shape)*np.nan
            FgBlobIndex      = np.ones(x.shape)*np.nan

        Numsample = ptstrj['xtracks'].shape[0]
        fnum      = ptstrj['xtracks'].shape[1]
        

        """First cluster using just direction Information"""
        upup = ((Xdir>=0)*(Ydir>=0))[0]
        upupind = np.array(range(Numsample))[upup]
        upupTrjID = trjID[upup]

        updown = ((Xdir>=0)*(Ydir<=0))[0]
        updownind = np.array(range(Numsample))[~upup*updown]
        updownTrjID = trjID[~upup*updown]

        downup = ((Xdir<=0)*(Ydir>=0))[0]
        downupind = np.array(range(Numsample))[~upup*(~updown)*downup]
        downupTrjID = trjID[~upup*(~updown)*downup]

        downdown = ((Xdir<=0)*(Ydir<=0))[0]
        downdownind = np.array(range(Numsample))[~upup*(~updown)*(~downup)*downdown]
        downdownTrjID = trjID[~upup*(~updown)*(~downup)*downdown]
        """ind is the location in this truncation, trjID is the real absolute ID"""
        DirInd = [upupind,updownind,downupind,downdownind]
        DirTrjID = [upupTrjID,updownTrjID,downupTrjID,downdownTrjID]
        if len(upupind)+len(updownind)+len(downupind)+len(downdownind)!=Numsample:
            pdb.set_trace()

        DirName = ['upup','updown','downup','downdown']
        cumNumSample =  0 
        for dirii in range(4):
            if cumNumSample==Numsample: ## already done on all samples, no need to try other directions
                break
            sameDirInd = DirInd[dirii]
            sameDirTrjID = DirTrjID[dirii]
            NumGoodsampleSameDir = len(sameDirInd)
            cumNumSample += NumGoodsampleSameDir
            if NumGoodsampleSameDir==0:
                continue

            print'building adj mtx ....', NumGoodsampleSameDir,'*',NumGoodsampleSameDir
            dataForKernel = np.array([x[sameDirInd,:],y[sameDirInd,:],xspd[sameDirInd,:],yspd[sameDirInd,:],hue[sameDirInd,:],fg_blob_center_X[sameDirInd,:],fg_blob_center_Y[sameDirInd,:]])
            # FeatureMtx,FeatureMtx_norm = getRawDataFeatureMtx(dataForKernel)
            # TwoD_Emedding(FeatureMtx_norm)
            # pdb.set_trace()
            color = np.array([np.random.randint(0,255) for _ in range(3*int(NumGoodsampleSameDir))]).reshape(NumGoodsampleSameDir,3)
            num = np.arange(fnum)

            if adj_methods =="Gaussian":
                if len(sorted(glob.glob(savePath+'mean_std_ForKernel'+DirName[dirii]+str(matidx+1+start_position_offset).zfill(3))))>0:
                    print "mean_std_ForKernel and extremeValue already stored, load..."
                    mean_std_ForKernel = pickle.load(open(DataPathobj.adjpath+'mean_std_ForKernel'+DirName[dirii]+str(matidx+1+start_position_offset).zfill(3),'rb'))
                    extremeValue = pickle.load(open(DataPathobj.adjpath+'extremeValue'+DirName[dirii]+str(matidx+1+start_position_offset).zfill(3),'rb'))
                else:              
                    mean_std_ForKernel,extremeValue = getMuSigma(dataForKernel)
                    pickle.dump(mean_std_ForKernel,open(DataPathobj.adjpath+'mean_std_ForKernel'+DirName[dirii]+str(matidx+1+start_position_offset).zfill(3),'wb'))
                    pickle.dump(extremeValue,open(DataPathobj.adjpath+'extremeValue'+DirName[dirii]+str(matidx+1+start_position_offset).zfill(3),'wb'))



            # SBS = np.zeros([NumGoodsampleSameDir,NumGoodsampleSameDir])
            """store all pair feature distances"""
            """Nsample*Nsample* 5 distance features"""
            if len(sorted(glob.glob(savePath+'normalized_feature_diff_tensor'+DirName[dirii]+str(matidx+1+start_position_offset).zfill(3))))>0:
                print "normalized distance diff already stored, load..."
                normalized_feature_diff_tensor = pickle.load(open(savePath+'normalized_feature_diff_tensor'+DirName[dirii]+str(matidx+1+start_position_offset).zfill(3),'rb'))
            if len(sorted(glob.glob(savePath+'feature_diff_tensor'+DirName[dirii]+str(matidx+1+start_position_offset).zfill(3))))>0:
                print "distance diff already stored, load..."
                feature_diff_tensor = pickle.load(open(savePath+'feature_diff_tensor'+DirName[dirii]+str(matidx+1+start_position_offset).zfill(3),'rb'))
            else:
                normalized_feature_diff_tensor = np.ones([NumGoodsampleSameDir,NumGoodsampleSameDir,5])*np.nan
                # feature_diff_tensor = np.ones([NumGoodsampleSameDir,NumGoodsampleSameDir,5])*np.nan
                for i in range(NumGoodsampleSameDir):
                    print "i", i
                    for j in range(i+1, NumGoodsampleSameDir):
                        tmp1 = x[i,:]!=0
                        tmp2 = x[j,:]!=0
                        idx  = num[tmp1&tmp2]
                      
                        if len(idx)>= Parameterobj.trjoverlap_len_thresh: # has overlapping
                            sidx   = idx[1:] # for speeds
                            sxdiff,sydiff,mdis = get_spd_dis_diff(xspd[i,sidx],xspd[j,sidx],yspd[i,sidx],yspd[j,sidx],x[i,idx],x[j,idx],y[i,idx],y[j,idx])
                            # if i in [0, 1, 24, 55, 111, 88] and j in [0, 1, 24, 55, 111, 88]:
                            #     print 'trjID1,trjID2',trjID[i],trjID[j]
                            #     print 'sxdiff,sydiff,mdis',sxdiff,sydiff,mdis
                            # huedis = np.mean(np.abs(hue[i,sidx]-hue[j,sidx]))
                            huedis = get_hue_diff(hue[i,sidx],hue[j,sidx])
                            if Parameterobj.useSBS:
                                cxi = np.nanmedian(fg_blob_center_X[i,idx])
                                cyi = np.nanmedian(fg_blob_center_Y[i,idx])
                                cxj = np.nanmedian(fg_blob_center_X[j,idx])
                                cyj = np.nanmedian(fg_blob_center_Y[j,idx])

                                """if not inside a blob, use trj's own center"""
                                if np.isnan(cxi) or np.isnan(cyi): 
                                    cxi = np.nanmedian(x[i,idx])
                                    cyi = np.nanmedian(y[i,idx])
                                if np.isnan(cxj) or np.isnan(cyj):
                                    cxj = np.nanmedian(x[j,idx])
                                    cyj = np.nanmedian(y[j,idx])

                                """if not inside a blob, assign trj to nearest blob???"""

   
                                centerDis = np.sqrt((cxi-cxj)**2+(cyi-cyj)**2)

                                if np.isnan(centerDis):
                                    # centerDis = mean_std_ForKernel[-2]+mean_std_ForKernel[-1] # treat as very far away, mu+sigma away
                                    pdb.set_trace()
                            else:
                                centerDis=0
                            # print 'centerDis',centerDis
                            trj1 = [x[i,idx],y[i,idx]]
                            trj2 = [x[j,idx],y[j,idx]]
            
                            dataForKernel_ele = [sxdiff,sydiff,mdis,huedis,centerDis]
                            # """counting the sharing blob numbers of two trjs"""
                            # if useSBS:
                            #     SBS[i,j] = sameBlobScore(np.array(FgBlobIndex[i,idx]),np.array(FgBlobIndex[j,idx]))
                            
                            """normalize the distance"""
                            normalized_feature_diff_tensor[i,j,:] = normalize_features(dataForKernel_ele,mean_std_ForKernel,extremeValue,Parameterobj.useSBS)
                            # feature_diff_tensor[i,j,:] = dataForKernel_ele

                        # else: # overlapping too short
                        #     # SBS[i,j] = 0
                        #     # adj[i,j] = 0
                        #     feature_diff_tensor[i,j,:] = np.nan


                # pickle.dump(normalized_feature_diff_tensor,open(savePath+'normalized_feature_diff_tensor'+DirName[dirii]+str(matidx+1+start_position_offset).zfill(3),'wb'))
                pickle.dump(feature_diff_tensor,open(savePath+'feature_diff_tensor'+DirName[dirii]+str(matidx+1+start_position_offset).zfill(3),'wb'))



            """test only for one car, see different features' role in the adj"""
            # diff_feature_on_one_car(dataForKernel,feature_diff_tensor,trjID)
            """build adj mtx"""
            adj = np.zeros([NumGoodsampleSameDir,NumGoodsampleSameDir])
            if adj_methods =="Thresholding":
                adj=get_thresholding_adj(adj,feature_diff_tensor)

            if adj_methods =="Gaussian":
                # adj,fully_adj=get_gaussian_adj(adj,feature_diff_tensor,sameDirTrjID,afterNormalize = False)
                adj,fully_adj=get_gaussian_adj(adj,normalized_feature_diff_tensor,sameDirTrjID,afterNormalize = True)


            # SBS = SBS + SBS.transpose()  #add diag twice
            # np.fill_diagonal(SBS, np.diagonal(SBS)/2)

            
            # np.fill_diagonal(adj, 1)
            """diagonal actually doesn't matter in Spectral Clustering"""
            np.fill_diagonal(adj, 0)

            """save each same direction adj"""
            temp = (adj>0).astype(int)
            s,c = connected_components(temp) #s is the total CComponent, c is the label
            
            # pdb.set_trace()
            """delete trjs that formed isolated very small CC"""
            non_isolatedCC = []
            for CClabel in np.unique(c):
                if len(np.where(c==CClabel)[0])>=3:
                    non_isolatedCC+=list(np.where(c==CClabel)[0])


            adj = adj[non_isolatedCC,:][:,non_isolatedCC]
            c = c[non_isolatedCC]
            sameDirTrjID = sameDirTrjID[non_isolatedCC]
            
            sparsemtx = csr_matrix(adj)
            result['adj_'+DirName[dirii]]   = sparsemtx
            result['c_'+DirName[dirii]]     = c
            result['trjID_'+DirName[dirii]] = sameDirTrjID
            result['non_isolatedCC'+DirName[dirii]] = non_isolatedCC

            # ss,cc = connected_components((adj>0).astype(int)) #s is the total CComponent, c is the label
            # if ss>1:
            #     pdb.set_trace()



            """often, one vehicle is already over-segmented by CC, why"""
            # final_trjID   = np.uint32(loadmat(open(os.path.join(DataPathobj.unifiedLabelpath,'concomp'+'c_'+DirName[dirii]+'.mat')))['trjID'][0]) 
            # final_cc_label = loadmat(open(os.path.join(DataPathobj.unifiedLabelpath,'concomp'+'c_'+DirName[dirii]+'.mat')))['label'][0]

            # CCind_overSeg = [[19,26,24],[12,30],[11,37],[46,13],[54,31],[83,86,74,90],[95,103],[89,101],[92,75],[97,84],[204,211,225],[192,198],[210,217],[51,25],[61,66],[309,282],
            # [92,75],[200,280],[304,310]]
            # CC_overseg_trjID = {}
            # for mm in range(len(CCind_overSeg)):
            #     CC_overseg_trjID[mm] = []
            #     # for CCind in CCind_overSeg[mm]:
            #     #     CC_overseg_trjID[mm]+= list(final_trjID[np.where(final_cc_label == CCind)[0]])
            #     for CCind in CCind_overSeg[mm]:
            #         CC_overseg_trjID[mm]+= list(trjID[np.where(c == CCind)[0]])


            # # pickle.dump(CC_overseg_trjID,open(os.path.join(DataPathobj.adjpath,'CC_overseg_trjID'),'wb'))
            
            # CCoverseg(matidx,adj,fully_adj,sameDirTrjID,x,y,non_isolatedCC)

        print "saving adj..."
        """save all adj, not seperated by directions"""
        # sparsemtx = csr_matrix(adj)
        # s,c       = connected_components(sparsemtx) #s is the total CComponent, c is the label
        # result          = {}
        # result['adj']   = sparsemtx
        # result['c']     = c
        # result['trjID'] = ptsidx

        if Parameterobj.useWarpped:
            # savename = 'usewarpped_'+adj_methods+'_Adj_300_5_5_'+str(matidx+1+start_position_offset).zfill(3)
            savename = 'usewarpped_'+adj_methods+'_April22_'+str(matidx+1+start_position_offset).zfill(3)

        else:
            # savename = adj_methods+'_diff_dir_'+str(matidx+1+start_position_offset).zfill(3)
            # savename = 'spa_velo_hard_thresholded_'+adj_methods+'_diff_dir_'+str(matidx+1+start_position_offset).zfill(3)
            # savename = '20knn_'+adj_methods+'_diff_dir_'+str(matidx+1+start_position_offset).zfill(3)
            # savename = '20knn&thresh_'+adj_methods+'_diff_dir_'+str(matidx+1+start_position_offset).zfill(3)
            # savename = 'onlyBlobThresh'+adj_methods+'_diff_dir_'+str(matidx+1+start_position_offset).zfill(3)
            # savename = 'SpaSpdBlobthresh_'+adj_methods+'_diff_dir_'+str(matidx+1+start_position_offset).zfill(3)
            # savename = 'thresholding_adj_spatial_'+adj_methods+'_diff_dir_'+str(matidx+1+start_position_offset).zfill(3)
            savename = 'normalize_thresholding_adj_all_'+adj_methods+'_diff_dir_'+str(matidx+1+start_position_offset).zfill(3)
            # savename = 'baseline_thresholding_adj_all'+adj_methods+'_diff_dir_'+str(matidx+1+start_position_offset).zfill(3)


        savename = os.path.join(savePath,savename)
        savemat(savename,result)

            

        """ visualization, see if connected components make sense"""
        # pdb.set_trace()        
        # s111,c111 = connected_components(sparsemtx) #s is the total CComponent, c is the label
        # color     = np.array([np.random.randint(0,255) for _ in range(3*int(s111))]).reshape(s111,3)
        # fig888    = plt.figure(888)
        # ax        = plt.subplot(1,1,1)
        # # im = plt.imshow(np.zeros([528,704,3]))
        # for i in range(s111):
        #     ind = np.where(c111==i)[0]
        #     print ind
        #     for jj in range(len(ind)):
        #         startlimit = np.min(np.where(x[ind[jj],:]!=0))
        #         endlimit = np.max(np.where(x[ind[jj],:]!=0))
        #         # lines = ax.plot(x[ind[jj],startlimit:endlimit], y[ind[jj],startlimit:endlimit],color = (0,1,0),linewidth=2)
        #         lines = ax.plot(x[ind[jj],startlimit:endlimit], y[ind[jj],startlimit:endlimit],color = (color[i-1].T)/255.,linewidth=2)
        #         fig888.canvas.draw()
        #     plt.pause(0.0001) 
        # plt.show()
        # pdb.set_trace()





