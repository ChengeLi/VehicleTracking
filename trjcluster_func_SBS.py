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
from DataPathclass import *
DataPathobj = DataPath(dataSource,VideoIndex)
from parameterClass import *
Parameterobj = parameter(dataSource,VideoIndex)



'test for PCA'
from sklearn.decomposition import PCA
def PCAembedding(data,n_components):
    print 'PCA projecting...'
    pca = PCA(n_components= n_components,whiten=False)
    pca.fit(data)
    return pca



pca = PCAembedding(FeatureMtx_norm,200)
FeatureAfterPCA = pca.transform(FeatureMtx_norm)

pca3 = PCAembedding(FeatureMtx_norm,3)
FeatureAfterPCA3 = pca3.transform(FeatureMtx_norm)

pca50 = PCAembedding(FeatureMtx_norm,50)
FeatureAfterPCA50 = pca50.transform(FeatureMtx_norm)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=4.0, learning_rate=1000.0, n_iter=1000, n_iter_without_progress=30, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5)

tsne.fit_transform(FeatureMtx_norm50) 




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

    for i in range(NumGoodsample):
        # print "i", i
        # for j in range(i, min(NumGoodsample,i+1000)):
        for j in range(i, NumGoodsample):
            tmp1 = x[i,:]!=0
            tmp2 = x[j,:]!=0
            idx  = num[tmp1&tmp2]
            
            if len(idx)>5: # has overlapping
                # print len(idx)
            # if len(idx)>=30: # at least overlap for 100 frames
                sidx   = idx[0:-1] # for speed
                sxdiff = np.mean(np.abs(xspd[i,sidx]-xspd[j,sidx]))
                sydiff = np.mean(np.abs(yspd[i,sidx]-yspd[j,sidx]))                    
                # mdis   = np.mean(np.abs(x[i,idx]-x[j,idx])+np.abs(y[i,idx]-y[j,idx])) #mahhattan distance
                mdis = np.mean(np.sqrt((x[i,idx]-x[j,idx])**2+(y[i,idx]-y[j,idx])**2))
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
                    huedis = np.mean(np.abs(hue[i,sidx]-hue[j,sidx]))
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

    'fit Gaussian to find mu and sigma'
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




def adj_gaussian_element(dataForKernel_ele,mean_std_ForKernel,extremeValue,useSBS):
    # sigma_xspd_diff = 0.7
    # sigma_yspd_diff = 0.7
    # sigma_spatial_distance = 200

    [sxdiff,sydiff,mdis,huedis,centerDis] = dataForKernel_ele
    # [mu_xspd_diff,sigma_xspd_diff,mu_yspd_diff,sigma_yspd_diff,mu_spatial_distance,sigma_spatial_distance,mu_hue_distance,sigma_hue_distance, mu_center_distance,sigma_center_distance] =     mean_std_ForKernel 

    [min_sx,max_sx,min_sy,max_sy,min_mdis,max_mdis,min_hue,max_hue,min_center,max_center]=extremeValue

    'normalize'
    sxdiff_normalized    = (sxdiff-min_sx)/(max_sx-min_sx)
    sydiff_normalized    = (sydiff-min_sy)/(max_sy-min_sy)
    mdis_normalized      = (mdis-min_mdis)/(max_mdis-min_mdis)
    huedis_normalized    = (huedis-min_hue)/(max_hue-min_hue)
    centerDis_normalized = (centerDis-min_center)/(max_center-min_center)


    if useSBS:
        # adj_element = np.exp((-sxdiff**2/(2*sigma_xspd_diff**2)+(-sydiff**2/(2*sigma_yspd_diff**2))+(-mdis**2/(2*sigma_spatial_distance**2)) + (-huedis**2/(2*sigma_hue_distance**2)) +(-centerDis**2/(2*sigma_center_distance**2)) ))
    
        # adj_element = np.exp((-sxdiff**2/(2*sigma_xspd_diff**2)+(-sydiff**2/(2*sigma_yspd_diff**2))+(-mdis**2/(2*sigma_spatial_distance**2)) + (-huedis/100) +(-centerDis/100)))

        adj_element = np.exp(-sxdiff_normalized-sydiff_normalized-mdis_normalized-huedis_normalized-centerDis_normalized)

    else:
        # adj_element = np.exp((-sxdiff**2/2*stdx**2)+(-sydiff**2/2*stdy**2)+(-mdis**2/2*stdd**2))
        adj_element = np.exp(-sxdiff_normalized-sydiff_normalized-mdis_normalized
        )

    if math.isnan(adj_element):
        pdb.set_trace()
    return adj_element


def adj_cosine_element(i_trj,j_trj):
    # cosine similarity
    cos_element = np.dot(i_trj,j_trj)/np.sqrt(sum(abs(i_trj)**2))/np.sqrt(sum(abs(j_trj)**2))
    cos_element = (cos_element+1)/2 # make them all positive  
    # print "cos_element: ", str(cos_element)
    if math.isnan(cos_element):
        pdb.set_trace()
    return cos_element


def get_adj_element(adj_methods,dataForKernel_ele,mean_std_ForKernel,extremeValue,useSBS):
    # dth     = 300 #30*1.5
    # yspdth  = 0.7 #0.9 for warpped #5 #y speed threshold
    # xspdth  = 0.7 #0.9 for warpped #5 #x speed threshold

    [sxdiff,sydiff,mdis,huedis,centerDis] = dataForKernel_ele
    # if dataSource =='Johnson':
    #     # dth    = 80
    #     # yspdth = 0.2 #filtered out 2/3 pairs
    #     # xspdth = 0.35 
    #     dth    = 500
    #     yspdth = 5
    #     xspdth = 1
        
    try:
        yspdth = mean_std_ForKernel[2]+mean_std_ForKernel[3] #mean+sigma
        xspdth = mean_std_ForKernel[0]+mean_std_ForKernel[1]
    except:
        yspdth = 0.4*extremeValue[3] #if mean is empty
        xspdth = 0.4*extremeValue[1]

    if mdis < Parameterobj.nullDist_for_adj:
        if adj_methods =="Thresholding":
            if (sxdiff <xspdth ) & (sydiff<yspdth):
                adj_element = 1
            else:
                adj_element = 0

        if adj_methods =="Gaussian":
            adj_element=adj_gaussian_element(dataForKernel_ele,mean_std_ForKernel,extremeValue,Parameterobj.useSBS)

        if adj_methods == "Cosine":
            i_trj    = np.concatenate((x[i,idx], xspd[i,sidx],y[i,idx],yspd[i,sidx]), axis=1)
            j_trj    = np.concatenate((x[j,idx], xspd[j,sidx],y[j,idx],yspd[j,sidx]), axis=1)
            adj[i,j] = adj_cosine_element(i_trj,j_trj)
        """visualize the neighbours"""
        # lines = ax.plot(x[i,idx], y[i,idx],color = (color[i-1].T)/255.,linewidth=2)
        # lines = ax.plot(x[j,idx], y[j,idx],color = (color[j-1].T)/255.,linewidth=2)
        # fig888.canvas.draw()
        # plt.pause(0.0001)
    else:
        adj_element = 0
    return adj_element


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
    'use mean of the spd diff'
    # sxdiff = np.mean(np.abs(xspd[i,sidx]-xspd[j,sidx]))
    # sydiff = np.mean(np.abs(yspd[i,sidx]-yspd[j,sidx]))                    
    'use MAX of the spd diff!'
    sxdiff = np.max(np.abs(xspd[i,sidx]-xspd[j,sidx])[:])
    sydiff = np.max(np.abs(yspd[i,sidx]-yspd[j,sidx])[:])                    

    # mdis   = np.mean(np.abs(x[i,idx]-x[j,idx])+np.abs(y[i,idx]-y[j,idx])) #mahhattan distance
    mdis = np.mean(np.sqrt((x[i,idx]-x[j,idx])**2+(y[i,idx]-y[j,idx])**2))  #euclidean distance
    return sxdiff,sydiff,mdis


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
    # for matidx in range(2,4):
        # matfile = matfiles[matidx]
        print "Processing truncation...", str(matidx+1)
        ptstrj = loadmat(matfile)
        if len(ptstrj['trjID'])==0:
            continue
        ptsidx = ptstrj['trjID'][0]
        if not Parameterobj.useWarpped:            
            x    = csr_matrix(ptstrj['xtracks'], shape=ptstrj['xtracks'].shape).toarray()
            y    = csr_matrix(ptstrj['ytracks'], shape=ptstrj['ytracks'].shape).toarray()
            t    = csr_matrix(ptstrj['Ttracks'], shape=ptstrj['Ttracks'].shape).toarray()
            xspd = csr_matrix(ptstrj['xspd'], shape=ptstrj['xspd'].shape).toarray()
            yspd = csr_matrix(ptstrj['yspd'], shape=ptstrj['yspd'].shape).toarray()
        else:
            x    = csr_matrix(ptstrj['xtracks_warpped'],shape=ptstrj['xtracks'].shape).toarray()
            y    = csr_matrix(ptstrj['ytracks_warpped'],shape=ptstrj['ytracks'].shape).toarray()
            t    = csr_matrix(ptstrj['Ttracks'],shape=ptstrj['Ttracks'].shape).toarray()
            xspd = csr_matrix(ptstrj['xspd_warpped'], shape=ptstrj['xspd'].shape).toarray()
            yspd = csr_matrix(ptstrj['yspd_warpped'], shape=ptstrj['yspd'].shape).toarray()

        
        if Parameterobj.useSBS:
            hue   = csr_matrix(ptstrj['Huetracks'], shape=ptstrj['Huetracks'].shape).toarray()
            FgBlobIndex      = csr_matrix(ptstrj['fg_blob_index'], shape=ptstrj['fg_blob_index'].shape).toarray()
            fg_blob_center_X = csr_matrix(ptstrj['fg_blob_center_X'], shape=ptstrj['fg_blob_center_X'].shape).toarray()
            fg_blob_center_Y = csr_matrix(ptstrj['fg_blob_center_Y'], shape=ptstrj['fg_blob_center_Y'].shape).toarray()

            FgBlobIndex[FgBlobIndex==0]=np.nan
            fg_blob_center_X[FgBlobIndex==0]=np.nan
            fg_blob_center_Y[FgBlobIndex==0]=np.nan
        else:
            hue = []
            fg_blob_center_X =[]
            fg_blob_center_Y = []
            FgBlobIndex      = []

        dataForKernel = np.array([x,y,xspd,yspd,hue,fg_blob_center_X,fg_blob_center_Y])
        FeatureMtx,FeatureMtx_norm = getRawDataFeatureMtx(dataForKernel)


        Numsample = ptstrj['xtracks'].shape[0]
        fnum      = ptstrj['xtracks'].shape[1]
        color = np.array([np.random.randint(0,255) for _ in range(3*int(Numsample))]).reshape(Numsample,3)
        NumGoodsample = len(x)
        # construct adjacency matrix
        print'building adj mtx ....', NumGoodsample,'*',NumGoodsample
        adj = np.zeros([NumGoodsample,NumGoodsample])
        # SBS = np.zeros([NumGoodsample,NumGoodsample])
        num = np.arange(fnum)

        


        pdb.set_trace()
        if adj_methods =="Gaussian":
            mean_std_ForKernel,extremeValue = getMuSigma(dataForKernel)

        # build adjacent mtx
        for i in range(NumGoodsample):
            print "i", i
            # plt.cla()
            for j in range(i+1, NumGoodsample):
                tmp1 = x[i,:]!=0
                tmp2 = x[j,:]!=0
                idx  = num[tmp1&tmp2]
                if len(idx)> Parameterobj.trjoverlap_len_thresh: # has overlapping
                    sidx   = idx[0:-1] # for speed
                    sxdiff,sydiff,mdis = get_spd_dis_diff(xspd[i,sidx],xspd[j,sidx],yspd[i,sidx],yspd[j,sidx],x[i,idx],x[j,idx],y[i,idx],y[j,idx])


                    if Parameterobj.useSBS:
                        huedis = np.mean(np.abs(hue[i,sidx]-hue[j,sidx]))
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

                        if np.isnan(centerDis):
                            # centerDis = mean_std_ForKernel[-2]+mean_std_ForKernel[-1] # treat as very far away, mu+sigma away
                            pdb.set_trace()
                    else:
                        huedis=0
                        centerDis=0
                    # print 'centerDis',centerDis
                    trj1 = [x[i,idx],y[i,idx]]
                    trj2 = [x[j,idx],y[j,idx]]
    
                    dataForKernel_ele = [sxdiff,sydiff,mdis,huedis,centerDis]
    
                    """counting the sharing blob numbers of two trjs"""
                    # if useSBS:
                    #     SBS[i,j] = sameBlobScore(np.array(FgBlobIndex[i,idx]),np.array(FgBlobIndex[j,idx]))
                    
                    adj[i,j] = get_adj_element(adj_methods,dataForKernel_ele,mean_std_ForKernel,extremeValue,Parameterobj.useSBS)

                else: # overlapping too short
                    # SBS[i,j] = 0
                    adj[i,j] = 0

        # SBS = SBS + SBS.transpose()  #add diag twice
        # np.fill_diagonal(SBS, np.diagonal(SBS)/2)

        adj = adj + adj.transpose() 
        np.fill_diagonal(adj, 1)
        
        print "saving adj..."
        sparsemtx = csr_matrix(adj)
        s,c       = connected_components(sparsemtx) #s is the total CComponent, c is the label
        result    = {}
        result['adj']     = sparsemtx
        result['c']       = c
        result['trjID']   = ptsidx

        if Parameterobj.useWarpped:
            savename = 'usewarpped_'+adj_methods+'_Adj_300_5_5_'+str(matidx+1+start_position_offset).zfill(3)
        else:
            savename = adj_methods+'_spd_max_value_'+str(matidx+1+start_position_offset).zfill(3)
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





