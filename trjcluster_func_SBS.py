import os
import cv2
import math
import pdb
import pickle
import numpy as np
import glob as glob
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.io import loadmat,savemat
from scipy.sparse.csgraph import connected_components

from utilities.inspectAdj import knn_graph
from sklearn.preprocessing import StandardScaler, Imputer
from trjComparison import get_spd_dis_diff, get_hue_diff, sameBlobScore

from DataPathclass import *
DataPathobj = DataPath(dataSource,VideoIndex)
from parameterClass import *
Parameterobj = parameter(dataSource,VideoIndex)

isVisualize = False
# adj_methods = "Thresholding"
adj_methods = "Gaussian"

class adjacencyMatrix(object):
    """construct adjcency matrix for each truncation"""
    def __init__(self):
        self.matfilepath = DataPathobj.smoothpath
        self.matfiles = sorted(glob.glob(self.matfilepath + '*.mat'))
        self.savePath = DataPathobj.adjpath
        self.result = {} #for the save in the end

    def prepare_input_data(self,matidx):
        print "Processing truncation...", str(matidx+1)
        self.ptstrj = loadmat(self.matfiles[matidx])
        """if no trj in this file, just continue"""
        assert len(self.ptstrj['trjID'])>0
        print 'total number of trjs in this trunk', len(self.ptstrj['trjID'])

        self.trjID = self.ptstrj['trjID'][0]
        self.hue = csr_matrix(self.ptstrj['Huetracks'], shape=self.ptstrj['Huetracks'].shape).toarray()

        if not Parameterobj.useWarpped:            
            self.x    = csr_matrix(self.ptstrj['xtracks'], shape=self.ptstrj['xtracks'].shape).toarray()
            self.y    = csr_matrix(self.ptstrj['ytracks'], shape=self.ptstrj['ytracks'].shape).toarray()
            self.t    = csr_matrix(self.ptstrj['Ttracks'], shape=self.ptstrj['Ttracks'].shape).toarray()
            self.xspd = csr_matrix(self.ptstrj['xspd'], shape=self.ptstrj['xspd'].shape).toarray()
            self.yspd = csr_matrix(self.ptstrj['yspd'], shape=self.ptstrj['yspd'].shape).toarray()
            self.Xdir = csr_matrix(self.ptstrj['Xdir'], shape=self.ptstrj['Xdir'].shape).toarray()
            self.Ydir = csr_matrix(self.ptstrj['Ydir'], shape=self.ptstrj['Ydir'].shape).toarray()
        else:
            self.x    = csr_matrix(self.ptstrj['xtracks_warpped'],shape=self.ptstrj['xtracks_warpped'].shape).toarray()
            self.y    = csr_matrix(self.ptstrj['ytracks_warpped'],shape=self.ptstrj['ytracks_warpped'].shape).toarray()
            self.t    = csr_matrix(self.ptstrj['Ttracks'],shape=self.ptstrj['Ttracks'].shape).toarray()
            self.xspd = csr_matrix(self.ptstrj['xspd_warpped'], shape=self.ptstrj['xspd_warpped'].shape).toarray()
            self.yspd = csr_matrix(self.ptstrj['yspd_warpped'], shape=self.ptstrj['yspd_warpped'].shape).toarray()
            self.Xdir = csr_matrix(self.ptstrj['Xdir_warpped'], shape=self.ptstrj['Xdir_warpped'].shape).toarray()
            self.Ydir = csr_matrix(self.ptstrj['Ydir_warpped'], shape=self.ptstrj['Ydir_warpped'].shape).toarray()

        if Parameterobj.useSBS:
            self.FgBlobIndex = csr_matrix(self.ptstrj['fg_blob_index'], shape=self.ptstrj['fg_blob_index'].shape).toarray()
            self.fg_blob_center_X = csr_matrix(self.ptstrj['fg_blob_center_X'], shape=self.ptstrj['fg_blob_center_X'].shape).toarray()
            self.fg_blob_center_Y = csr_matrix(self.ptstrj['fg_blob_center_Y'], shape=self.ptstrj['fg_blob_center_Y'].shape).toarray()

            self.FgBlobIndex[self.FgBlobIndex==0]=np.nan
            self.fg_blob_center_X[self.FgBlobIndex==0]=np.nan
            self.fg_blob_center_Y[self.FgBlobIndex==0]=np.nan
        else:
            self.fg_blob_center_X = np.ones(self.x.shape)*np.nan
            self.fg_blob_center_Y = np.ones(self.x.shape)*np.nan
            self.FgBlobIndex      = np.ones(self.x.shape)*np.nan

        self.Numsample = self.ptstrj['xtracks'].shape[0]
        self.fnum      = self.ptstrj['xtracks'].shape[1]
        
    def directionGroup(self ):
        """pre group based on directions"""
        upup = ((self.Xdir>=0)*(self.Ydir>=0))[0]
        upupind = np.array(range(self.Numsample))[upup]
        upupTrjID = self.trjID[upup]

        updown = ((self.Xdir>=0)*(self.Ydir<=0))[0]
        updownind = np.array(range(self.Numsample))[~upup*updown]
        updownTrjID = self.trjID[~upup*updown]

        downup = ((self.Xdir<=0)*(self.Ydir>=0))[0]
        downupind = np.array(range(self.Numsample))[~upup*(~updown)*downup]
        downupTrjID = self.trjID[~upup*(~updown)*downup]

        downdown = ((self.Xdir<=0)*(self.Ydir<=0))[0]
        downdownind = np.array(range(self.Numsample))[~upup*(~updown)*(~downup)*downdown]
        downdownTrjID = self.trjID[~upup*(~updown)*(~downup)*downdown]
        
        """ind is the location in this truncation, trjID is the real absolute ID"""
        self.DirInd = [upupind,updownind,downupind,downdownind]
        self.DirTrjID = [upupTrjID,updownTrjID,downupTrjID,downdownTrjID]
        assert len(upupind)+len(updownind)+len(downupind)+len(downdownind)==self.Numsample

        self.DirName = ['upup','updown','downup','downdown']


    def featureDiffTensorConstruct(self):
        self.feature_diff_tensor = np.ones([self.NumGoodsampleSameDir,self.NumGoodsampleSameDir,5])*np.nan
        for i in range(self.NumGoodsampleSameDir):
            print "i", i
            for j in range(i+1, self.NumGoodsampleSameDir):
                tmp1 = self.x[i,:]!=0
                tmp2 = self.x[j,:]!=0
                idx  = np.arange(self.fnum)[tmp1&tmp2]
                # trj1 = [self.x[i,idx],y[i,idx]]
                # trj2 = [self.x[j,idx],y[j,idx]]
                if len(idx)>= Parameterobj.trjoverlap_len_thresh: # has overlapping
                    sidx   = idx[1:] # for speeds
                    sxdiff,sydiff,mdis = get_spd_dis_diff(self.xspd[i,sidx],self.xspd[j,sidx],self.yspd[i,sidx],self.yspd[j,sidx], \
                        self.x[i,idx],self.x[j,idx],self.y[i,idx],self.y[j,idx])
                    huedis = get_hue_diff(self.hue[i,sidx],self.hue[j,sidx])
                    if Parameterobj.useSBS:
                        cxi = np.nanmedian(self.fg_blob_center_X[i,idx])
                        cyi = np.nanmedian(self.fg_blob_center_Y[i,idx])
                        cxj = np.nanmedian(self.fg_blob_center_X[j,idx])
                        cyj = np.nanmedian(self.fg_blob_center_Y[j,idx])

                        """if not inside a blob, use trj's own center"""
                        if np.isnan(cxi) or np.isnan(cyi): 
                            cxi = np.nanmedian(self.x[i,idx])
                            cyi = np.nanmedian(self.y[i,idx])
                        if np.isnan(cxj) or np.isnan(cyj):
                            cxj = np.nanmedian(self.x[j,idx])
                            cyj = np.nanmedian(self.y[j,idx])

                        """if not inside a blob, assign trj to nearest blob???"""
                        centerDis = np.sqrt((cxi-cxj)**2+(cyi-cyj)**2)

                        if np.isnan(centerDis):
                            # centerDis = mean_std_ForKernel[-2]+mean_std_ForKernel[-1] # treat as very far away, mu+sigma away
                            pdb.set_trace()
                    else:
                        centerDis=0
                    # print 'centerDis',centerDis

                    dataForKernel_ele = [sxdiff,sydiff,mdis,huedis,centerDis]
                    """counting the sharing blob numbers of two trjs"""
                    # if Parameterobj.useSBS:
                    #     SBS[i,j] = sameBlobScore(np.array(FgBlobIndex[i,idx]),np.array(FgBlobIndex[j,idx]))
                    
                    self.feature_diff_tensor[i,j,:] = dataForKernel_ele


    def adjConstruct(self):
        cumNumSample =  0 
        for dirii in range(4):
            if cumNumSample==self.Numsample: ## already done on all samples, no need to try other directions
                break
            sameDirInd = self.DirInd[dirii]
            NumGoodsampleSameDir = len(sameDirInd)
            cumNumSample += NumGoodsampleSameDir
            if NumGoodsampleSameDir==0:
                continue
            self.adjConstruct_EachDir(dirii)


    def adjConstruct_EachDir(self,directionInd):
        """contruct adj for each direction group"""
        sameDirInd = self.DirInd[directionInd]
        sameDirTrjID = self.DirTrjID[directionInd]
        self.NumGoodsampleSameDir = len(sameDirInd)  ## this will change every differnt direction group
        print self.DirName[directionInd] ,'same direction adj mtx ....', self.NumGoodsampleSameDir,'*',self.NumGoodsampleSameDir
        self.dataForKernel = np.array([self.x[sameDirInd,:],self.y[sameDirInd,:],self.xspd[sameDirInd,:],self.yspd[sameDirInd,:],self.hue[sameDirInd,:],self.fg_blob_center_X[sameDirInd,:],self.fg_blob_center_Y[sameDirInd,:]])
        # self.getRawDataFeatureMtx()
        
        if adj_methods =="Gaussian":
            if len(sorted(glob.glob(self.savePath+'extremeValue'+self.DirName[directionInd]+str(matidx+1).zfill(3))))>0:
                print "mean_std_ForKernel(ignore) and extremeValue already stored, load..."
                # self.mean_std_ForKernel = pickle.load(open(DataPathobj.adjpath+'mean_std_ForKernel'+self.DirName[directionInd]+str(matidx+1).zfill(3),'rb'))
                self.extremeValue = pickle.load(open(DataPathobj.adjpath+'extremeValue'+self.DirName[directionInd]+str(matidx+1).zfill(3),'rb'))
            else:     
                print "get extremeValue for normalization"         
                self.getMuSigma(directionInd)

        if Parameterobj.useSBS:
            SBS = np.zeros([self.NumGoodsampleSameDir,self.NumGoodsampleSameDir])

        """store all pair feature distances"""
        """Nsample*Nsample* 5 distance features"""
        feaName = 'feature_diff_tensor'

        if len(sorted(glob.glob(self.savePath+feaName+self.DirName[directionInd]+str(matidx+1).zfill(3))))>0:
            print "distance diff already stored, load..."
            self.feature_diff_tensor = pickle.load(open(self.savePath+feaName+self.DirName[directionInd]+str(matidx+1).zfill(3),'rb'))
        else:
            print "construct feature_diff_tensor, save..."
            self.featureDiffTensorConstruct()
            pickle.dump(self.feature_diff_tensor,open(self.savePath+feaName+self.DirName[directionInd]+str(matidx+1).zfill(3),'wb'))


        """test only for one car, see different features' role in the adj"""
        # self.diff_feature_on_one_car()
        """build adj mtx"""
        if adj_methods =="Thresholding":
            adj=self.get_thresholding_adj()

        if adj_methods =="Gaussian":
            self.get_gaussian_adj()

        # SBS = SBS + SBS.transpose()  #add diag twice
        # np.fill_diagonal(SBS, np.diagonal(SBS)/2)

        # np.fill_diagonal(adj, 1)
        """diagonal actually doesn't matter in Spectral Clustering"""
        np.fill_diagonal(self.adj, 0)

        """save each same direction adj"""
        temp = (self.adj>0).astype(int)
        s,c = connected_components(temp) #s is the total CComponent, c is the label
        
        """delete trjs that formed isolated very small CC"""
        self.non_isolatedCC = []
        for CClabel in np.unique(c):
            if len(np.where(c==CClabel)[0])>=3:
                self.non_isolatedCC+=list(np.where(c==CClabel)[0])

        self.adj = self.adj[self.non_isolatedCC,:][:,self.non_isolatedCC]
        self.c = c[self.non_isolatedCC]
        self.sameDirTrjID = sameDirTrjID[self.non_isolatedCC]
        
        sparsemtx = csr_matrix(self.adj)
        self.result['adj_'+self.DirName[directionInd]]   = sparsemtx
        self.result['c_'+self.DirName[directionInd]]     = self.c
        self.result['trjID_'+self.DirName[directionInd]] = self.sameDirTrjID
        self.result['non_isolatedCC'+self.DirName[directionInd]] = self.non_isolatedCC

        # ss,cc = connected_components((self.adj>0).astype(int)) #s is the total CComponent, c is the label
        # if ss>1:
        #     pdb.set_trace()


        """often, one vehicle is already over-segmented by CC, why"""
        # final_trjID   = np.uint32(loadmat(open(os.path.join(DataPathobj.unifiedLabelpath,'concomp'+'c_'+DirName[directionInd]+'.mat')))['trjID'][0]) 
        # final_cc_label = loadmat(open(os.path.join(DataPathobj.unifiedLabelpath,'concomp'+'c_'+DirName[directionInd]+'.mat')))['label'][0]

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


    def getMuSigma(self,directionInd):
        """loops! not efficient"""
        if len(self.dataForKernel)==7:
            [x,y,xspd,yspd,hue,fg_blob_center_X,fg_blob_center_Y] = self.dataForKernel
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
                idx  = np.arange(self.fnum)[tmp1&tmp2]
                
                if len(idx)>5: # has overlapping
                # if len(idx)>=30: # at least overlap for 100 frames
                    sidx   = idx[0:-1] # for speed
                    """use the max values, can also use the mean values"""
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

        """ignore mu, std for now, just use extreme values to normalize"""
        # self.mean_std_ForKernel = []
        # pickle.dump(self.mean_std_ForKernel,open(DataPathobj.adjpath+'mean_std_ForKernel'+self.DirName[directionInd]+str(matidx+1).zfill(3),'wb'))
        self.extremeValue = np.array([min(sxdiffAll[:]),max(sxdiffAll[:]), min(sydiffAll[:]),max(sydiffAll[:]),min(mdisAll[:]),max(mdisAll[:]),min(huedisAll[:]),max(huedisAll[:]),min(centerDisAll[:]),max(centerDisAll[:])])
        pickle.dump(self.extremeValue,open(DataPathobj.adjpath+'extremeValue'+self.DirName[directionInd]+str(matidx+1).zfill(3),'wb'))


    def get_thresholding_adj(self):
        self.adj = np.ones([self.NumGoodsampleSameDir,self.NumGoodsampleSameDir])
        [sxdiff,sydiff,mdis,huedis,centerDis]=self.feature_diff_tensor

        # yspdth = self.mean_std_ForKernel[2]+self.mean_std_ForKernel[3] #mean+sigma
        # xspdth = self.mean_std_ForKernel[0]+self.mean_std_ForKernel[1]
        xspdth = 0.5;
        yspdth = 0.5;

        self.adj = self.adj*(sxdiff <xspdth )*(sydiff<yspdth)
        self.adj[np.isnan(self.adj)] = 0
        self.adj = self.adj + self.adj.transpose() 


    def normalize_features(self):
        """use extreame values to normalize"""
        # [min_sx,max_sx,min_sy,max_sy,min_mdis,max_mdis,min_hue,max_hue,min_center,max_center]=self.extremeValue
        # sxdiff_normalized    = (self.feature_diff_tensor[:,:,0]-min_sx)/(max_sx-min_sx)
        # sydiff_normalized    = (self.feature_diff_tensor[:,:,1]-min_sy)/(max_sy-min_sy)
        # mdis_normalized      = (self.feature_diff_tensor[:,:,2]-min_mdis)/(max_mdis-min_mdis)
        # huedis_normalized    = (self.feature_diff_tensor[:,:,3]-min_hue)/(max_hue-min_hue)
        # centerDis_normalized = (self.feature_diff_tensor[:,:,4]-min_center)/(max_center-min_center)

        """within each thresholded CC, using the threshold to normalize"""
        sxdiff_normalized    = self.feature_diff_tensor[:,:,0]/Parameterobj.nullXspd_for_adj
        sydiff_normalized    = self.feature_diff_tensor[:,:,1]/Parameterobj.nullYspd_for_adj
        mdis_normalized      = self.feature_diff_tensor[:,:,2]/Parameterobj.nullDist_for_adj
        huedis_normalized    = self.feature_diff_tensor[:,:,3] ##ignore
        centerDis_normalized = self.feature_diff_tensor[:,:,4]/Parameterobj.nullBlob_for_adj

        return [sxdiff_normalized,sydiff_normalized,mdis_normalized,huedis_normalized,centerDis_normalized]


    def standard_scaler_normalization(self):
        """normalize across all samples for each feature"""
        for ii in range(5):
            data_feature_mtx = self.feature_diff_tensor[:,:,ii]
            """fillin in the NaN with axis mean"""
            imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
            data_feature_mtx = imp.fit_transform(data_feature_mtx)
            pdb.set_trace()
            scaler = StandardScaler().fit(data_feature_mtx)
            # pickle.dump(scaler,open('./clf/scaler','wb'))
            data_feature_mtx = scaler.transform(data_feature_mtx)
            "but.... there will be negative values, since assume mu=0"

            """normalize across all features for each sample"""
            # scaler = StandardScaler().fit(data_feature_mtx.T)
            # pickle.dump(scaler,open('./clf/scaler_across_all_fea','wb'))
            # data_feature_mtx = scaler.transform(data_feature_mtx.T).T
        

    def get_gaussian_adj(self):
        self.adj = np.zeros([self.NumGoodsampleSameDir,self.NumGoodsampleSameDir])

        """assign different weights to different features"""
        weight = Parameterobj.adj_weight

        """1. normalize the feature tensor:"""
        [sxdiff_normalized,sydiff_normalized,mdis_normalized,huedis_normalized,centerDis_normalized] = self.normalize_features()

        # fully_adj = np.exp(-sxdiff_normalized-sydiff_normalized-mdis_normalized-huedis_normalized-centerDis_normalized)
        fully_adj = np.exp(- (weight[0]*sxdiff_normalized+weight[1]*sydiff_normalized+weight[2]*mdis_normalized+weight[3]*huedis_normalized+weight[4]*centerDis_normalized))
        fully_adj[np.isnan(fully_adj)] = 0

        # """construct KNN graph from fully connected graph"""
        # adj_knn = self.knn_graph(knn = 20)
        self.fully_adj = fully_adj + fully_adj.transpose() 
        self.adj = self.fully_adj.copy()

        """2. Hard thresholding adj"""
        """based on spatial distance"""
        self.adj = self.adj*(self.feature_diff_tensor[:,:,2]< Parameterobj.nullDist_for_adj)

        """based on velocities"""
        self.adj = self.adj*(self.feature_diff_tensor[:,:,0]< Parameterobj.nullXspd_for_adj)
        self.adj = self.adj*(self.feature_diff_tensor[:,:,1]< Parameterobj.nullYspd_for_adj)

        """based on blob center dist"""
        # self.adj = self.adj*(self.feature_diff_tensor[:,:,4]< Parameterobj.nullBlob_for_adj)

        """based on hue dist"""
        """Hue info is very very weak, even with 0.001, still almost fully connected"""
        # self.adj = self.adj*(self.feature_diff_tensor[:,:,3]< 0.001)

        self.adj = self.adj + self.adj.transpose() 
        assert np.sum(self.adj==self.adj.T)==self.adj.shape[0]*self.adj.shape[1]


    def saveADJ(self,matidx):
        if Parameterobj.useWarpped:
            savename = 'usewarpped_'
        else:
            savename = ''        
        savename = savename+'Aug10_objoriented'+adj_methods+str(matidx+1).zfill(3)
        # savename = savename+'baseline_thresholding_adj_all'+adj_methods+'_diff_dir_'+str(matidx+1+start_position_offset).zfill(3)
        savename = os.path.join(self.savePath,savename)
        savemat(savename,self.result)


    def visCC(self):
        """fix me.... :/"""

        """to visualize the neighbours"""
        if isVisualize:
            fig888 = plt.figure()
            ax     = plt.subplot(1,1,1)

        """ visualization, see if connected components make sense"""
        s111,c111 = connected_components(sparsemtx) #s is the total CComponent, c is the label
        color     = np.array([np.random.randint(0,255) for _ in range(3*int(s111))]).reshape(s111,3)
        fig888    = plt.figure(888)
        ax        = plt.subplot(1,1,1)
        # im = plt.imshow(np.zeros([528,704,3]))
        for i in range(s111):
            ind = np.where(c111==i)[0]
            print ind
            for jj in range(len(ind)):
                startlimit = np.min(np.where(x[ind[jj],:]!=0))
                endlimit = np.max(np.where(x[ind[jj],:]!=0))
                # lines = ax.plot(x[ind[jj],startlimit:endlimit], y[ind[jj],startlimit:endlimit],color = (0,1,0),linewidth=2)
                lines = ax.plot(x[ind[jj],startlimit:endlimit], y[ind[jj],startlimit:endlimit],color = (color[i-1].T)/255.,linewidth=2)
                fig888.canvas.draw()
            plt.pause(0.0001) 
        plt.show()


if __name__ == '__main__':
    adjObj = adjacencyMatrix()
    for matidx in range(len(adjObj.matfiles)):
        print "building adj mtx ....", matidx
        adjObj.prepare_input_data(matidx)
        """First cluster using just direction Information"""
        adjObj.directionGroup()
        adjObj.adjConstruct()


        print "saving adj..."
        adjObj.saveADJ(matidx)
        """ visualization, see if connected components make sense"""







