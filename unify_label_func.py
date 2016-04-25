import os
import pdb
import glob as glob
import cPickle as pickle
import numpy as np
from scipy.io import loadmat,savemat
from DataPathclass import *
DataPathobj = DataPath(dataSource,VideoIndex)
from parameterClass import *
Parameterobj = parameter(dataSource,VideoIndex)


def unify_label(matfiles,savename,label_choice):
    DirName = ['upup','updown','downup','downdown']
    for dirii in range(4):
        atmp     = []
        flab     = [] #final labels
        ftrjID   = [] #final trjID
        LabelName = label_choice+DirName[dirii]
        IDName = 'trjID_'+DirName[dirii]
        if useCC:
            LabelName = 'c_'+DirName[dirii]
            IDName = 'trjID_'+DirName[dirii]
        
        try:
            M1 = loadmat(matfiles[0])[IDName][0]
        except: ## if no this direction, key error, continue
            continue
        for matidx in range(len(matfiles)-1): 
            if matidx == 0:
                # L1 = loadmat(matfiles[matidx])[label_choice][0]
                # L1 = L1+1 # class label starts from 1 instead of 0
                # M1 = loadmat(matfiles[matidx])['trjID'][0]
                L1 = loadmat(matfiles[matidx])[LabelName][0]
                L1 = L1+1 # class label starts from 1 instead of 0
                M1 = loadmat(matfiles[matidx])[IDName][0]

                # L1 = loadmat(matfiles[matidx])['newlabel'][0]
                # L1 = L1+1
                # M1 = loadmat(matfiles[matidx])['newtrjID'][0]
            else:
                L1 = L2
                M1 = M2

            # L2 = loadmat(matfiles[matidx+1])['labels_DPGMMupup'][0]
            L2 = loadmat(matfiles[matidx+1])[LabelName][0]
            L2 = L2+1
            M2 = loadmat(matfiles[matidx+1])[IDName][0]
            # L2 = loadmat(matfiles[matidx+1])['newlabel'][0]
            # L2 = L2+1
            # M2 = loadmat(matfiles[matidx+1])['newtrjID'][0]

            L1max = max(L1)  ## not duplicate
            L2[:] = L2 + L1max + 1 # to make sure there're no duplicate labels

            commonidx = np.intersect1d(M1,M2)  #trajectories existing in both 2 trucations
            
            print('L1 : {0}, L2 : {1} ,common term : {2}').format(len(np.unique(L1)),len(np.unique(L2)),len(commonidx))
       
            for i in commonidx:
                if i not in atmp:
                    # label1    = L1[np.where((M1 == i)!=0)[0][0]]  # use np.arange(len(M1))[M1 == i]
                    # label2    = L2[np.where((M2 == i)!=0)[0][0]]
                    label1 = L1[M1==i][0]
                    label2 = L2[M2==i][0]
                    if label2<=L1max: # already been united with a previous chunk label1
                        # print "previous split, now as one."
                        L1[M1==i] = label2
                        continue

                    idx1  = np.where(L1==label1)
                    idx2  = np.where(L2==label2)
                    tmp1  = np.intersect1d(M1[idx1],commonidx) # appear in both trunks and label is label1 
                    tmp2  = np.intersect1d(M2[idx2],commonidx)
                    L1idx =list(idx1[0])
                    L2idx =list(idx2[0])
                    # diff1     = np.setdiff1d(tmp1,tmp2)  # this difference only accounts for elements in tmp1
                    # diff      = np.union1d(np.setdiff1d(tmp1,np.intersect1d(tmp1,tmp2)) , np.setdiff1d(tmp2,np.intersect1d(tmp1,tmp2) ))
                    atmp      = np.unique(np.hstack((tmp1,tmp2)))
                    L1[L1idx] = label1  ## keep the first appearing label
                    L2[L2idx] = label1   

            if matidx == 0 :
                flab[:]   = flab + list(L1) 
                ftrjID[:] = ftrjID + list(M1)
            flab[:]   = flab +list(L2)
            ftrjID[:] = ftrjID + list(M2)

        #== eliminate duplicate part == 
           
        data      = np.vstack((ftrjID,flab))
        result    = data[:,data[0].argsort()]
        labels    = list(result[1])
        savetrjID = list(result[0])
        dpidx     = np.where(np.diff(sorted(savetrjID)) == 0)[0]  #find duplicate ID in trjID
        

        for k in dpidx[::-1]:
            labels.pop(k)
            savetrjID.pop(k)

        result          = {}
        result['label'] = labels
        result['trjID'] = savetrjID
        savename = os.path.join(DataPathobj.unifiedLabelpath,savename+label_choice)
        savemat(savename+LabelName,result)




if __name__ == '__main__':
    # 
    """to visulize the connected component"""
    global useCC
    useCC = False
    # global useRawSmooth
    # useRawSmooth = True


    # if useRawSmooth:
    #     matfilePath = DataPathobj.smoothpath
    if useCC:
        matfilePath = DataPathobj.adjpath
    else:
        matfilePath = DataPathobj.sscpath

    label_choice = Parameterobj.clustering_choice
    if useCC:
        # matfilesAll = sorted(glob.glob(matfilePath +'*knn&thresh*.mat'))
        # matfilesAll = sorted(glob.glob(matfilePath +'*onlyBlobThresh*.mat'))
        # matfilesAll = sorted(glob.glob(matfilePath +'*SpaSpdBlobthresh*.mat')) #thresholded by spatial dis, spd dis and blob center dis
        # matfilesAll = sorted(glob.glob(matfilePath +'*NoBlobThreshGaussian_diff_dir*.mat')) 
        matfilesAll = sorted(glob.glob(matfilePath +'*thresholding_adj_all_G*.mat')) 
        # matfilesAll = sorted(glob.glob(matfilePath +'*thresholding_adj_spatial_*.mat')) 

    else:
        matfilesAll = sorted(glob.glob(matfilePath +'*.mat'))

    if Parameterobj.useWarpped:
        matfilesAll = sorted(glob.glob(matfilePath +'usewarpped_*.mat'))    

    numTrunc = len(matfilesAll)
    savename = ''
    if numTrunc<=200:
        if useCC:
            savename = 'concomp'+savename
        else:
            savename = 'Complete_result'+savename
        if Parameterobj.useWarpped:
            savename = 'usewarpped_'+savename

        unify_label(matfilesAll,savename,label_choice)
    else:
        for kk in range(0,numTrunc,25):
            print "saved trunk",str(kk+1).zfill(3),'to' ,str(min(kk+25,numTrunc)).zfill(3)
            matfiles = matfilesAll[kk:min(kk+25,numTrunc)]
            savename = os.path.join(DataPathobj.unifiedLabelpath,'result_'+label_choice+str(kk+1).zfill(3)+'-'+str(min(kk+25,numTrunc)).zfill(3))
            unify_label(matfiles,savename,label_choice)
            



