import os
import pdb
import glob as glob
import cPickle as pickle
import numpy as np
from scipy.io import loadmat,savemat


def unify_label(matfiles,savename):
    atmp     = []
    flab     = [] #final labels
    ftrjID   = [] #final trjID


    for matidx in range(len(matfiles)-1): 
        if matidx == 0:
            L1 = loadmat(matfiles[matidx])['label'][0]
            L1 = L1+1 # class label starts from 1 instead of 0
            M1 = loadmat(matfiles[matidx])['trjID'][0]
        else:
            L1 = L2
            M1 = M2

        L2 = loadmat(matfiles[matidx+1])['label'][0]
        L2 = L2+1
        M2 = loadmat(matfiles[matidx+1])['trjID'][0]


        labelnum = max(L1)  ## not duplicate
        L2[:]    = L2 + labelnum+1 # to make sure there're no duplicate labels

        pdb.set_trace()
        commonidx = np.intersect1d(M1,M2)  #trajectories existing in both 2 trucations
        
        print('L1 : {0}, L2 : {1} ,common term : {2}').format(len(np.unique(L1)),len(np.unique(L2)),len(commonidx))
   
        for i in commonidx:
            if i not in atmp:
                # label1    = L1[np.where((M1 == i)!=0)[0][0]]  # use np.arange(len(M1))[M1 == i]
                # label2    = L2[np.where((M2 == i)!=0)[0][0]]
                label1 = L1[M1==i][0]
                label2 = L2[M2==i][0]

                idx1      = np.where(L1==label1)
                idx2      = np.where(L2==label2)
                tmp1      = np.intersect1d(M1[idx1],commonidx) # appear in both trunks and label is label1 
                tmp2      = np.intersect1d(M2[idx2],commonidx)
                # pdb.set_trace()
                L1idx     =list(idx1[0])
                L2idx     =list(idx2[0])
                diff1     = np.setdiff1d(tmp1,tmp2)  # this difference only accounts for elements in tmp1
                diff      = np.union1d(np.setdiff1d(tmp1,np.intersect1d(tmp1,tmp2)) , np.setdiff1d(tmp2,np.intersect1d(tmp1,tmp2) ))
                atmp      = np.unique(np.hstack((tmp1,tmp2)))
                L1[L1idx] = label1  ## keep the first appearing label
                L2[L2idx] = label1   

        pdb.set_trace()
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
    savemat(savename,result)




if __name__ == '__main__':
    dataSource = 'Johnson'
# def unify_label_main(dataSource):
    if dataSource == 'DoT':
        # for linux
        matfilePath = '/media/My Book/CUSP/AIG/DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/ssc/'
        savePath    = '/media/My Book/CUSP/AIG/DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'
        # for mac
        # matfilePath = '../DoT/CanalSt@BaxterSt-96.106/...???'
        # savePath    = '../DoT/CanalSt@BaxterSt-96.106/finalresult/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'
    if dataSource == 'Johnson':
        # Jay & Johnson
        matfilePath ='/media/My Book/CUSP/AIG/Jay&Johnson/roi2/ssc/'
        savePath    = '/media/My Book/CUSP/AIG/Jay&Johnson/roi2/'

    matfilesAll = sorted(glob.glob(matfilePath +'*.mat'))
    numTrunc    = len(matfilesAll)

    if numTrunc<=100:
        savename = os.path.join(savePath,'Complete_result')
        unify_label(matfilesAll,savename)
    else:
        for kk in range(0,numTrunc,25):
            print "saved trunk",str(kk+1).zfill(3),'to' ,str(min(kk+25,numTrunc)).zfill(3)
            matfiles = matfilesAll[kk:min(kk+25,numTrunc)]
            savename = os.path.join(savePath,'result_'+str(kk+1).zfill(3)+'-'+str(min(kk+25,numTrunc)).zfill(3))
            unify_label(matfiles,savename)
            



