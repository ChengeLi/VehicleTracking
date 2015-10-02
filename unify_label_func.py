from scipy.io import loadmat,savemat
import pdb,glob
import cPickle as pickle
import numpy as np


def unify_label(matfilePath = '../DoT/CanalSt@BaxterSt-96.106/labels/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/',\
    savename = '../DoT/CanalSt@BaxterSt-96.106/finalresult/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/result' ):

    num = 0
    atmp = []
    flab = []  #final labels
    fmask = [] #final mask

    matfiles = sorted(glob.glob(matfilePath +'*.mat'))

    for matidx in range(len(matfiles)-1): 
        if matidx == 0:
            L1 = loadmat(matfiles[matidx])['label'][0]
            M1 = loadmat(matfiles[matidx])['mask'][0]
        else:
            L1 = L2
            M1 = M2

        L2 = loadmat(matfiles[matidx+1])['label'][0]
        M2 = loadmat(matfiles[matidx+1])['mask'][0]

        labelnum = max(L1)  ## not duplicate
        L2[:] = L2 + labelnum+1 # to make sure there're no duplicate labels

        commonidx = np.intersect1d(M1,M2)  #trajectories existing in both 2 trucations
        
        print('L1 : {0}, L2 : {1} ,common term : {2}').format(len(np.unique(L1)),len(np.unique(L2)),len(commonidx))
   
        for i in commonidx:
            if i not in atmp:
                label1 = L1[np.where((M1 == i)!=0)[0][0]]  # use np.arange(len(M1))[M1 == i]
                label2 = L2[np.where((M2 == i)!=0)[0][0]]
                idx1 = np.where(L1==label1)
                idx2 = np.where(L2==label2)
                tmp1 = np.intersect1d(M1[idx1],commonidx) # appear in both trunks and label is label1 
                tmp2 = np.intersect1d(M2[idx2],commonidx)

                L1idx =list(idx1[0])
                L2idx =list(idx2[0])
                diff1 = np.setdiff1d(tmp1,tmp2)  # this difference only accounts for elements in tmp1
                diff = np.union1d(np.setdiff1d(tmp1,np.intersect1d(tmp1,tmp2)) , np.setdiff1d(tmp2,np.intersect1d(tmp1,tmp2) ))
                atmp = np.unique(np.hstack((tmp1,tmp2)))
                L1[L1idx] = label1  ## keep the first appearing label
                L2[L2idx] = label1   


        if matidx == 0 :
            flab[:] = flab + list(L1) 
            fmask[:] = fmask + list(M1)
        
        flab[:] = flab +list(L2)
        fmask[:] = fmask + list(M2)

    #== eliminate duplicate part == 
       
    data = np.vstack((fmask,flab))
    result = data[:,data[0].argsort()]
    labels = list(result[1])
    mask   = list(result[0])
    dpidx = np.where(np.diff(sorted(mask)) == 0)[0]  #find duplicate ID in mask
        

    for k in dpidx[::-1]:
        labels.pop(k)
        mask.pop(k)

    result = {}
    result['label']   = labels
    result['mask']= mask



    savemat(savename,result)
















