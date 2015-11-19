# after raw klt 
# filtering
import os
import math
import pdb,glob
import numpy as np
from scipy.io import loadmat,savemat
from scipy.sparse import csr_matrix

def trj_filter(x, y, t, xspeed, yspeed, blob_index, mask, Numsample , minspdth = 15, fps = 23):
    # fps for DoT Canal is 23
    # Jay & Johnson is 30

    transth  = 100*fps   #transition time (red light time) 100 seconds
    mask_re       = []
    x_re          = []
    y_re          = []
    t_re          = []
    blob_index_re = []
    xspd          = []
    yspd          = []

    speed = np.abs(xspeed)+np.abs(yspeed)
    print "minimum speed threshold set to be: ",minspdth
    # lenfile = open('length.txt', 'wb')
    # spdfile = open('./tempFigs/maxspeed.txt', 'wb')
    # stoptimefile = open('stoptime.txt', 'wb')
    for i in range(Numsample):
        if sum(x[i,:]!=0)>4:  # new jay st  # chk if trj is long enough
        # if sum(x[i,:]!=0)>50:  # canal
            # spdfile.write(str(i)+' '+str(max(speed[i,:][x[i,1:]!=0][1:-1]))+'\n')
            # lenfile.write(str(i)+' '+str(sum(x[i,:]!=0))+'\n')
            # pdb.set_trace()
            try:
                # spdfile.write(str(i)+' '+str(max(speed[i,:][x[i,1:]!=0][1:-1]))+'\n')
                if max(speed[i,:][x[i,1:]!=0][1:-1])>minspdth: # check if it is a moving point
                    # stoptimefile.write(str(i)+' '+str(sum(speed[i,:][x[i,1:]!=0][1:-1] < 3))+'\n')
                    if sum(speed[i,:][x[i,1:]!=0][1:-1] < 3) < transth:  # check if it is a stationary point

                        mask_re.append(mask[i]) # ID 
                        x_re.append(x[i,:])
                        y_re.append(y[i,:])
                        t_re.append(t[i,:])
                        blob_index_re.append(blob_index[i,:])
                        xspd.append(xspeed[i,:])
                        yspd.append(yspeed[i,:])
            except:
                pass
    # spdfile.close()
    # stoptimefile.close()
    pdb.set_trace()
    x_re          = np.array(x_re)
    y_re          = np.array(y_re)
    t_re          = np.array(t_re)
    blob_index_re = np.array(blob_index_re)
    xspd          = np.array(xspd)
    yspd          = np.array(yspd)
    return mask_re, x_re, y_re, t_re, blob_index_re, xspd, yspd



def prepare_input_data():
    # for linux
    matfilepath = '/media/My Book/CUSP/DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/klt/'
    savePath    = '/media/My Book/CUSP/DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/klt/filtered/'
    # for mac
    # matfilepath = '../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'
    # savePath    = '../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/filtered/'
    # matfilepath = '../tempFigs/roi2/'
    # savePath = '../tempFigs/roi2/filtered/' 
    matfiles = sorted(glob.glob(matfilepath + 'klt_*.mat'))
    # start_position = 0 #already processed 10 files
    # matfiles = matfiles[start_position:]
    return matfiles,savePath



if __name__ == '__main__':
    matfiles,savePath= prepare_input_data()
    for matidx,matfile in enumerate(matfiles):
        print "Processing truncation...", str(matidx+1)
        ptstrj = loadmat(matfile)
        mask = ptstrj['mask'][0]
        x    = csr_matrix(ptstrj['xtracks'], shape=ptstrj['xtracks'].shape).toarray()
        y    = csr_matrix(ptstrj['ytracks'], shape=ptstrj['ytracks'].shape).toarray()
        t    = csr_matrix(ptstrj['Ttracks'], shape=ptstrj['Ttracks'].shape).toarray()
        blob_index = csr_matrix(ptstrj['fg_blob_index'], shape=ptstrj['fg_blob_index'].shape).toarray()
        if len(t)>0: 
            t[t==np.max(t)]=np.nan

        
        Numsample = ptstrj['xtracks'].shape[0]
        trunclen  = ptstrj['xtracks'].shape[1]
        color     = np.array([np.random.randint(0,255) for _ in range(3*int(Numsample))]).reshape(Numsample,3)

        startPt = np.zeros((Numsample,1))
        endPt   = np.zeros((Numsample,1))

        for tt in range(Numsample):
            if len(t)>0:
                startPt[tt] =  np.mod( np.nanmin(t[tt,:]), trunclen) #ignore all nans
                endPt[tt]   =  np.mod( np.nanmax(t[tt,:]), trunclen) 
            else:
                startPt[tt] =  np.min(np.where(x[tt,:]!=0))
                endPt[tt]   =  np.max(np.where(x[tt,:]!=0))


        # xspeed = np.diff(x)*((x!=0)[:,1:])  # wrong!
        # yspeed = np.diff(y)*((y!=0)[:,1:])
        
        xspeed = np.diff(x) 
        yspeed = np.diff(y)

        for ii in range(Numsample):
            if math.isnan(startPt[ii]) or math.isnan(endPt[ii]):
                pdb.set_trace()
                xspeed[ii, :] = 0 # discard
                yspeed[ii, :] = 0 
            else:
                pdb.set_trace()
                xspeed[ii, int(max(startPt[ii]-1,0))] = 0 
                xspeed[ii, int(endPt[ii]-1)] = 0 
                yspeed[ii, int(max(startPt[ii]-1,0))] = 0 
                yspeed[ii, int(endPt[ii]-1)] = 0 
        
        
        
        print "Num of original samples is " , Numsample
        mask_re, x_re, y_re, t_re, blob_index_re, xspd,yspd = trj_filter(x, y, t, xspeed, yspeed, blob_index, mask, Numsample , minspdth = 1, fps = 4)
        print('initialization finished....')
        
        NumGoodsample = len(x_re)
        print "Num of Good samples is" , NumGoodsample

        result    = {}
        result['mask']          = mask_re
        result['xtracks']       = x_re       
        result['ytracks']       = y_re
        result['Ttracks']       = t_re
        result['xspd']          = xspd
        result['yspd']          = yspd
        result['fg_blob_index'] = blob_index_re

        savename = os.path.join(savePath,'len4overlap1trj_'+matfiles[matidx][-7:-4].zfill(3))
        savemat(savename,result)

