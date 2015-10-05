
import os
from scipy.io import loadmat,savemat
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import numpy as np
import pdb,glob


def trjcluster(matfilepath = '../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/',\
    savePath = '../DoT/CanalSt@BaxterSt-96.106/adj/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'):

    matfiles = sorted(glob.glob(matfilepath + 'klt_*.mat'))

    for matidx,matfile in enumerate(matfiles):

        ptstrj = loadmat(matfile)
        x = csr_matrix(ptstrj['xtracks'], shape=ptstrj['xtracks'].shape).toarray()
        y = csr_matrix(ptstrj['ytracks'], shape=ptstrj['ytracks'].shape).toarray()
        t = csr_matrix(ptstrj['Ttracks'], shape=ptstrj['Ttracks'].shape).toarray()

        ptsidx = ptstrj['idxtable'][0]
        sample = ptstrj['xtracks'].shape[0]
        fnum   = ptstrj['xtracks'].shape[1]

        #x[x<0]=0
        #y[y<0]=x0


        xspeed = np.diff(x)*((x!=0)[:,1:])
        yspeed = np.diff(y)*((y!=0)[:,1:])
        speed = np.abs(xspeed)+np.abs(yspeed)
        # chk if trj is long enough
        mask = []
        x_re = []
        y_re = []
        t_re = []
        xspd = []
        yspd = []
        minspdth = 15 #threshold of min speed
        # minspdth = 5 #for Canal data

        fps = 4
        transth  = 60*fps   #transition time (red light time)


        print('initialization finished....')

        for i in range(sample):
            if sum(x[i,:]!=0)>4:  # chk if trj is long enough
                try:
                    if max(speed[i,:][x[i,1:]!=0][1:-1])>minspdth: # check if it is a moving point
                        if sum(speed[i,:][x[i,1:]!=0][1:-1] < 3) < transth:  # check if it is a stationary point

                            mask.append(ptsidx[i]) # ID 
                            x_re.append(x[i,:])
                            y_re.append(y[i,:])
                            t_re.append(t[i,:])

                            xspd.append(xspeed[i,:])
                            yspd.append(yspeed[i,:])
                except:
                    pass

        sample = len(x_re)
        adj    = np.zeros([sample,sample])
        dth    = 30*1.5
        spdth  = 5
        num    = np.arange(fnum)
        x_re   = np.array(x_re)
        y_re   = np.array(y_re)
        t_re   = np.array(t_re)
        
        xspd = np.array(xspd)
        yspd = np.array(yspd)

        print(sample)
        print('building adj mtx ....')

        # build adjacent mtx
        for i in range(sample):
            print i
            for j in range(sample):
                if i<j:
                    tmp1 = x_re[i,:]!=0
                    tmp2 = x_re[j,:]!=0
                    idx  = num[tmp1&tmp2]
                    if len(idx)>0:
                        sidx = idx[1:-1]
                        sxdiff = np.mean(np.abs(xspd[i,sidx]-xspd[j,sidx]))
                        sydiff = np.mean(np.abs(yspd[i,sidx]-yspd[j,sidx]))
                        if (sxdiff <spdth ) & (sydiff<spdth ):
                            mdis = np.mean(np.abs(x_re[i,idx]-x_re[j,idx])+np.abs(y_re[i,idx]-y_re[j,idx]))
                            #mahhattan distance
                            if mdis < dth:
                                adj[i,j] = 1
                else:
                    adj[i,j] = adj[j,i]

        sparsemtx = csr_matrix(adj)
        s,c       = connected_components(sparsemtx)
        result    = {}
        result['adj']     = adj
        result['c']       = c
        result['mask']    = mask
        result['Ttracks'] = t_re

        savename = os.path.join(savePath,'trj_'+str(matidx+1).zfill(3))

        savemat(savename,result)
