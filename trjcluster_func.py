import os
from scipy.io import loadmat,savemat
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import numpy as np
import pdb,glob
import matplotlib.pyplot as plt
import math


def trj_filter(x, y, t, xspeed, yspeed, speed, ptsidx, Numsample , minspdth = 15, fps = 4):
    transth  = 60*fps   #transition time (red light time)
    mask = []
    x_re = []
    y_re = []
    t_re = []
    xspd = []
    yspd = []
    print minspdth
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

                        mask.append(ptsidx[i]) # ID 
                        x_re.append(x[i,:])
                        y_re.append(y[i,:])
                        t_re.append(t[i,:])

                        xspd.append(xspeed[i,:])
                        yspd.append(yspeed[i,:])
            except:
                pass
    # spdfile.close()
    # stoptimefile.close()
    # pdb.set_trace()
    x_re   = np.array(x_re)
    y_re   = np.array(y_re)
    t_re   = np.array(t_re)
    xspd = np.array(xspd)
    yspd = np.array(yspd)
    return mask, x_re, y_re, t_re, xspd, yspd



def prepare_input_data(isAfterWarpping,isLeft=True):
    if isAfterWarpping:
        if isLeft:
            matPath = '../DoT/CanalSt@BaxterSt-96.106/leftlane/'
            matfiles = sorted(glob.glob(matPath +'warpped_'+'*.mat'))
            savePath = '../DoT/CanalSt@BaxterSt-96.106/leftlane/adj/'
        else:
            matPath = '../DoT/CanalSt@BaxterSt-96.106/rightlane/'
            matfiles = sorted(glob.glob(matPath +'warpped_'+'*.mat'))
            savePath = '../DoT/CanalSt@BaxterSt-96.106/rightlane/adj/'
    else:
        matfilepath = '../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'
        savePath = '../DoT/CanalSt@BaxterSt-96.106/adj/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'
        # matfilepath = './tempFigs/roi2/'
        # savePath = './tempFigs/roi2/' 
        matfiles = sorted(glob.glob(matfilepath + 'klt_*.mat'))

    return matfiles,savePath





# def trjcluster(matfilepath = '../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/',\
#     savePath = '../DoT/CanalSt@BaxterSt-96.106/adj/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'):
if __name__ == '__main__':

    isAfterWarpping   = True
    isLeft            = False
    matfiles,savePath = prepare_input_data(isAfterWarpping,isLeft)

    for matidx,matfile in enumerate(matfiles):
        print "Processing truncation...", str(matidx+1)
        ptstrj = loadmat(matfile)
        x = csr_matrix(ptstrj['xtracks'], shape=ptstrj['xtracks'].shape).toarray()
        y = csr_matrix(ptstrj['ytracks'], shape=ptstrj['ytracks'].shape).toarray()
        t = csr_matrix(ptstrj['Ttracks'], shape=ptstrj['Ttracks'].shape).toarray()
        if len(t)>0: 
            t[t==0]=nan

        ptsidx    = ptstrj['mask'][0]
        Numsample = ptstrj['xtracks'].shape[0]
        fnum      = ptstrj['xtracks'].shape[1]

        startPt = np.zeros((Numsample,1))
        endPt = np.zeros((Numsample,1))

        for tt in range(Numsample):
            if len(t)>0:
                startPt[tt] =  mod( np.nanmin(t[tt,:]), 600) #ignore all nans
                endPt[tt]   =  mod( np.nanmax(t[tt,:]), 600) 
            else:
                startPt[tt] =  np.min(np.where(x[tt,:]!=0))
                endPt[tt]   =  np.max(np.where(x[tt,:]!=0))


        # """to visualize the neighbours"""
        # fig888 = plt.figure()
        # ax = plt.subplot(1,1,1)
        # color = np.array([np.random.randint(0,255) for _ in range(3*int(Numsample))]).reshape(Numsample,3)



        # xspeed = np.diff(x)*((x!=0)[:,1:])  # wrong!
        # yspeed = np.diff(y)*((y!=0)[:,1:])
        
        xspeed = np.diff(x) 
        yspeed = np.diff(y)

        for ii in range(Numsample):
            if math.isnan(startPt[ii]) or math.isnan(endPt[ii]):
                xspeed[ii, :] = 0 # discard
                yspeed[ii, :] = 0 
            else:
                xspeed[ii, int(max(startPt[ii]-1,0))] = 0 
                xspeed[ii, int(endPt[ii]-1)] = 0 
                yspeed[ii, int(max(startPt[ii]-1,0))] = 0 
                yspeed[ii, int(endPt[ii]-1)] = 0 


        
        speed = np.abs(xspeed)+np.abs(yspeed)
        
        print "Num of original samples is" , Numsample
        if isAfterWarpping:
            x_re = x
            y_re = y
            t_re = []
            xspd = xspeed
            yspd = yspeed
            mask = ptsidx
            print('initialization finished....')
        else:
            mask, x_re, y_re, t_re, xspd,yspd = trj_filter(x, y, t, xspeed, yspeed, speed, ptsidx, Numsample , minspdth = 1, fps = 4)
        print('initialization finished....')




        NumGoodsample = len(x_re)
        adj    = np.zeros([NumGoodsample,NumGoodsample])
        dth    = 30*1.5
        spdth  = 5 #speed threshold
        locth = 100
        num    = np.arange(fnum)


        print "Num of Good samples is" , NumGoodsample
        # print "min(x):", np.min(x), " max(x):",  np.max(x)
        print "min(y):", np.min(y), " max(y):",  np.max(y)

        print('building adj mtx ....')
        # build adjacent mtx
        for i in range(NumGoodsample):
            # plt.cla()

            for j in range(NumGoodsample):
                if i<j:
                    tmp1 = x_re[i,:]!=0
                    tmp2 = x_re[j,:]!=0
                    idx  = num[tmp1&tmp2]
                    if len(idx)>0: # has overlapping
                    # if len(idx)>=30: # at least overlap for 100 frames
                        sidx     = idx[1:-1]
                        sxdiff   = np.mean(np.abs(xspd[i,sidx]-xspd[j,sidx]))
                        sydiff   = np.mean(np.abs(yspd[i,sidx]-yspd[j,sidx]))
                        
                        if (sxdiff <spdth ) & (sydiff<spdth ):
                            mdis = np.mean(np.abs(x_re[i,idx]-x_re[j,idx])+np.abs(y_re[i,idx]-y_re[j,idx]))
                            #mahhattan distance
                            if mdis < dth:
                                adj[i,j] = 1
                                """visualize the neighbours"""
                                # lines = ax.plot(x_re[i,idx], y_re[i,idx],color = (color[i-1].T)/255.,linewidth=2)
                                # lines = ax.plot(x_re[j,idx], y_re[j,idx],color = (color[j-1].T)/255.,linewidth=2)
                                # fig888.canvas.draw()
                                # plt.pause(0.0001)
                else:
                    adj[i,j] = adj[j,i]

        sparsemtx = csr_matrix(adj)
        s,c       = connected_components(sparsemtx) #s is the total CComponent, c is the label
        result    = {}
        result['adj']     = adj
        result['c']       = c
        result['mask']    = mask
        result['xtracks'] = x_re       
        result['ytracks'] = y_re
        result['Ttracks'] = t_re
        result['xspd']    = xspd
        result['yspd']    = yspd

        if not isAfterWarpping:
            savename = os.path.join(savePath,'len4overlap1trj_'+str(matidx+1).zfill(3))
            savemat(savename,result)
        else:
            savename = os.path.join(savePath,'warpped_Adj_'+str(matidx+1).zfill(3))
            savemat(savename,result)
            

        """ visualization """
        # pdb.set_trace()        
        # s111,c111 = connected_components(sparsemtx) #s is the total CComponent, c is the label
        # color = np.array([np.random.randint(0,255) for _ in range(3*int(s111))]).reshape(s111,3)
        # fig888 = plt.figure(888)
        # ax = plt.subplot(1,1,1)
        # # im = plt.imshow(np.zeros([528,704,3]))
        # for i in range(s111):
        #     ind = np.where(c111 ==i)[0]
        #     print ind
        #     for jj in range(len(ind)):
        #         startlimit = np.min(np.where(x_re[ind[jj],:]!=0))
        #         endlimit = np.max(np.where(x_re[ind[jj],:]!=0))
        #         # lines = ax.plot(x_re[ind[jj],startlimit:endlimit], y_re[ind[jj],startlimit:endlimit],color = (0,1,0),linewidth=2)
        #         lines = ax.plot(x_re[ind[jj],startlimit:endlimit], y_re[ind[jj],startlimit:endlimit],color = (color[i-1].T)/255.,linewidth=2)
        #         fig888.canvas.draw()
        #     plt.pause(0.0001) 
        # pdb.set_trace()




 # lines = ax.plot(x[980,np.where(x[980,:]!=0)], y[980,np.where(x[980,:]!=0)],color = (1,1,0))
