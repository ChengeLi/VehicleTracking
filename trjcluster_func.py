import os
from scipy.io import loadmat,savemat
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import numpy as np
import pdb,glob
import matplotlib.pyplot as plt
import math


def adj_gaussian_element(sxdiff, sydiff, mdis):
    sigma_xspd_diff = 0.7
    sigma_yspd_diff = 0.7
    sigma_spatial_distance = 200
    adj_element = np.exp((-sxdiff**2/sigma_xspd_diff**2)+(-sydiff**2/sigma_yspd_diff**2)+(-mdis**2/sigma_spatial_distance**2))
    return adj_element


def adj_cosine_element(i_trj,j_trj):
    # cosine similarity
    cos_element = np.dot(i_trj,j_trj)/np.sqrt(sum(abs(i_trj)**2))/np.sqrt(sum(abs(j_trj)**2))
    cos_element = (cos_element+1)/2 # make them all positive  
    print "cos_element: ", str(cos_element)
    if math.isnan(cos_element):
        pdb.set_trace()
    return cos_element



def construct_adj_gaussian(NumGoodsample, x_re, y_re):
    adj = np.zeros([NumGoodsample,NumGoodsample])
    num = np.arange(fnum)
    print('using gaussian kernel...')
    for i in range(NumGoodsample):
        for j in range(NumGoodsample):
            if i<=j:            
                tmp1 = x_re[i,:]!=0
                tmp2 = x_re[j,:]!=0
                idx  = num[tmp1&tmp2]
                if len(idx)>3: # has overlapping
                # if len(idx)>=30: # at least overlap for 100 frames
                    sidx     = idx[1:-1]
                    sxdiff   = np.mean(np.abs(xspd[i,sidx]-xspd[j,sidx]))
                    sydiff   = np.mean(np.abs(yspd[i,sidx]-yspd[j,sidx]))
                    mdis = np.mean(np.abs(x_re[i,idx]-x_re[j,idx])+np.abs(y_re[i,idx]-y_re[j,idx]))

                    adj[i,j] = adj_gaussian_element(sxdiff, sydiff, mdis)
                    if math.isnan(adj[i,j]):
                        pdb.set_trace()
                """visualize the neighbours"""
                # lines = ax.plot(x_re[i,idx], y_re[i,idx],color = (color[i-1].T)/255.,linewidth=2)
                # lines = ax.plot(x_re[j,idx], y_re[j,idx],color = (color[j-1].T)/255.,linewidth=2)
                # fig888.canvas.draw()
                # plt.pause(0.0001)
            else:
                adj[i,j] = adj[j,i]
    return adj

def construct_adj_cosine(NumGoodsample, x_re, y_re):
    adj = np.zeros([NumGoodsample,NumGoodsample])
    num = np.arange(fnum)
    print('using gaussian kernel...')
    for i in range(NumGoodsample):
        for j in range(NumGoodsample):
            if i<=j:            
                tmp1 = x_re[i,:]!=0
                tmp2 = x_re[j,:]!=0
                idx  = num[tmp1&tmp2]
                if len(idx)>0: # has overlapping
                # if len(idx)>=30: # at least overlap for 100 frames
                    sidx = idx[1:-1]
                    if len(sidx)<1:
                        continue
                    i_trj = np.concatenate((x_re[i,idx], xspd[i,sidx],y_re[i,idx],yspd[i,sidx]), axis=1)
                    j_trj = np.concatenate((x_re[j,idx], xspd[j,sidx],y_re[j,idx],yspd[j,sidx]), axis=1)
                    adj[i,j] = adj_cosine_element(i_trj,j_trj)

                """visualize the neighbours"""
                # lines = ax.plot(x_re[i,idx], y_re[i,idx],color = (color[i-1].T)/255.,linewidth=2)
                # lines = ax.plot(x_re[j,idx], y_re[j,idx],color = (color[j-1].T)/255.,linewidth=2)
                # fig888.canvas.draw()
                # plt.pause(0.0001)
            else:
                adj[i,j] = adj[j,i]
    return adj




def construct_adj_thresholding(NumGoodsample, x_re, y_re):
    print('Building Binary adj mtx after thresholing the speed and spatial location differences.')
    adj     = np.zeros([NumGoodsample,NumGoodsample])
    # spdfile = open('./mdis.txt', 'wb')
    # dth     = 300 #30*1.5
    # yspdth  = 0.7 #0.9 for warpped #5 #y speed threshold
    # xspdth  = 0.7 #0.9 for warpped #5 #x speed threshold
    
    dth     = 50+200
    yspdth  = 5 #y speed threshold
    xspdth  = 5 #x speed threshold

    num     = np.arange(fnum)
    # build adjacent mtx
    for i in range(NumGoodsample):
        # plt.cla()
        for j in range(i+1, min(NumGoodsample,i+1500)):
            tmp1 = x_re[i,:]!=0
            tmp2 = x_re[j,:]!=0
            idx  = num[tmp1&tmp2]
            if len(idx)>5: # has overlapping
            # if len(idx)>=30: # at least overlap for 100 frames
                sidx     = idx[1:-1]
                sxdiff   = np.mean(np.abs(xspd[i,sidx]-xspd[j,sidx]))
                sydiff   = np.mean(np.abs(yspd[i,sidx]-yspd[j,sidx]))
                # spdfile.write(str(i)+' '+str(sxdiff)+'\n')

                if (sxdiff <xspdth ) & (sydiff<yspdth ):
                    mdis = np.mean(np.abs(x_re[i,idx]-x_re[j,idx])+np.abs(y_re[i,idx]-y_re[j,idx]))
                    # spdfile.write(str(i)+' '+str(j)+' '+str(mdis)+'\n')
                    # print "mdis: ", mdis
                    #mahhattan distance
                    if mdis < dth:
                        adj[i,j] = 1
                        # adj[i,j] = construct_adj(sxdiff, sydiff, mdis)
                        """visualize the neighbours"""
                        # lines = ax.plot(x_re[i,idx], y_re[i,idx],color = (color[i-1].T)/255.,linewidth=2)
                        # lines = ax.plot(x_re[j,idx], y_re[j,idx],color = (color[j-1].T)/255.,linewidth=2)
                        # fig888.canvas.draw()
                        # plt.pause(0.0001)
    # spdfile.close()
    adj = adj + adj.transpose()
    np.fill_diagonal(adj, 1)
    return adj



def prepare_input_data(isAfterWarpping,isLeft=True):
    if isAfterWarpping:
        if isLeft:
            matPath  = '../DoT/CanalSt@BaxterSt-96.106/leftlane/'
            matfiles = sorted(glob.glob(matPath +'warpped_'+'*.mat'))
            savePath = '../DoT/CanalSt@BaxterSt-96.106/leftlane/adj/'
        else:
            matPath  = '../DoT/CanalSt@BaxterSt-96.106/rightlane/'
            matfiles = sorted(glob.glob(matPath +'warpped_'+'*.mat'))
            savePath = '../DoT/CanalSt@BaxterSt-96.106/rightlane/adj/'
    else:
        # matfilepath   = '../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'
        # savePath      = '../DoT/CanalSt@BaxterSt-96.106/adj/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'
        matfilepath = '../tempFigs/roi2/filtered/'
        savePath    = '../tempFigs/roi2/' 
        matfiles    = sorted(glob.glob(matfilepath + 'len*.mat'))

    return matfiles,savePath





# def trjcluster(matfilepath = '../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/',\
#     savePath = '../DoT/CanalSt@BaxterSt-96.106/adj/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'):
if __name__ == '__main__':
    isAfterWarpping   = False
    isLeft            = False
    matfiles,savePath = prepare_input_data(isAfterWarpping,isLeft)

    # """to visualize the neighbours"""
    fig888 = plt.figure()
    ax = plt.subplot(1,1,1)


    for matidx,matfile in enumerate(matfiles):
        print "Processing truncation...", str(matidx+1)
        ptstrj = loadmat(matfile)
        x = csr_matrix(ptstrj['xtracks'], shape=ptstrj['xtracks'].shape).toarray()
        y = csr_matrix(ptstrj['ytracks'], shape=ptstrj['ytracks'].shape).toarray()
        t = csr_matrix(ptstrj['Ttracks'], shape=ptstrj['Ttracks'].shape).toarray()
        if len(t)>0: 
            t[t==0]=np.nan

        ptsidx    = ptstrj['mask'][0]
        Numsample = ptstrj['xtracks'].shape[0]
        fnum      = ptstrj['xtracks'].shape[1]
        color     = np.array([np.random.randint(0,255) for _ in range(3*int(Numsample))]).reshape(Numsample,3)

        startPt = np.zeros((Numsample,1))
        endPt = np.zeros((Numsample,1))

        for tt in range(Numsample):
            if len(t)>0:
                startPt[tt] =  np.mod( np.nanmin(t[tt,:]), 600) #ignore all nans
                endPt[tt]   =  np.mod( np.nanmax(t[tt,:]), 600) 
            else:
                startPt[tt] =  np.min(np.where(x[tt,:]!=0))
                endPt[tt]   =  np.max(np.where(x[tt,:]!=0))


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
        
        # duplicate
        # fix me====
        x_re = x
        y_re = y
        t_re = []
        xspd = xspeed
        yspd = yspeed
        mask = ptsidx

        print('building adj mtx ....')
        NumGoodsample = len(x_re)
        print "NumGoodsample: ", NumGoodsample
        adj = construct_adj_thresholding(NumGoodsample, x_re, y_re)
        # adj = construct_adj_gaussian(NumGoodsample, x_re, y_re)
        # adj = construct_adj_cosine(NumGoodsample, x_re, y_re)
        # pdb.set_trace()

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
            savename = os.path.join(savePath,'Adj_'+str(matidx).zfill(3))
            savemat(savename,result)
        else:
            savename = os.path.join(savePath,'warpped_Adj_'+str(matidx).zfill(3))
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




