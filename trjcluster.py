from scipy.io import loadmat,savemat
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import numpy as np
import pdb,glob


# inifilename = 'HR'
# matfiles = sorted(glob.glob('./mat/20150222_Mat/'+inifilename+'*.mat'))
#matfiles = sorted(glob.glob('./mat/20150222_Mat/'+inifilename+'_w_T'+'*.mat'))  #read the files with the time tracks
# matfiles = sorted(glob.glob('./mat/'+inifilename+'*.mat'))

matfiles = sorted(glob.glob('../DoT/5Ave@42St-96.81/mat/5Ave@42St-96.81_2015-06-16_16h04min40s686ms/'+'*.mat'))

# -- utilities
vthr = 15 #threshold of min speed
fps  = 4 # GGD: careful about this!  This is video dependent...
tthr = 60*fps   #transition time (red light time)


for matidx,matfile in enumerate(matfiles):

    ptstrj = loadmat(matfile)
    xx = csr_matrix(ptstrj['xtracks'], shape=ptstrj['xtracks'].shape).toarray()
    yy = csr_matrix(ptstrj['ytracks'], shape=ptstrj['ytracks'].shape).toarray()
    tt = csr_matrix(ptstrj['Ttracks'], shape=ptstrj['Ttracks'].shape).toarray()

    ptsidx = ptstrj['idxtable'][0]

    nklt, nfrm = xx.shape

    # select good frames
    good_fr = xx!=0

    # calculate velocities
    vx  = np.diff(xx)*good_fr[:,1:]*good_fr[:,:-1]
    vy  = np.diff(yy)*good_fr[:,1:]*good_fr[:,:-1]
    vel = np.sqrt(vx*vx+vy*vy) # vel = np.abs(vx)+np.abs(vy)

    # first eliminate KLT points that are 
    # 1. tracked for fewer than 1 second
    # 2. have a max velocity below threshold (for pedestrians)
    # 3. GGD: I removed this criterion: sum(vel<3)<tthr
    print("TRJCLUSTER: Careful!!!  Must handle 0 good points case!!!")

    good_pts = (good_fr.sum(1)>fps)&(vel.max(1)>vthr)
    xx       = xx[good_pts]
    yy       = yy[good_pts]
    tt       = tt[good_pts]
    good_fr  = good_fr[good_pts]
    mask     = ptsidx[good_pts]


    ## GGD: we paused here...

    sample = len(x_re)
    adj = np.zeros([sample,sample])
    dth = 30*1.5
    spdth = 5
    num = arange(nfrm)
    x_re = array(x_re)
    y_re = array(y_re)
    t_re = array(t_re)
    
    xspd = array(xspd)
    yspd = array(yspd)

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
                        mdis = mean(np.abs(x_re[i,idx]-x_re[j,idx])+np.abs(y_re[i,idx]-y_re[j,idx]))
                        #mahhattan distance
                        if mdis < dth:
                            adj[i,j] = 1
            else:
                adj[i,j] = adj[j,i]

    sparsemtx = csr_matrix(adj)
    s,c = connected_components(sparsemtx)
    result = {}
    result['adj'] = adj
    result['c']   = c
    result['mask']= mask
    result['Ttracks'] = t_re

    # savename = './mat/20150222_Mat/adj/'+inifilename+'_adj_'+str(matidx+1).zfill(3)
    # savename = './mat/20150222_Mat/adj/'+inifilename+'_adj_withT_'+str(matidx+1).zfill(3)
    
    savename = '../DoT/5Ave@42St-96.81/adj/5Ave@42St-96.81_2015-06-16_16h04min40s686ms/' + str(matidx+1).zfill(3)

    savemat(savename,result)

