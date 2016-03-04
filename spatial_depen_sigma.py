####
# -------------------------------------
# |we can either warp the trajectories in advance and then construct the adj,
# |or, we can use a spatially changing sigma
# |for now, we use block-different sigmas.
# -------------------------------------

def getMuSigma(x,y,xspd,yspd):

    sxdiffAll = []
    sydiffAll = []
    mdisAll = []
	num = np.arange(fnum)
    # build adjacent mtx
    for i in range(NumGoodsample):
        # print "i", i
        # plt.cla()
        for j in range(i, min(NumGoodsample,i+1000)):
            tmp1 = x[i,:]!=0
            tmp2 = x[j,:]!=0
            idx  = num[tmp1&tmp2]
            if len(idx)>5: # has overlapping
            # if len(idx)>=30: # at least overlap for 100 frames
                sidx   = idx[0:-1] # for speed
                sxdiff = np.mean(np.abs(xspd[i,sidx]-xspd[j,sidx]))
                sydiff = np.mean(np.abs(yspd[i,sidx]-yspd[j,sidx]))                    
                mdis   = np.mean(np.abs(x[i,idx]-x[j,idx])+np.abs(y[i,idx]-y[j,idx])) #mahhattan distance
                
                sxdiffAll.append(sxdiff)
                sydiffAll.append(sydiff)
                mdisAll.append(mdis)

    mu_xspd_diff,sigma_xspd_diff = fitGaussian(sxdiffAll)
    mu_yspd_diff,sigma_yspd_diff = fitGaussian(sydiffAll)
    mu_spatial_distance,sigma_spatial_distance = fitGaussian(mdisAll)

    return mu_xspd_diff,sigma_xspd_diff,mu_yspd_diff,sigma_yspd_diff,mu_spatial_distance,sigma_spatial_distance






def getMuSigma_spatial(x,y,xspd,yspd):

    sxdiffAll = []
    sydiffAll = []
    mdisAll = []
	num = np.arange(fnum)

    # build adjacent mtx
    for i in range(NumGoodsample):
        # print "i", i
        # plt.cla()
        for j in range(i, min(NumGoodsample,i+1000)):
            tmp1 = x[i,:]!=0
            tmp2 = x[j,:]!=0
            idx  = num[tmp1&tmp2]  ###????


            # median location of the ith and jth trj
            midy1 = np.median(y[i,tmp1])
            midy2 = np.median(y[j,tmp2])

			'used to hard-threshold Gaussian adj'
            # midx1 = np.median(x[i,tmp1]) 
            # midx2 = np.median(y[j,tmp2])


            if len(idx)>5: # has overlapping
            # if len(idx)>=30: # at least overlap for 100 frames
                sidx   = idx[0:-1] # for speed
                sxdiff = np.mean(np.abs(xspd[i,sidx]-xspd[j,sidx]))
                sydiff = np.mean(np.abs(yspd[i,sidx]-yspd[j,sidx]))                    
                mdis   = np.mean(np.abs(x[i,idx]-x[j,idx])+np.abs(y[i,idx]-y[j,idx])) #mahhattan distance
                
                sxdiffAll.append(sxdiff)
                sydiffAll.append(sydiff)
                mdisAll.append(mdis)

    mu_xspd_diff,sigma_xspd_diff = fitGaussian(sxdiffAll)
    mu_yspd_diff,sigma_yspd_diff = fitGaussian(sydiffAll)
    mu_spatial_distance,sigma_spatial_distance = fitGaussian(mdisAll)

    return mu_xspd_diff,sigma_xspd_diff,mu_yspd_diff,sigma_yspd_diff,mu_spatial_distance,sigma_spatial_distance































