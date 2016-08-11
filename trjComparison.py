import numpy as np
"""comparison functions for trj pairs"""
def get_spd_dis_diff(xspd_i,xspd_j,yspd_i,yspd_j,xi,xj,yi,yj):
    """use mean of the spd diff"""
    # sxdiff = np.mean(np.abs(xspd[i,sidx]-xspd[j,sidx]))
    # sydiff = np.mean(np.abs(yspd[i,sidx]-yspd[j,sidx]))                    
    """use MAX of the spd diff!"""
    sxdiff = np.max(np.abs(xspd_i-xspd_j)[:])
    sydiff = np.max(np.abs(yspd_i-yspd_j)[:])                    

    """use MAX of the dist diff!"""
    # mdis   = np.mean(np.abs(x[i,idx]-x[j,idx])+np.abs(y[i,idx]-y[j,idx])) #mahhattan distance
    # mdis = np.mean(np.sqrt((x[i,idx]-x[j,idx])**2+(y[i,idx]-y[j,idx])**2))  #euclidean distance
    mdis = np.max(np.sqrt((xi-xj)**2+(yi-yj)**2))  #euclidean distance
    return sxdiff,sydiff,mdis

def get_hue_diff(hue_i,hue_j):
    huedis = np.abs(np.nanmean(hue_i)-np.nanmean(hue_j))
    return huedis

def sameBlobScore():
    SBS = sameBlobScore(np.array(FgBlobIndex[i,idx]),np.array(FgBlobIndex[j,idx]))
    return SBS

    """plot foreground blob center"""
    # plt.plot(fg_blob_center_X[i,:][fg_blob_center_X[i,:]!=0],fg_blob_center_Y[i,:][fg_blob_center_X[i,:]!=0],'b')
    # plt.plot(fg_blob_center_X[i,idx],fg_blob_center_Y[i,idx],'g')
    # plt.plot(cxi,cyi,'r')
    # plt.draw()




