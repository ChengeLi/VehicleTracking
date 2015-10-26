import klt_func as klt
import trjcluster_func as trjcluster
import subspace_clustering_func as subspace_cluster
import unify_label_func as unify_label
import visualization_func as visual


isVideo = False
# myvideo = '/media/TOSHIBA EXT/Video from CUSP/C0007.MP4'
myImage = './tempFigs/roi2/'

print("running KLT...")
# klt.klt_tracker(isVideo,dataPath=myvideo,savePath='')
klt.klt_tracker(isVideo,dataPath=myImage,savePath='./tempFigs/roi2/')

print("running trjcluster...")
trjcluster.trjcluster('./tempFigs/roi2/',savePath='./tempFigs/roi2/')

print("running subspace_cluster...")
subspace_cluster.ssclustering('./tempFigs/roi2/','./tempFigs/roi2/')

print("running unify_label...")
unify_label.unify_label('./tempFigs/roi2/sscConstructedAdj_CC','./tempFigs/roi2/sscConstructedAdj_CCResult.mat')

print("running visualization...")
visual.visualization(isVideo,myvideo,'result.mat','')


