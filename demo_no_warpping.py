# demo for videos no need to warp

import klt_func as klt
import trjcluster_func as trjcluster
import subspace_clustering_func as subspace_cluster
import unify_label_func as unify_label
import visualization_func as visual


isVideo   = True
linux_video_src = '/media/TOSHIBA/DoTdata/VideoFromCUSP/C0007.MP4'#complete
# dataPath  = '../tempFigs/roi2/imgs/'
# dataPath = '/media/TOSHIBA/DoTdata/VideoFromCUSP/roi2/imgs/'
savePath = '../tempFigs/roi2/'

print("running KLT...")
klt.klt_tracker(isVideo,dataPath=linux_video_src,savePath=savePath)

print("filter the trjs...")
execfile('trj_filter.py')

print("running trjcluster...")
# trjcluster.trjcluster(dataPath=dataPath,savePath=savePath)
execfile('trjcluster_func.py')

print("running subspace_cluster...")
# subspace_cluster.ssclustering(dataPath=dataPath,savePath=savePath)
execfile('subspace_cluster.py')

print("running unify_label...")
# unify_label.unify_label('../tempFigs/roi2/ssc_','../tempFigs/roi2/Result_89-115.mat')
execfile('unify_label_func.py')

# visualize 
# save the final result to dic format 
execfile('trj2dic.py') 


execfile('getVehiclesPairs.py')






