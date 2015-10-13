import klt_func as klt
import trjcluster_func as trjcluster
import subspace_clustering_func as subspace_cluster
import unify_label_func as unify_label
import visualization_func as visual


isVideo = False
# myvideo = 'myvideo.mp4'

print("running KLT...")
# klt.klt_tracker(isVideo,dataPath=myvideo,savePath='')
klt.klt_tracker(isVideo)

print("running trjcluster...")
trjcluster.trjcluster('','')

print("running subspace_cluster...")
subspace_cluster.ssclustering('','')

print("running unify_label...")
unify_label.unify_label('','result.mat')

print("running visualization...")
visual.visualization(isVideo,myvideo,'result.mat','')


