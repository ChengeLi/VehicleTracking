import klt_func as klt
import trjcluster_func as trjcluster
import subspace_clustering_func as subspace_cluster
import unify_label_func as unify_label
import visualization_func as visual


isVideo = False

# dataPath = './tempFigs/roi2/'
# savePath = './tempFigs/roi2/'
 
dataPath = '../DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'
savePath = '../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'



print("running KLT...")
# klt.klt_tracker(isVideo,dataPath=myvideo,savePath='')
klt.klt_tracker(isVideo,dataPath=dataPath,savePath=savePath)

print("running trjcluster...")
trjcluster.trjcluster(dataPath=dataPath,savePath=savePath)

print("running subspace_cluster...")
subspace_cluster.ssclustering(dataPath=dataPath,savePath=savePath)


print("running unify_label...")
unify_label.unify_label('./tempFigs/roi2/sscConstructedAdj_CC','./tempFigs/roi2/sscConstructedAdj_CCResult.mat')

print("running visualization...")
visual.visualization(isVideo,myvideo,'result.mat','')


