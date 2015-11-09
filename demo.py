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




# warpping the trjs
print("running trjcluster...")
trjcluster.trjcluster(dataPath=dataPath,savePath=savePath)

execfile('trj2dic.py') #  # generate dic format files from x_re, y_re

execfile('TrjWarping.py') # warp the trjs and save to regular matrix format

execfile('trj2dic.py') # visualize the warped trjs in warpped imgs
# noted you need to modify the prameters in trj2dic in different situation!

# after warpping, going back to the clustering:
execfile('trjcluster_func.py')

execfile('subspace_cluster.py')

# unify labels
print "unify left labels:"
unify_label.unify_label('../DoT/CanalSt@BaxterSt-96.106/leftlane/sscLabels/','../DoT/CanalSt@BaxterSt-96.106/leftlane/result/final_warpped_left')
print "unify right labels"
unify_label.unify_label('../DoT/CanalSt@BaxterSt-96.106/rightlane/sscLabels/','../DoT/CanalSt@BaxterSt-96.106/rightlane/result/final_warpped_right')


# visualize the clustered warped trjs in warpped imgs
# save the final result to dic format 
execfile('trj2dic.py') 


getVehiclePairs.py






