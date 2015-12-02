# demo for videos no need to warp

import klt_func as klt
import trj_filter as trj_filter
import trjcluster_func_SBS as trjcluster
import subspace_cluster as subspace_cluster
import unify_label_func as unify_label
import trj2dic as trj2dic


# import visualization_func as visual

"""whole process for Jay&Johnson"""
# linux_video_src = '/media/TOSHIBA/DoTdata/VideoFromCUSP/C0007.MP4'#complete
# dataPath = '/media/TOSHIBA/DoTdata/VideoFromCUSP/roi2/imgs/'
# savePath = '../tempFigs/roi2/'

dataPath = '/media/My Book/CUSP/AIG/Jay&Johnson/roi2/imgs/'
savePath = '/media/My Book/CUSP/AIG/Jay&Johnson/roi2/klt/'
print("running KLT...")
klt.klt_tracker(isVideo= False,dataPath=dataPath,savePath=savePath,useSameBlockScore = False,isVisualize = False,dataSource = 'Johnson')

print("filter the trjs...")
# execfile('trj_filter.py')
trj_filter.filtering_main_function(fps = 30,dataSource = 'Johnson')

print("running trjcluster...")
trjcluster.trjcluster(useSBS=False,dataSource = 'Johnson')

print("running subspace_cluster...")
subspace_cluster.ssc_main(dataSource = 'Johnson')

print("running unify_label...")
unify_label.unify_label_main(dataSource = 'Johnson')

# visualize 
# save the final result to dic format 
trj2dic.trj2dic_main(isVideo  = False, dataSource = 'Johnson')

execfile('getVehiclesPairs.py')



# ==========================================================================
"""whole process for Canal"""

isVideo  = True
if isVideo:
    dataPath = '../DoT/Convert3/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms.avi'
else:
    dataPath = '../DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'
savePath = '/media/My Book/CUSP/AIG/DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/klt/'

print("running KLT...")
klt.klt_tracker(isVideo = True,dataPath=dataPath,savePath=savePath)

print("filter the trjs...")
# execfile('trj_filter.py')
trj_filter.filtering_main_function(fps = 23)

print("running trjcluster...")
trjcluster.trjcluster(useSBS=True,dataSource = 'DoT')

print("running subspace_cluster...")
# execfile('subspace_cluster.py')
subspace_cluster.ssc_main(dataSource = 'DoT')

print("running unify_label...")
unify_label.unify_label_main(dataSource = 'DoT')

# visualize 
# save the final result to dic format 
trj2dic.trj2dic_main(isVideo = True, dataSource = 'DoT')


execfile('getVehiclesPairs.py')






