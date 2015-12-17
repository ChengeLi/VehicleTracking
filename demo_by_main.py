# demo for videos no need to warp



"""whole process for Jay&Johnson"""

print("running KLT...")
execfile('klt_func.py')
print("filter the trjs...")
execfile('trj_filter.py')

print("running trjcluster...")
execfile('trjcluster_func_SBS.py')

print("running subspace_cluster...")
execfile('subspace_cluster.py')

print("running unify_label...")
execfile('unify_label_func.py')
# visualize 
print "save the final result to dic format" 
execfile('trj2dic.py')
print "get trj pairs" 
execfile('getVehiclesPairs.py')



