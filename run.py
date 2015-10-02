# execfile('klt.py')
# execfile('trjcluster.py')
# execfile('subspace_cluster.py')
# execfile('unify_labels.py')
# execfile('visualization.py')



import klt_func as klt
import trjcluster_func as trjcluster
import subspace_clustering_func as subspace_cluster
import unify_label_func as unify_label
import visualization_func as visual


isVideo = 0
klt.klt_tracker(isVideo)

trjcluster.trjcluster()
subspace_cluster.ssclustering()
unify_label.unify_label()


visual.visualization()


