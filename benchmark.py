"""this is a scrip to implement Saunier's 06 algorithm used as the benchmark"""

from scipy.io import loadmat
from scipy.sparse import csr_matrix


from DataPathclass import *
DataPathobj = DataPath(dataSource,VideoIndex)
from parameterClass import *
Parameterobj = parameter(dataSource,VideoIndex)



def saunier06():		
	"""connect threshold"""

	D_connection = Parameterobj.nullDist_for_adj #?????   ## in the paper, they used 5 meters(world)


	"""segmentation threshold"""
	D_segmentation = Parameterobj.nullDist_for_adj   ## in the paper, they used 0.3 meters(world)




	# feature_diff_tensor

	# x

	# y



def get_co_occur_for_trjs(xi, xj,ti,tj):

	ti[ti==max(ti)] = 0
	tj[tj==max(tj)] = 0
	appeartime1 = np.min(ti)
	gonetime1   = np.max(ti)
	appeartime2 = np.min(tj)
	gonetime2   = np.max(tj)

	if appeartime1 > appeartime2: ## swap 1 and 2, s.t. appeartime1 always <=appeartime2
		temp        = appeartime2
		appeartime2 = appeartime1
		appeartime1 = temp
		temp2     = gonetime2
		gonetime2 = gonetime1
		gonetime1 = temp2


	subSampRate = 1
	if gonetime1 >= appeartime2:
		if gonetime1 <=gonetime2:
			cooccur_ran = range(appeartime2, gonetime1+subSampRate,subSampRate)
		else:
			cooccur_ran = range(appeartime2, gonetime2+subSampRate,subSampRate)
		coorccurStatus = 1	

	else:
		print "no co-occurance!"
		coorccurStatus = 0
		cooccur_ran    = []

	return coorccurStatus, cooccur_ran



# for i in range(NumGoodsampleSameDir):
#     print "i", i
#     for j in range(i+1, NumGoodsampleSameDir):
#     	xi = x[i,:]
#     	xj = x[j,:]
#     	ti = t[i,:]
#     	tj = t[j,:]


# 		[coorccurStatus, cooccur_ran] = get_co_occur_for_trjs(xi,xj,ti,tj)
# 		if coorccurStatus:
# 			print len(cooccur_ran)
# 			for tt in cooccur_ran:


# 				distance = np.mean(np.sqrt((co1X-co2X)**2+(co1Y-co2Y)**2))
			


def connect_segment(xi,xj,yi,yj,tt,dij_t):
	D_connection = 60
	D_segmentation = 20

	if xi[tt]!=0 and xj[tt]!=0:
		dij = np.sqrt((xi[tt]-xj[tt])**2+(yi[tt]-yj[tt])**2)
		if dij <D_connection:
			connection = 1

		dij_t[tt] = dij  ## trji and trji at time tt
		extremeDis = max(dij_t)-min(dij_t)
		if extremeDis>D_segmentation:
			connection = 0

	else:
		connection = 0
		dij = 0


	return connection,dij



if __name__ == '__main__':

	matfilepath = DataPathobj.smoothpath
	matfiles = sorted(glob.glob(matfilepath + 'klt*.mat'))
	for matidx,matfile in enumerate(matfiles):
		# for matidx in range(5,len(matfiles)):
		# for matidx in range(3,4,1):
		# matfile = matfiles[matidx]
		result = {} #for the save in the end
		print "Processing truncation...", str(matidx+1)
		ptstrj = loadmat(matfile)
		"""if no trj in this file, just continue"""
		try:
			print 'total number of trjs in this trunk', len(ptstrj['trjID'])
		except:
			continue
		if len(ptstrj['trjID'])==0:
			continue
		trjID = ptstrj['trjID'][0]

		if not Parameterobj.useWarpped:            
			x    = csr_matrix(ptstrj['xtracks'], shape=ptstrj['xtracks'].shape).toarray()
			y    = csr_matrix(ptstrj['ytracks'], shape=ptstrj['ytracks'].shape).toarray()
			t    = csr_matrix(ptstrj['Ttracks'], shape=ptstrj['Ttracks'].shape).toarray()
			xspd = csr_matrix(ptstrj['xspd'], shape=ptstrj['xspd'].shape).toarray()
			yspd = csr_matrix(ptstrj['yspd'], shape=ptstrj['yspd'].shape).toarray()
			Xdir = csr_matrix(ptstrj['Xdir'], shape=ptstrj['Xdir'].shape).toarray()
			Ydir = csr_matrix(ptstrj['Ydir'], shape=ptstrj['Ydir'].shape).toarray()
		else:
			x    = csr_matrix(ptstrj['xtracks_warpped'],shape=ptstrj['xtracks'].shape).toarray()
			y    = csr_matrix(ptstrj['ytracks_warpped'],shape=ptstrj['ytracks'].shape).toarray()
			t    = csr_matrix(ptstrj['Ttracks'],shape=ptstrj['Ttracks'].shape).toarray()
			xspd = csr_matrix(ptstrj['xspd_warpped'], shape=ptstrj['xspd'].shape).toarray()
			yspd = csr_matrix(ptstrj['yspd_warpped'], shape=ptstrj['yspd'].shape).toarray()
			Xdir = csr_matrix(ptstrj['Xdir_warpped'], shape=ptstrj['Xdir_warpped'].shape).toarray()
			Ydir = csr_matrix(ptstrj['Ydir_warpped'], shape=ptstrj['Ydir_warpped'].shape).toarray()

		if Parameterobj.useSBS:
			FgBlobIndex = csr_matrix(ptstrj['fg_blob_index'], shape=ptstrj['fg_blob_index'].shape).toarray()
			fg_blob_center_X = csr_matrix(ptstrj['fg_blob_center_X'], shape=ptstrj['fg_blob_center_X'].shape).toarray()
			fg_blob_center_Y = csr_matrix(ptstrj['fg_blob_center_Y'], shape=ptstrj['fg_blob_center_Y'].shape).toarray()

			FgBlobIndex[FgBlobIndex==0]=np.nan
			fg_blob_center_X[FgBlobIndex==0]=np.nan
			fg_blob_center_Y[FgBlobIndex==0]=np.nan
		else:
			fg_blob_center_X = np.ones(x.shape)*np.nan
			fg_blob_center_Y = np.ones(x.shape)*np.nan
			FgBlobIndex      = np.ones(x.shape)*np.nan

		Numsample = ptstrj['xtracks'].shape[0]
		NumGoodsampleSameDir = Numsample ## ignore directions for now
		# fnum = ptstrj['xtracks'].shape[1]
		dijt = np.zeros((NumGoodsampleSameDir,NumGoodsampleSameDir,Parameterobj.trunclen)) # d_ij(t) across time
		connectionMap = np.zeros((NumGoodsampleSameDir,NumGoodsampleSameDir,Parameterobj.trunclen))
		for tt in range(Parameterobj.trunclen):  ## construct a Nsample by Nsample "connection map" at every frame
		
			for i in range(NumGoodsampleSameDir):
				print "i", i
				for j in range(i+1, NumGoodsampleSameDir):
					xi = x[i,:]
					xj = x[j,:]
					yi = y[i,:]
					yj = y[j,:]
					connection, dij= connect_segment(xi,xj,yi,yj,tt,dijt[i,j,:])
					connectionMap[i,j,tt] = connection
					dijt[i,j,tt]  = dij

			connectionMap[:,:,tt]







































