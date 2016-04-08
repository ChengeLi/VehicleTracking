'''
compare our result with ground truth
'''
import os
import pickle as pickle
import numpy as np
import pdb
import matplotlib.pyplot as plt
from DataPathclass import *
DataPathobj = DataPath(dataSource,VideoIndex)
from parameterClass import *
Parameterobj = parameter(dataSource,VideoIndex)




if __name__ == '__main__':

	"""load GT trj dictionary:"""
	GTtrjdic = pickle.load(open(DataPathobj.DataPath+'/GTtrjdictionary_'+dataSource,'rb'))
	if dataSource == 'NGSIM':
		GTtrjdic = pickle.load(open(DataPathobj.DataPath+'/GTtrjdictionary_'+dataSource+'_cam4.p','rb'))

	"""load our system result trj dictionary:"""
	VehicleObjDic = pickle.load(open(os.path.join(DataPathobj.pairpath,'VehicleObjDic.p'),'rb'))



	"""find the same time, assign the nearest one to the GT"""
	"""nearest neighbour"""



	for ii in GTtrjdic.keys():
		pass

















