# This is a program for inspecting data files
import pickle
import numpy as np


xdiff = np.array(pickle.load( open( "./roi2_xspdiff0.p", "rb" ) ))
ydiff = np.array(pickle.load( open( "./roi2_yspdiff0.p", "rb" ) ))
mdis  = np.array(pickle.load( open( "./roi2_mdis0.p", "rb" ) ))



mean(xdiff)
std(xdiff)








