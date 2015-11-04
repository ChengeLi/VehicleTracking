#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
Usage
-----
lk_track.py [<video_source>]
Keys
----
ESC - exit
'''

import os
import cv2
import pdb
import numpy as np
import pickle as pkl
from glob import glob
from time import clock
from scipy.io import savemat
from scipy.sparse import csr_matrix


# -- utilities
lk_params = dict(winSize=(15, 15), maxLevel=2, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                            10, 0.03))

# feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=7, 
#                       blockSize=7)
feature_params = dict(maxCorners=1000, qualityLevel=0.2, minDistance=3, 
                      blockSize=5)
idx    = 0
tdic   = [0]
start  = []
end    = []
dicidx = 0
track_len = 10
detect_interval = 5
tracksdic = {} 
frame_idx = 0
pregood = []
trunclen = 600
lenoffset = 0


# video_src = '/home/andyc/Videos/jayst.mp4'
# video_src = '/home/andyc/Videos/video0222.mp4'
# video_src = '../VideoData/video0222.mp4'
# cam = cv2.VideoCapture(video_src)
# nrows = cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
# ncols = cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
# nframe = int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

# -- get the full image list
#imlist = sorted(glob('../VideoData/20150220/*.jpg'))
# imlist = sorted(glob('images/*.jpg'))

# imagepath = '../DoT/5Ave@42St-96.81/5Ave@42St-96.81_2015-06-16_16h04min40s686ms/'
imagepath = '../DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'
imlist = sorted(glob(imagepath + '*.jpg'))

nframe = len(imlist)

# -- read in first frame and set dimensions
frame  = cv2.imread(imlist[0])
frameL  = np.zeros_like(frame[:,:,0])
frameLp = np.zeros_like(frameL)

# -- set mask
mask = 255*np.ones_like(frameL)

# -- set low number of frames for testing
nframe = 1801


while (frame_idx < nframe):
    frame[:,:,:] = cv2.imread(imlist[frame_idx])
    frameL[:,:]  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #*self.mask

    vis = frame.copy()
    
    if len(tracksdic) > 0:
        pnts_old = np.float32([tracksdic[i][-1][:2] for i in tracksdic]) \
            .reshape(-1, 1, 2)

        pnts_new, st, err  = cv2.calcOpticalFlowPyrLK(frameLp, frameL, 
                                                      pnts_old, None, 
                                                      **lk_params)
        pnts_oldr, st, err = cv2.calcOpticalFlowPyrLK(frameL, frameLp, 
                                                      pnts_new, None, 
                                                      **lk_params)
        dist = abs(pnts_old-pnts_oldr).reshape(-1, 2).max(-1)
        good = dist < 1
 
        for (x, y), good_flag, idx in zip(pnts_new.reshape(-1, 2), good, 
                                          tracksdic.keys()):
            if not good_flag:
                end[idx] = (frame_idx-1)
                tracksdic[idx].append((-100,-100,frame_idx))
                continue
            if x != -100:
                tracksdic[idx].append((x,y,frame_idx))

            cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)


    if frame_idx % detect_interval == 0:

        # GGD: this is (I *think*) eliminating redundant non-moving points
        mask[:,:] = 255
        for x, y in [np.int32(tracksdic[tr][-1][:2]) for tr in tracksdic]:
            cv2.circle(mask, (x, y), 5, 0, -1)    

        corners = cv2.goodFeaturesToTrack(frameL,mask=mask,**feature_params)

        if corners is not None:
            for x, y in np.float32(corners).reshape(-1, 2):
                try:
                    tracksdic[dicidx].append((x,y,frame_idx))
                except:
                    tracksdic[dicidx] = []
                    tracksdic[dicidx].append((x,y,frame_idx))
                dicidx += 1 

            start  = start + [frame_idx]*corners.shape[0]
            end    = end + [-1]*corners.shape[0]

    print('{0} - {1}'.format(frame_idx,len(tracksdic)))

    # switch previous frame
    frame_idx += 1
    frameLp[:,:] = frameL[:,:]
    # ## CG: for visulization
    cv2.imshow('klt', vis)
    # Wait 0 milliseconds
    cv2.waitKey(5)


    # dump trajectories to file
    if  (frame_idx % trunclen) == 0:

        # initialize track arrays
        Xtracks = np.zeros([len(tracksdic),trunclen])
        Ytracks = np.zeros([len(tracksdic),trunclen])
        Ttracks = np.zeros([len(tracksdic),trunclen])

        # set first frame in this chunk
        offset  = frame_idx - trunclen

        # loop through the current trajectories list
        for trjidx, ii in enumerate(tracksdic.keys()):

            # set the starting and ending frame index
            st_ind = start[ii]
            en_ind = end[ii]

            # if en_ind is -1, then the traj is still alive,
            # otherwise, the trajectory is dead (but still in the
            # tracks dictionary, otherwise it would have been
            # removed).
            if en_ind==-1:
                ttrack = np.array(tracksdic[ii]).T
            else:
                ttrack = np.array(tracksdic[ii])[:-1].T # don't save -100s

            # if st_ind is -1, then the track existed in the previous
            # truncation and all points except the last one of the
            # previous truncation were removed, so only save from the
            # second point.
            if st_ind==-1:
                st_ind = offset
                ttrack = ttrack[:,1:]

            # put trajectory into matrix
            tstart, tstop = st_ind-offset, en_ind-offset+1

            if en_ind!=-1:
                Xtracks[trjidx,:][tstart:tstop] = ttrack[0]
                Ytracks[trjidx,:][tstart:tstop] = ttrack[1]
                Ttracks[trjidx,:][tstart:tstop] = ttrack[2]
            else:
                Xtracks[trjidx,:][tstart:] = ttrack[0]
                Ytracks[trjidx,:][tstart:] = ttrack[1]
                Ttracks[trjidx,:][tstart:] = ttrack[2]

        # put tracks into sparse matrix
        trk ={}
        
        trk['xtracks'] = csr_matrix(Xtracks)
        trk['ytracks'] = csr_matrix(Ytracks)
        trk['Ttracks'] = csr_matrix(Ttracks)
        trk['mask']    = tracksdic.keys()

        # save as matlab file... :-/
        # savename = './mat/20150222_Mat/HR_w_T______test_' + \
            # str(frame_idx/trunclen).zfill(3)

        # savename = '../DoT/5Ave@42St-96.81/mat/5Ave@42St-96.81_2015-06-16_16h04min40s686ms/' + str(frame_idx/trunclen).zfill(3)
        savename = '../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/' + str(frame_idx/trunclen).zfill(3)
        savemat(savename,trk)


        # for dead tracks, remove them.  for alive tracks, remove all
        # points except the last one (in order to track into the next
        # frame), and set the start frame to -1.
        deadtrj = np.where(np.array(end)>=0)[0] # ==0 is for the case when the tracks ends at 0 frame
        lenoffset += len(deadtrj)
        for i in tracksdic.keys():   
            if i in deadtrj:
                tracksdic.pop(i)
                end[i] = -2
            else:
                tracksdic[i] = [tracksdic[i][-1]]
                start[i] = -1  # it's from the last trunction
