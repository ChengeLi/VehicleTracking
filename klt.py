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

feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=7, 
                      blockSize=7)

idx    = 0
tdic   = [0]
start  = []
end    = []
oldlen = 0
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
imlist = sorted(glob('../VideoData/20150222/*.jpg'))
nframe = len(imlist)

# -- read in first frame and set dimensions
frame  = cv2.imread(imlist[0])
frameL  = np.zeros_like(frame[:,:,0])
frameLp = np.zeros_like(frameL)

# -- set low number of frames for testing
nframe = 3000

while (frame_idx < nframe):
    frame[:,:,:] = cv2.imread(imlist[frame_idx])
    frameL[:,:]  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #*self.mask
    
    if len(tracksdic) > 0:
        pnts_old = np.float32([tracksdic[i][-1][:2] for i in tracksdic]) \
            .reshape(-1, 1, 2)
        # pdb.set_trace()
        pnts_new, st, err  = cv2.calcOpticalFlowPyrLK(frameLp, frameL, 
                                                      pnts_old, None, 
                                                      **lk_params)
        pnts_oldr, st, err = cv2.calcOpticalFlowPyrLK(frameL, frameLp, 
                                                      pnts_new, None, 
                                                      **lk_params)
        dist = abs(pnts_old-pnts_oldr).reshape(-1, 2).max(-1)
        good = dist < 1
 
# GGD: !!! The block below never seems to be exectuted... why is this here???
        # if (len(pregood)>0):
        #     # pdb.set_trace()
        #     good[:len(pregood)] = good[:len(pregood)]&good
        #     #good = (good & inroi)
        #     pregood = good
                    
        for (x, y), good_flag, idx in zip(pnts_new.reshape(-1, 2), good, 
                                          tracksdic.keys()):
            #if idx == 144:
            #    pdb.set_trace()
            if not good_flag:

                end[idx] = (frame_idx-1)
                tracksdic[idx].append((-100,-100,frame_idx))
                #self.tracks[idx].append((-100., -100.,self.frame_idx))
                #self.Ttracks[idx].append(self.frame_idx)
                continue
            if x != -100:
                tracksdic[idx].append((x,y,frame_idx))

            #self.tracks[idx].append((x, y,self.frame_idx))
            #self.Ttracks[idx].append(self.frame_idx)

#    vis = frame.copy()
#            cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)
        #cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
        #draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

    if frame_idx % detect_interval == 0:

        mask = 255*np.ones_like(frameL)

        for x, y in [np.int32(tracksdic[tr][-1][:2]) for tr in tracksdic]:
            cv2.circle(mask, (x, y), 5, 0, -1)    

        corners = cv2.goodFeaturesToTrack(frameL,mask=mask,**feature_params)
         
        if corners is not None:
            for x, y in np.float32(corners).reshape(-1, 2):
                #self.tracks.append([(x, y,self.frame_idx)])
                        
                try:
                    tracksdic[dicidx].append((x,y,frame_idx))
                except:
                    tracksdic[dicidx] = []
                    tracksdic[dicidx].append((x,y,frame_idx))
                dicidx += 1 

            start = start + [frame_idx]*(len(tracksdic)+lenoffset-oldlen) ##add the start time for new added feature pts
            end   = end + [-1]*(len(tracksdic)+lenoffset-oldlen)
            oldlen = len(tracksdic)+lenoffset

            #pdb.set_trace()
    print('{0} - {1}'.format(frame_idx,len(tracksdic)))



    frame_idx += 1
    frameLp[:,:] = frameL[:,:]
    #  visualize:============================== 
    # cv2.imshow('lk_track', vis)
            
    #name = '/home/andyc/image/AIG/lking/'+str(self.frame_idx).zfill(5)+'.jpg'
    #cv2.imwrite(name,vis)
            
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break
        
    #========== save data ===================#    
    if  (frame_idx % trunclen) == 0:
        # pdb.set_trace()
        #fbr = pickle.load(open('./mask/forbiddenregion_mask.pkl','rb')) #forbbiden region

        Xtracks = np.zeros([len(tracksdic),trunclen])
        Ytracks = np.zeros([len(tracksdic),trunclen])
        Ttracks = np.zeros([len(tracksdic),trunclen])

        offset = ((frame_idx // trunclen)-1)*trunclen # which trunclen, starting from 0th
        for trjidx,i in enumerate(tracksdic.keys()):

            endfidx = end[i]  # end frame number for pts i

            startfidx = start[i]
            if (endfidx >= offset) or (endfidx == -1): #offset is the current trunck

                if endfidx == -1 : # trj still alive                                                                                
                    k =array(tracksdic[i]).T    # shape is: 3X600 3:(x,y,t)
                else:              # trj is dead in this chunk                                                                                
                    k =array(tracksdic[i])[:-1].T #(save all but the last -1)

                if startfidx == -1 : #exist in previous trucation
                    startfidx = offset
                    k = k[:,1:]

                if endfidx != -1 : #trj is dead in this chunk, save
                    '''
                    xx,yy = k[0:2,-1]
                    if xx >= ncols:
                        xx = ncols-1
                    if xx < 0:
                        xx = 0
                    if yy >= nrows:
                        yy = nrows-1
                    if yy < 0:
                        yy = 0
                    #if fbr[yy,xx] == 1: 
                    #    Xtracks[trjidx,:][startfidx-offset:endfidx-offset+1] = k[0] 
                    #    Ytracks[trjidx,:][startfidx-offset:endfidx-offset+1] = k[1]
                    '''
                    Xtracks[trjidx,:][startfidx-offset:endfidx-offset+1] = k[0]
                    Ytracks[trjidx,:][startfidx-offset:endfidx-offset+1] = k[1]
                    Ttracks[trjidx,:][startfidx-offset:endfidx-offset+1] = k[2]
                else:
                    Xtracks[trjidx,:][startfidx-offset:] = k[0]
                    Ytracks[trjidx,:][startfidx-offset:] = k[1]
                    Ttracks[trjidx,:][startfidx-offset:] = k[2]


        #pdb.set_trace()
        #==== save files ====

        trk ={}
        
        trk['xtracks'] = csr_matrix(Xtracks)
        trk['ytracks'] = csr_matrix(Ytracks)
        trk['idxtable'] = tracksdic.keys()
        trk['Ttracks'] = csr_matrix(Ttracks)

        savename = './mat/20150222_Mat/HR_w_T'+str(frame_idx/trunclen).zfill(3)
        # savename = './mat/20150220_Mat/HR'+str(frame_idx/trunclen).zfill(3)
        savemat(savename,trk)

        #===== release memory =====
        deadtrj = np.where(array(end)>0 )[0]  # dead trjs' dying positions
        lenoffset += len(deadtrj)
        for i in range(oldlen):
            if i in deadtrj:
                tracksdic.pop(i)
                end[i] = -2
            else:
                try: #if trj exist and cross two truncations
                    tracksdic[i] = [tracksdic[i][-1]]
                    start[i] = -1  # it's from the last trunction
                except: # if trj already been removed
                    pass

#if (frame_idx % trunclen) !=0:



