import os
import cv2
import pdb
import math
import pickle
import numpy as np
import glob as glob
from time import clock
from scipy.io import loadmat,savemat
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt


# def klt_tracker(isVideo, \
#  dataPath = '../DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/',\
#  savePath = '../DoT/CanalSt@BaxterSt-96.106/klt/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/',\
#  useBlobCenter = True,isVisualize = False,dataSource = 'DoT'):

if __name__ == '__main__':
#     isVideo  = True
#     if isVideo:
#         dataPath = '../DoT/Convert3/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms.avi'
#     else:
#         dataPath = '../DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'
#     savePath = '/media/My Book/CUSP/AIG/DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/klt/'
#     dataSource = 'DoT'
    # useBlobCenter = True
    # isVisualize = False
    # frame_idx_bias = 84600 #start from the 47th minute
    frame_idx_bias = 0 #johnson new
    isVideo= False
    useBlobCenter = False
    isVisualize = False
    dataSource = 'Johnson'
    dataPath = '/media/My Book/CUSP/AIG/Jay&Johnson/roi2/imgs/'
    savePath = '/media/My Book/CUSP/AIG/Jay&Johnson/roi2/subSamp/klt/'
    subSampRate = 6 # since 30 fps may be too large, subsample the images back to 5 FPS

    # dataPath = '/media/My Book/CUSP/AIG/Jay&Johnson/JohnsonNew/1st.mov'
    # savePath = '/media/My Book/CUSP/AIG/Jay&Johnson/JohnsonNew/subSamp/klt/'
    # subSampRate = 6 # since 30 fps may be too large, subsample the images back to 5 FPS



    # -- utilities
    plt.figure(num=None, figsize=(8, 11))
    """ new jay st """
    if dataSource == 'Johnson':
        lk_params = dict(winSize=(5, 5), maxLevel=2, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),flags = cv2.OPTFLOW_LK_GET_MIN_EIGENVALS) #maxLevel: level of pyramid
        feature_params = dict(maxCorners=1000, qualityLevel=0.1, minDistance=3, 
                              blockSize=3)  #qualityLevel, below which dots will be rejected

    """ canal st """
    if dataSource == 'DoT':
        lk_params = dict(winSize=(15, 15), maxLevel=2, 
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10, 0.03)) 
        # feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=7, 
        #                       blockSize=7)  
        feature_params = dict(maxCorners=1000, qualityLevel=0.2, minDistance=3, 
                              blockSize=5)  # old jayst 
    
    # idx             = 0

    # track_len       = 10
    # pregood         = []
    trunclen        = 600
    # lenoffset       = 0
    detect_interval = 5


    previousLastFiles = sorted(glob.glob(savePath+'*klt_*'))
    if len(previousLastFiles)>0:
        if len(previousLastFiles) >1:
            previousLastFile = previousLastFiles[-1]
        else: previousLastFile = previousLastFiles[0]
        
        lastTrj   = loadmat(previousLastFile)
        lastID    = np.max(lastTrj['trjID'][0])

        dicidx      = lastID+1 #counting from the last biggest global ID
        lastT       = lastTrj['Ttracks']
        lastT[lastT==np.max(lastT)] = np.nan
        tracksdic   = {}
        start       = {}
        end         = {}
        for kk in range((lastTrj['lastPtsKey'][0]).shape[0]):
            key            = lastTrj['lastPtsKey'][0][kk]
            start[key]     = np.nanmin(lastT[kk,:])
            if math.isnan(start[key]): 
                print "key:",key, "kk:",kk
                # pdb.set_trace()
                start.pop(key)
                continue
            end[key]       = -1 #all alive trj
            tracksdic[key] = []
            tracksdic[key].append(tuple(lastTrj['lastPtsValue'][kk,:]))
    else:
        dicidx    = 0 # start from 0
        tracksdic = {} 
        start     = {}
        end       = {}

    if frame_idx_bias==[]:
        frame_idx_bias = len(previousLastFiles)*600
    frame_idx      = (0 + frame_idx_bias)/subSampRate 
    start_position = frame_idx_bias

    if useBlobCenter:
        # linux local:
        # blobColorImgList = sorted(glob.glob('../DoT/CanalSt@BaxterSt-96.106/incPCP/Canal_blobImage/*.jpg'))
        # linux:
        # blobColorImgList = sorted(glob.glob('/media/TOSHIBA/DoTdata/CanalSt@BaxterSt-96.106/incPCP/Canal_blobImage/*.jpg'))
        # Mac
        # blobColorImgList = sorted(glob.glob('/Volumes/TOSHIBA/DoTdata/CanalSt@BaxterSt-96.106/incPCP/Canal_blobImage/*.jpg'))
        # blobPath             = '/media/TOSHIBA/DoTdata/CanalSt@BaxterSt-96.106/incPCP/'
        blobindPath                 = '/media/My Book/CUSP/AIG/DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/BlobLabels/'
        blob_ind_sparse_matrices    = sorted(glob.glob(blobindPath + 'blob*.p'))
        blobCenterPath              = '/media/My Book/CUSP/AIG/DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/BlobCenters/'
        blob_center_sparse_lists = sorted(glob.glob(blobCenterPath + 'blob*.p'))


    
    if isVideo:
        video_src = dataPath
        cap       = cv2.VideoCapture(video_src)
        nrows     = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        ncols     = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        nframe    = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        
        print 'reading buffer...'
        cap.set ( cv2.cv.CV_CAP_PROP_POS_FRAMES , max(0,start_position-1))
        status, frame = cap.read()
        frameL        = np.zeros_like(frame[:,:,0]) #just initilize, will be set in the while loop
        if len(previousLastFiles)>0:
            frameLp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #set the previous to be the last frame in last truncation
        else:    
            frameLp = np.zeros_like(frameL)

    if not isVideo:  # -- get the full image list
        imagepath = dataPath
        imlist    = sorted(glob.glob(imagepath + '*.jpg'))
        nframe    = len(imlist)
        # -- read in first frame and set dimensions
        frame     = cv2.imread(imlist[max(0,start_position-1)])
        frameL    = np.zeros_like(frame[:,:,0]) #just initilize, will be set in the while loop
        if len(previousLastFiles)>0:
            frameLp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #set the previous to be the last frame in last truncation
        else:    
            frameLp = np.zeros_like(frameL)




    # -- set mask, all ones = no mask
    mask = 255*np.ones_like(frameL)

    # -- set low number of frames for testing
    # nframe = 1801
    if isVideo: cap.set ( cv2.cv.CV_CAP_PROP_POS_FRAMES , max(0,frame_idx))
    while (frame_idx*subSampRate < nframe):
        if useBlobCenter and ((frame_idx % trunclen) == 0):
            print "load foreground blob index matrix file...."
            blobIndLists       = []
            blobIndListfile    = blob_ind_sparse_matrices[frame_idx % trunclen]
            blobIndLists       = pickle.load( open( blobIndListfile, "rb" ) )
            
            blobCenterLists    = []
            blobCenterListfile = blob_center_sparse_lists[frame_idx % trunclen]
            blobCenterLists    = pickle.load( open( blobCenterListfile, "rb" ) )

            
        if not isVideo:
            frame[:,:,:] = cv2.imread(imlist[frame_idx*subSampRate])
        if isVideo:
            try:
                status, frame[:,:,:] = cap.read()
            except:
                print "exception!!"
                frame_idx = nframe
                continue

        if useBlobCenter:
            BlobIndMatrixCurFrm = (blobIndLists[np.mod(frame_idx,trunclen)]).todense()
            BlobCenterCurFrm    = blobCenterLists[np.mod(frame_idx,trunclen)]

        frameL[:,:]  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
        ## histogram equalization, more contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        frameL_he = clahe.apply(frameL)
        frameL = frameL_he

        # for visulization
        vis = frame.copy()

        """Tracking"""
        # bad_idx =[]
        if len(tracksdic) > 0:
            try:
                pnts_old = np.float32([tracksdic[i][-1][:2] for i in sorted(tracksdic.keys())]).reshape(-1, 1, 2)
            except: 
                pnts_old = np.float32([tracksdic[i][:2] for i in sorted(tracksdic.keys())]).reshape(-1, 1, 2)

            pnts_new, st, err  = cv2.calcOpticalFlowPyrLK(frameLp, frameL, 
                                                          pnts_old, None, 
                                                          **lk_params)
            pnts_oldr, st, err = cv2.calcOpticalFlowPyrLK(frameL, frameLp, 
                                                          pnts_new, None, 
                                                          **lk_params)
            dist = abs(pnts_old-pnts_oldr).reshape(-1, 2).max(-1)
            good = dist < 1

     
            for (x, y), good_flag, idx in zip(pnts_new.reshape(-1, 2), good, sorted(tracksdic.keys())):
                x = min(x,frameLp.shape[1]-1)
                y = min(y,frameLp.shape[0]-1)

                if not good_flag:
                    end[idx] = (frame_idx-1)
                    tracksdic[idx].append((-100,-100,frame_idx))
                    # bad_idx.append(idx)
                    continue
                if x != -100:
                    # if x>frameLp.shape[1] or y>frameLp.shape[0]:
                    #     print x, y
                    if useBlobCenter:
                        blobInd    = BlobIndMatrixCurFrm[y,x]
                        if blobInd!=0:
                            blobCenter = BlobCenterCurFrm[blobInd-1]
                            tracksdic[idx].append((x,y,frame_idx,np.int8(blobInd),blobCenter[0],blobCenter[1]))
                        else:
                            tracksdic[idx].append((x,y,frame_idx,0,np.NaN,np.NaN))

                    else:
                        tracksdic[idx].append((x,y,frame_idx))
                cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)


        """Detecting new points"""
        if frame_idx % detect_interval == 0: 

            # GGD: this is (I *think*) eliminating redundant non-moving points
            mask[:,:] = 255
            for x, y in [np.int32(tracksdic[tr][-1][:2]) for tr in tracksdic.keys()]:
                cv2.circle(mask, (x, y), 5, 0, -1)    

            corners = cv2.goodFeaturesToTrack(frameL,mask=mask,**feature_params)
            # if frame_idx==69605:
            #     pdb.set_trace()
            if corners is not None:
                for x, y in np.float32(corners).reshape(-1, 2):
                    # create new dic item using new dicidx since these are new points:
                    tracksdic[dicidx] = [] 
                    start[dicidx]     = frame_idx
                    end[dicidx]       = -1
                    if useBlobCenter:
                        blobInd    = BlobIndMatrixCurFrm[y,x]
                        blobCenter = BlobCenterCurFrm[blobInd]
                        tracksdic[dicidx].append((x,y,frame_idx,np.int8(blobInd),blobCenter[0],blobCenter[1]))
                    else:
                        tracksdic[dicidx].append((x,y,frame_idx))
                    dicidx += 1

        print('{0} - {1}'.format(frame_idx*subSampRate,len(tracksdic)))

        if isVisualize:
            # cv2.imshow('klt', vis)
            # cv2.waitKey(5)    
            plt.imshow(vis[:,:,::-1])
            plt.pause(0.00001)
    
        # switch previous frame
        frameLp[:,:] = frameL[:,:]
        frame_idx   += 1


        # dump trajectories to file
        # trunclen = min(trunclen,frame_idx - frame_idx/trunclen*600) #the very last truncation length may be less than original trunclen 
        if  ((frame_idx>0) & ((frame_idx) % trunclen == 0)) or (frame_idx==nframe):
            print "saving===!!!"   
            print('{0} - {1}'.format(frame_idx,len(tracksdic)))         
            Xtracks = np.zeros([len(tracksdic),trunclen])
            Ytracks = np.zeros([len(tracksdic),trunclen])
            # initialize T track using numbers that will never appear in reality
            # "won't-appear" fillers": frame_idx+3*trunclen
            # this way, we won't lose the REAL 0's, i.e. starts from 0 frame, when filtering in the trj_filter.py
            Ttracks = np.ones([len(tracksdic),trunclen])*(frame_idx+3*trunclen)
            if useBlobCenter:
                BlobIndtracks = np.zeros([len(tracksdic),trunclen]) #blob index starts from 1
                BlobCenterX   = np.zeros([len(tracksdic),trunclen]) 
                BlobCenterY   = np.zeros([len(tracksdic),trunclen]) 
            # set first frame in this chunk
            offset  = frame_idx - trunclen

            # loop through the current trajectories list
            for ii, trjidx in enumerate(tracksdic.keys()):

                # set the starting and ending frame index
                st_ind = start[trjidx]
                en_ind = end[trjidx]

                # if en_ind is -1, then the traj is still alive,
                # otherwise, the trajectory is dead (but still in the
                # tracks dictionary, otherwise it would have been
                # removed).
                if en_ind==-1: #not yet finished, save whole row
                    ttrack = np.array(tracksdic[trjidx]).T
                else: #already ended within this truncation
                    ttrack = np.array(tracksdic[trjidx][:-1]).T # don't save -100s

                # if st_ind is -1, then the track existed in the previous
                # truncation and all points except the last one of the
                # previous truncation were removed, so only save from the
                # second point.
                # if st_ind=='fromPre':
                if st_ind<offset:
                    # print "trj point is from previous truncation!"
                    st_ind = offset
                    ttrack = ttrack[:,1:] #because the first point is the last point from pre trunc, already saved

                # put trajectory into matrix
                tstart, tstop = st_ind-offset, en_ind-offset+1

                if en_ind==-1:
                    Xtracks[ii,:][tstart:tstart+len(ttrack[0,:])] = ttrack[0,:]
                    Ytracks[ii,:][tstart:tstart+len(ttrack[1,:])] = ttrack[1,:]
                    Ttracks[ii,:][tstart:tstart+len(ttrack[2,:])] = ttrack[2,:]
                    if useBlobCenter:
                        BlobIndtracks[ii,:][tstart:] = ttrack[3,:]
                        BlobCenterX[ii,:][tstart:]   = ttrack[4,:]
                        BlobCenterY[ii,:][tstart:]   = ttrack[5,:]

                else:
                    Xtracks[ii,:][tstart:tstop] = ttrack[0,:]
                    Ytracks[ii,:][tstart:tstop] = ttrack[1,:]
                    Ttracks[ii,:][tstart:tstop] = ttrack[2,:]
                    if useBlobCenter:
                        BlobIndtracks[ii,:][tstart:tstop] = ttrack[3,:]
                        BlobCenterX[ii,:][tstart:tstop]   = ttrack[4,:]
                        BlobCenterY[ii,:][tstart:tstop]   = ttrack[5,:]





            # put tracks into sparse matrix
            trk ={}
            # Ttracks = Ttracks+frame_idx_bias # use the actual frame index as the key, to save data
            trk['xtracks']       = csr_matrix(Xtracks)
            trk['ytracks']       = csr_matrix(Ytracks)
            trk['Ttracks']       = Ttracks
            trk['trjID']         = tracksdic.keys()
            if useBlobCenter:
                trk['fg_blob_index']    = csr_matrix(BlobIndtracks)
                trk['fg_blob_center_X'] = csr_matrix(BlobCenterX)
                trk['fg_blob_center_Y'] = csr_matrix(BlobCenterY)
            # for dead tracks, remove them.  for alive tracks, remove all
            # points except the last one (in order to track into the next
            # frame).
            deadtrj   = np.array(end.keys())[np.array(end.values())>=0]# ==0 is for the case when the tracks ends at 0 frame
            # lenoffset += len(deadtrj)
            for i in sorted(tracksdic.keys()):   
                if i in deadtrj:
                    tracksdic.pop(i)
                else:
                    tracksdic[i] = [tracksdic[i][-1]]#save the last one
            # pdb.set_trace()
            trk['lastPtsValue'] = np.array(tracksdic.values())[:,0,:]
            trk['lastPtsKey']   = np.array(tracksdic.keys())
            # pdb.set_trace()
            # save as matlab file... :-/
            # savename = './klt/20150222_Mat/HR_w_T______test_' + \
                # str(frame_idx/trunclen).zfill(3)

            # savename = '../DoT/5Ave@42St-96.81/klt/5Ave@42St-96.81_2015-06-16_16h04min40s686ms/' + str(frame_idx/trunclen).zfill(3)
            savename = os.path.join(savePath,'klt_'+str(frame_idx/trunclen).zfill(3))
            savemat(savename,trk)













