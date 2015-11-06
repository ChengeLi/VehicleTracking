import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
import time
# import scipy.ndimage.morphology as ndm
# from scipy.ndimage.filters import median_filter as mf



def read_video(video_name, readlength, skipTime = 0, skipChunk = 0):
    # cap = cv2.VideoCapture('../Videos/TLC00005.AVI')
    # cap = cv2.VideoCapture('./TLC00000.AVI')
    # cap = cv2.VideoCapture('./TLC00001.AVI')  # a different view

    # cap = cv2.VideoCapture('./sternberg_park__mid_block_leonard_st_/TLC00004.AVI')
    cap = cv2.VideoCapture(video_name)
    Numfrm = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) #20480
    scaninterval = int(Numfrm/100.0)
    # startFrm = range(0, Numfrm, scaninterval)

    kkthInterval =  skipChunk      #start to get dark from 26th

    Frmrate = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
    Numfrm=readlength
    frameH = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    frameW = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    # vid = np.zeros([15350,480,640,3],dtype = uint8)
    vid = np.zeros([Numfrm,frameH,frameW,3], dtype = np.uint8)

    
    

    start_position = int(skipTime*(Frmrate))
    # start_position = startFrm[kkthInterval]


    print 'reading buffer...'
    for ii in range(start_position):
        print(ii)
        rval, img = cap.read()

    print 'reading frames...'
    for ii in range(Numfrm):
        true_position = ii+start_position

        print(true_position)
        rval, vid[ii] = cap.read()
        # name ='../DoT/5Ave@42St-96.81/5Ave@42St-96.81_2015-06-16_16h04min40s686ms/'+str(true_position).zfill(8)+'.jpg' # save several whole frames for testing 
        name ='../DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'+str(true_position).zfill(8)+'.jpg' # save several whole frames for testing 
        cv2.imwrite(name,vid[ii])
    return vid, start_position



if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1, sharey=False)
    # video_name = '/home/chengeli/CUSP/AIG/DoT/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms_3.avi'
    im = plt.imshow(np.zeros([528,704,3]))
    # video_name = '../DoT/Convert3/5Ave@42St-96.81/5Ave@42St-96.81_2015-06-16_16h04min40s686ms.avi'
    video_name = '../DoT/Convert3/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms.mp4'
    # video_name = '../DoT/ASF_files/Canal St @ Baxter St - 96.106_2015-06-18_09h00min00s000ms.asf'
    # video_name = '/Users/Chenge/Desktop/5Ave@42St-96.81_2015-06-16_16h04min40s686ms\ 2.avi'
    # video_name = '/Users/Chenge/Desktop/5Ave@42St-96.81_2015-06-16_18h00min00s002ms.asf'
    # video_name = '../DoT/Convert3/5Ave@42St-96.81/5Ave@42St-96.81_2015-06-16_16h04min40s686ms.avi'

    readlength = 2000
    vid, start_position = read_video(video_name, readlength, skipTime = 0, skipChunk = 0)
    


    plt.show()
    # im.axes.figure.canvas.show()
    aveBKG = np.zeros([528,704,3], dtype = np.float32)
    for ii in range(0,int(readlength), 1 ):  ## last stopped at 61 for TLC00000, 100 for 3
    #read every 5 frames
    ## when jumped out due to bug, just need to change the start
        
        true_position = ii+start_position
        # true_position = ii+startFrm[kkthInterval]

        print ('Now processing: '+str(true_position)+' '+ str(ii))

        im.set_data(vid[ii][:,:,::-1])
        im.axes.figure.canvas.draw()
        # plt.draw()
        # time.sleep(0.5)
        plt.pause(0.0001) 
    #     aveBKG  = aveBKG + vid[ii];

    # aveBKG = aveBKG/(readlength+1)



