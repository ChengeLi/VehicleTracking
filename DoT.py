import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
import time
# import scipy.ndimage.morphology as ndm
# from scipy.ndimage.filters import median_filter as mf


import csv
def visGT():
    ## visualize the Ground Truth

    video_name = '../DoT/Convert3/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms.avi'
    cap   = cv2.VideoCapture(video_name)

    cap.set ( cv2.cv.CV_CAP_PROP_POS_FRAMES , 0)
    color      = np.array([np.random.randint(0,255) for _ in range(3*int(1000))]).reshape(int(1000),3)

    f      =  open('../rejayjohnsonintersectionpairrelationships/Canal_1.csv', 'rb')
    reader = csv.reader(f)

    st,firstfrm = cap.read()
    nrows       = int(np.size(firstfrm,0))
    ncols       = int(np.size(firstfrm,1))

    fig = plt.figure('vis')
    axL = plt.subplot(1,1,1)
    im  = plt.imshow(np.zeros([nrows,ncols,3]))
    plt.axis('off')


    dots = []
    kk = 0
    for ii in range(0,1000,4):
        temp = np.array(reader.next())
        if np.double(temp[0])<frame_idx: # new car
            color = np.array([np.random.randint(0,255) \
                    for _ in range(3*int(1000))])\
                    .reshape(int(1000),3)

        frame_idx = np.double(temp[0])
        GTcenterXY = np.double(temp[-2:])

        cap.set ( cv2.cv.CV_CAP_PROP_POS_FRAMES , frame_idx)
        st,frame = cap.read()


        im.set_data(frame[:,:,::-1])
        plt.draw()

        dots.append(axL.scatter(GTcenterXY[0], GTcenterXY[1], s=10, color=(color[100].T)/255.,edgecolor='none')) 
        plt.draw()
        plt.show()
        plt.pause(0.00001)
        dots = []

        # name = '../GTfigure/'+str(int(kk).zfill(6)+'.jpg'
        # kk = kk+1
        # plt.savefig(name) ##save figure


def read_video(video_name, readlength, skipTime = 0, skipChunk = 0):
    # cap = cv2.VideoCapture('../Videos/TLC00005.AVI')
    # cap = cv2.VideoCapture('./TLC00000.AVI')
    # cap = cv2.VideoCapture('./TLC00001.AVI')  # a different view

    # cap = cv2.VideoCapture('./sternberg_park__mid_block_leonard_st_/TLC00004.AVI')
    cap = cv2.VideoCapture(video_name)
    Numfrm = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) #20480
    scaninterval = int(Numfrm/100.0)
    # startFrm = range(0, Numfrm, scaninterval)


    Frmrate = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
    # Numfrm=readlength
    frameH = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    frameW = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    
    
    vid = np.zeros([readlength,frameH,frameW,3], dtype = np.uint8)
    
    start_position = int(skipTime*(Frmrate))+skipChunk


    print 'reading buffer...'
    # for ii in range(start_position):
    #     print(ii)
    #     rval, img = cap.read()
    # or just set:
    cap.set ( cv2.cv.CV_CAP_PROP_POS_FRAMES , start_position)

    print 'reading frames...'
    for ii in range(0,10*readlength,10):
        true_position = ii+start_position
        cap.set ( cv2.cv.CV_CAP_PROP_POS_FRAMES , true_position)
        print(true_position)
        rval, vid[ii/10] = cap.read()

        # name ='../DoT/5Ave@42St-96.81/5Ave@42St-96.81_2015-06-16_16h04min40s686ms/'+str(true_position).zfill(8)+'.jpg' # save several whole frames for testing 
        # name ='../DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'+str(true_position).zfill(8)+'.jpg' # save several whole frames for testing 
        # name = '/media/My Passport/DoTimgs/Canal/'+str(true_position).zfill(8)+'.jpg'
        # cv2.imwrite(name,vid[ii])
    return vid,start_position



if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1, sharey=False)
    # video_name = '/home/chengeli/CUSP/AIG/DoT/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms_3.avi'
    im = plt.imshow(np.zeros([528,704,3]))
    # video_name = '../DoT/Convert3/5Ave@42St-96.81/5Ave@42St-96.81_2015-06-16_16h04min40s686ms.avi'
    video_name = '../DoT/Convert3/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms.avi'
    # video_name = '../DoT/ASF_files/Canal St @ Baxter St - 96.106_2015-06-18_09h00min00s000ms.asf'
    # video_name = '/Users/Chenge/Desktop/5Ave@42St-96.81_2015-06-16_16h04min40s686ms\ 2.avi'
    # video_name = '/Users/Chenge/Desktop/5Ave@42St-96.81_2015-06-16_18h00min00s002ms.asf'
    # video_name = '../DoT/Convert3/5Ave@42St-96.81/5Ave@42St-96.81_2015-06-16_16h04min40s686ms.avi'

    readlength = 5000
    vid,start_position = read_video(video_name, readlength, skipTime = 360, skipChunk = 0)
    

    # plt.show()
    # # im.axes.figure.canvas.show()
    # for ii in range(0,int(readlength), 1 ):  ## last stopped at 61 for TLC00000, 100 for 3
    # #read every 5 frames
    # ## when jumped out due to bug, just need to change the start
        
    #     true_position = ii+start_position
    #     # true_position = ii+startFrm[kkthInterval]

    #     print ('Now processing: '+str(true_position)+' '+ str(ii))

    #     im.set_data(vid[ii][:,:,::-1])
    #     im.axes.figure.canvas.draw()
    #     # plt.draw()
    #     # time.sleep(0.5)
    #     plt.pause(0.0001) 



    aveBKGsum = np.zeros([528,704,3], dtype = np.float32)
    for ii in range(vid.shape[0]):
        aveBKGsum = aveBKGsum + vid[ii];
    aveBKG = np.uint8(aveBKGsum/(readlength))

    plt.imshow(aveBKG)





# import glob as glob
# imgList = sorted(glob.glob('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/imgs/*.jpg'))
# for iii in range(0,len(imgList),3):
#     print iii
#     imgName = imgList[iii][46:-4]
#     newname = '/media/My Passport/DoTimgs/Jay_Johnson/'+str(imgName)+'.jpg'
#     # cv2.imwrite(name,vid[ii])
#     cv2.imwrite(newname,cv2.imread(imgList[iii]))




















