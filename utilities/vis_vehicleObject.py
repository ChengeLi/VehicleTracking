# this is to visualize the vehicle objects one by one
# directly from the object fire generated from getVehiclesPair.py
# using clean_obj_pair2.p

import sys
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import csv
import cPickle as pickle
import cv2
import pdb
from scipy.io import loadmat, savemat


def visualize_trj(fig,axL,im, labinf,vcxtrj, vcytrj,frame, color,frame_idx,start_frame_idx):
    dots       = []
    line_exist = 0


    for k in np.unique(labinf):
        if k !=-1:
            # if (len(list(vcxtrj[k][frame_idx-start_frame_idx]))==1) and (len(np.array(vcytrj[k][frame_idx-start_frame_idx]))==1): #only the virtual center
            print "x,y",vcxtrj[k],vcytrj[k]
            # pdb.set_trace()
            line       = axL.plot(vcxtrj[k],vcytrj[k],color = (color[k-1].T)/255.,linewidth=2)
            line_exist = 1
                # dots.append(axL.scatter(vcxtrj[k], vcxtrj[k], s=50, color=(color[k-1].T)/255.,edgecolor='black')) 
                # dots.append(axL.scatter(x, y, s=50, color=(color[k-1].T)/255.,edgecolor='none')) 
                # dots.append(axL.scatter(x, y, s=50, color=(1,0,0),edgecolor='none'))
            # else:
            #     """if draw dots"""
            #     for point in range(len(vcxtrj[k][-1])): #only need to plot the last one
            #         # pdb.set_trace()
            #         print "k = ", str(k), "point = ", str(point)
            #         dots.append(axL.scatter(vcxtrj[k][-1][point], vcytrj[k][-1][point], s=30, color=(color[k-1].T)/255.))
            #     """if draw lines"""
            #     # for kk in range(frame_idx):
            #     #     pdb.set_trace()
            #     #     for point in range(len(vcxtrj[k][frame_idx])): 
            #     #         line       = axL.plot(vcxtrj[k][:frame_idx][point],vcytrj[k][:][point],color = (color[k-1].T)/255.,linewidth=2)
            #     #         line_exist = 1

            """plot point, all trjs instead of just VC"""
            # for point in range(len(vcxtrj[k][-1])): #only need to plot the last one
            #     # print "k = ", str(k), "point = ", str(point)
            #     dots.append(axL.scatter(vcxtrj[k][-1][point], vcytrj[k][-1][point], s=20, color=(color[k-1].T)/255.))

    im.set_data(frame[:,:,::-1])
    fig.canvas.draw()
    plt.draw()
    plt.pause(0.00001) 

    # name = './canalResult/original/'+str(frame_idx).zfill(6)+'.jpg'
    # plt.savefig(name) ##save figure

    while line_exist :
        try:
            axL.line.pop(0)
        except:
            line_exist = 0
    for i in dots:
        i.remove()

    plt.show()



def visualize_single_trj(fig,axL,im,vcxtrj, vcytrj,frame, Color):
    dots       = []
    line_exist = 0
    # line       = axL.plot(vcxtrj,vcytrj,color = Color,linewidth=2)
    # line_exist = 1
    im.set_data(frame[:,:,::-1])
    # pdb.set_trace()
    # dots.append(axL.scatter(vcxtrj, vcytrj, s=20, color=Color,edgecolor='none')) 
    dots.append(plt.scatter(vcxtrj, vcytrj, s=8, color=Color,edgecolor='none')) 

    plt.draw()
    # plt.show()
    plt.pause(0.00001)
    dots = []

    
    # fig.canvas.draw()
    # plt.draw()
    # name = './canalResult/original/'+str(frame_idx).zfill(6)+'.jpg'
    # plt.savefig(name) ##save figure

    while line_exist :
        try:
            axL.line.pop(0)
        except:
            line_exist = 0
    for i in dots:
        i.remove()
    plt.draw()




if __name__ == '__main__':
    dataSource  = 'Johnson'
    isVideo     = False
    dataPath    = '/media/My Book/CUSP/AIG/Jay&Johnson/roi2/imgs/' 
    isVisualize = True

    if dataSource == 'Johnson':
        image_list = sorted(glob.glob('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/imgs/*.jpg'))
        # clean_vctime = pickle.load(open('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/Pair_clean_vctime.p','rb'))
        # clean_vcxtrj = pickle.load(open('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/Pair_clean_vcxtrj.p','rb'))
        # clean_vcytrj = pickle.load(open('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/Pair_clean_vcytrj.p','rb'))
        # obj_pair2 = TrjObj(clean_vcxtrj,clean_vcytrj,clean_vctime)
        obj_pair2  = pickle.load(open('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/subSamp/dic/complete_500-5-1/500-5-1clean_obj_pair2.p','rb'))
        savePath   = "/media/My Book/CUSP/AIG/Jay&Johnson/roi2/subSamp/pair_relationship/complete_500-5-1/"
        
        # obj_pair2  = pickle.load(open('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/alpha=0dot001/clean_obj_pair2.p','rb'))
        # savePath   = "/media/My Book/CUSP/AIG/Jay&Johnson/roi2/alpha=0dot001/"
    if isVideo:  
        video_src   = dataPath
        cap         = cv2.VideoCapture(video_src)
        st,firstfrm = cap.read()
    else:
        image_list = sorted(glob.glob(dataPath + '*.jpg'))
        firstfrm   = cv2.imread(image_list[0])

    nrows = int(np.size(firstfrm,0))
    ncols = int(np.size(firstfrm,1))

    final_clusterSize  = pickle.load(open('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/subSamp/dic/complete_500-5-1/final_clusterSize.p','rb'))
    # clusterSize20 = loadmat('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/subSamp/500-5-1Result/clusterSize20up.mat') 
    # clusterSize20_label = clusterSize20['cluster20up'][0]
    # temp = np.unique(np.int32(list(np.floor(clusterSize20_label)) + list(np.ceil(clusterSize20_label))))




    color = np.array([np.random.randint(0,255) for _ in range(3*int(max(obj_pair2.globalID)))]).reshape(int(max(obj_pair2.globalID)),3)
    interval = 20
    for intervalind in range(0,np.int(len(obj_pair2.globalID)/interval)+1):
        if intervalind>0:
            name = savePath+'big_cluster_size'+str((intervalind-1)*interval)+'-'+str((intervalind)*interval)+'.jpg'
            plt.savefig(name)
            plt.cla()
        fig   = plt.figure('vis')
        axL   = plt.subplot(1,1,1)
        im    = plt.imshow(np.zeros([nrows,ncols,3]))
        plt.axis('off')
        for kk in obj_pair2.globalID[intervalind*interval:(intervalind+1)*interval]:
        # for kk in set(obj_pair2.globalID).intersection(set(temp)):
            # print "now showing:",kk
            if (sum(final_clusterSize[kk])== len(final_clusterSize[kk])) and (len(final_clusterSize[kk])<10):
                print "single trj as cluster!"
                continue
            

            im         = plt.imshow(np.zeros([nrows,ncols,3]))
            living_ran = obj_pair2.frame[kk]
            vcxtrj     = obj_pair2.xTrj[kk]
            vcytrj     = obj_pair2.yTrj[kk]
            kthColor   = (color[kk-1].T)/255.

            frame_idx = living_ran[0]+1
            # print "frame_idx:", frame_idx
            print("frame {0}\r".format(frame_idx)),
            sys.stdout.flush()
            if isVisualize:
                if isVideo:
                    cap.set (cv2.cv.CV_CAP_PROP_POS_FRAMES,frame_idx)
                    status, frame = cap.read()
                else:
                    frame   = cv2.imread(image_list[frame_idx])
                    visualize_single_trj(fig,axL,im,vcxtrj, vcytrj,frame, kthColor)

             # fix me , show by frame   
            # subSampRate = 6
            # for frame_idx in range(47*5*60,57*5*60,1):
            #     realFrame_idx = frame_idx*subSampRate


            #     labinf      = list(set(mlabels[IDinCurFrm])) 
            #     PtsInCurFrm = obj_pair2.frame[ID]!=0 # in True or False, PtsInCurFrm appear in this frame,i.e. X!=0
            #     IDinCurFrm  = IDintrunk[PtsInCurFrm] #select IDs in this frame
            #     labinf = 
            #     visualize_trj(fig,axL,im, labinf, vcxtrj, vcytrj,frame, kthColor,frame_idx):




































