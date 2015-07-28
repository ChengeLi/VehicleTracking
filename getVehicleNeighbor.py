
#==============find the Neighbours after click======================================
# after clicking on the vehicle interested, the trajectories of this vehicle and its 
# neighbours will be returned and they will be shown in a 2nd figure to visualize.

# left click to select, middle click to confirm, right click to cancel 

import cPickle as pickle
from Trj_class_and_func_definitions import *
from glob import glob
import cv2



obj_pair = pickle.load(open("./mat/20150222_Mat/obj_pair.p","rb"))
# obj_pair2 = pickle.load(open("./mat/20150222_Mat/obj_pair2.p","rb"))
fig, ax1 = plt.subplots(1, 1, sharey=False)


image_listing = sorted(glob('../VideoData/20150222/*.jpg'))
       


def findNeighbors(frame_idx, IDs_in_frame, TrjObject):
    NeighboursID = []
    targ_pts = []
    radius = 100
    
    # pdb.set_trace()

    tmpName= image_listing[frame_idx]
    frame=cv2.imread(tmpName)
    im1=ax1.imshow(frame[:,:,::-1])
    # im1.set_data(uint8 (frame))
    fig.subplots_adjust(0,0,1,1)
    fig.canvas.draw()

    while not targ_pts:
        print "please select tartget points!"
        targ_pts=fig.ginput(0,0, mouse_add=1, mouse_pop=3, mouse_stop=2)#(4,-1) #list 

    targ_w = targ_pts[0][0]
    targ_h = targ_pts[0][1]

    IDalive = IDs_in_frame[frame_idx]
    for iddd in IDalive:
        VehicleObjjj = VehicleObj(TrjObject,iddd)
        starttime = VehicleObjjj.frame[0]
        xjjj = VehicleObjjj.xTrj[frame_idx-starttime]
        yjjj = VehicleObjjj.yTrj[frame_idx-starttime]
        
        if (xjjj-targ_w)**2+(yjjj-targ_h)**2<=radius**2:
            NeighboursID.append(iddd)

    return NeighboursID


IDs_in_frame = pickle.load(open("./mat/20150222_Mat/IDs_in_frame.p","rb" ))


for frame_idx in range(10):
    NeighboursID = findNeighbors(frame_idx, IDs_in_frame, obj_pair)
    print "frame",frame_idx, ":"
    print  NeighboursID

















