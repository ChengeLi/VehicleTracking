import cv2,time
import matplotlib.pyplot as plt
from matplotlib.pylab import * 
import numpy as np
import scipy.ndimage as nd
import sys
import os


import pdb


class patch:
    def __init__(self,fig,ax1, ax2,files,isVideoFlag,frmInd):
        self.fig=fig
        self.ax1=ax1
        self.ax2=ax2
      
        self.coordinates=[]
        self.again=1
        self.isVideoFlag=isVideoFlag

        if self.isVideoFlag==True:
            self.filename=None
            self.frame = files
            self.curname='frm'+str(frmInd)
        else: ## image file
            self.filename=files
            self.frame = nd.imread(self.filename)
            sidx = self.filename[::-1].index('/')
            eidx = self.filename[::-1].index('.')+1
            self.curname = self.filename[-sidx:-eidx]     #mask default name
            
        nrows,ncols,nd = self.frame.shape
        self.tmp = np.zeros([nrows+100,ncols+100,nd])
        self.tmp[50:nrows+50,50:ncols+50,:] = self.frame[:,:,::-1]

        self.im1=self.ax1.imshow(self.tmp)
        self.im2=self.ax2.imshow(self.tmp)   


    def connect(self):
        'connect to all the events we need'
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.fig.canvas.mpl_disconnect(self.cid)

    def on_key(self,event):
        print('you pressed', event.key, event.xdata, event.ydata)
        if event.key == 'right':
            # self.again = 0
            savepath = '../patch_park'


            for ii in range(size(self.coordinates)/2/2):
                # savename = savepath+'/'+self.curname+'_'+str(ii).zfill(3)+'.npy' # save as numpy array
                savename = savepath+'/'+self.curname+'_'+str(ii+1)+'.npy'
                patchpts=np.array(self.coordinates).round().astype(int)  #shape is 1*4*2 why 1??
                patchpts=patchpts[ii]  
                # -- get upper and lower corners
                # pdb.set_trace()
                ul, lr = patchpts[:,::-1][0:2]
                # -- convert to patch
                # pat = self.frame[ul[0]-50:lr[0]-50,ul[1]-50:lr[1]-50]
                

                height = float(lr[1]-ul[1])
                width  = float(lr[0]-ul[0])
                # centerHori=ul[0]+0.5*width
                # centerVert=ul[1]+0.5*height
                # center=(centerHori,centerVert)

                # append 1/8 of width border, 1/16 of height border
                pat = self.frame[ul[0]-width/float(8)-50:lr[0]+width/float(8)-50,ul[1]-height/float(16)-50:lr[1]+height/float(16)-50]
                # pdb.set_trace()
                np.save(savename,pat)
                savenameIMG = savepath+'/'+self.curname+'_'+str(ii+1)+'.png'
                cv2.imwrite(savenameIMG,pat)
                print('successfully saved!')

        else:
            if event.key == 'down':
                print('Move on to next image!')
                self.again = 0
                self.disconnect()
                self.getpatch()
                
                # print('I am here!!')
                # # pdb.set_trace()
                # return 

            else:
                self.again = 1
                print('Not save! Please re-select!')
                self.getpatch()

         


    def getpatch(self):
        self.coordinates=[] ## clear the coordinates        
        # pdb.set_trace()
        self.im1.set_data(uint8(self.tmp))
        self.fig.subplots_adjust(0,0,1,1)
        self.fig.canvas.draw()

        cnt = 1
        alpha = 0.5 #transparncy
        tmpmap = np.zeros(self.tmp.shape, np.uint8)
 


        while self.again==1:
            print("In the while loop!! please select region {0}...".format(cnt))
            print ('size of self.coordinates == ',size(self.coordinates))

            if self.again==1: #double check 
                newpts=self.fig.ginput(0,0, mouse_add=1, mouse_pop=3, mouse_stop=2)#(4,-1) #list 

                pts = np.array(newpts).round().astype(int)
                self.coordinates.append(pts)
                # cv2.fillPoly(tmpmap, pts =[pts.astype(int)], color=(255,0,0))
                print pts
            
            if size(self.coordinates)%4 !=0:
                print ("Selected points dimension wrong, reselect, please!!")
                print ('size(self.coordinates)%2  ==',size(self.coordinates)%2 )
                del self.coordinates[:]   ##delete the whole list
                continue


            for jj in range(int(size(self.coordinates)/2/2)):
                temp_pts=self.coordinates[jj] ## [2*jj:2*jj+2
                if temp_pts.shape[0]!=0:
                    cv2.rectangle(tmpmap,tuple(temp_pts[0]),tuple(temp_pts[1]),(0,255,0),2) ## must be tuple! 
                    # plt.figure(2),plt.imshow(uint8(tmpmap*alpha+self.tmp*(1-alpha))[:,:,::-1])
                    self.im2.set_data(uint8(tmpmap[:,:,::-1]*alpha+self.tmp[:,:,::-1]*(1-alpha)))
                    self.fig.subplots_adjust(0,0,1,1)
                    self.fig.canvas.draw()


        if self.again==0:
            return 
             


def loopofmain(fig,ax1, ax2,files,isVideoFlag,frmInd):      
    getpatchinstance=patch(fig,ax1, ax2,files,isVideoFlag,frmInd)
    getpatchinstance.connect()
    getpatchinstance.getpatch()
    getpatchinstance.disconnect()





# if __name__ == '__main__':
#     fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)

    
#     print 'press the right arrow key to proceed'
#     path1 = '/home/chengeli/CUSP/wholeFrames/'
#     isVideoFlag=False   
#     listing = os.listdir(path1)    
#     for ii in range(len(listing)):
#         files= path1+listing[ii]
#         print ('Now processing: '+listing[ii])
    
#         loopofmain(fig,ax1, ax2,files, isVideoFlag,ii)









# ===========comments and suggestions
# change to rectangle 
# change y/n 

# pickle is binary, save all things, including packages, ... 
# instead, save to an array 
# np.save('patch.npy',foo)

# load:
# test=np.load('/Users/Chenge/Desktop/CUSP/GUImask/maxresdefault.npy')
# figure(),plt.imshow(test,'gray')


# patch{0:05}.npy.format(ii)     # create the names with leading zeros 



# tunnel the X server
# ssh -X compute 




def findNeighbors():
    fig, ax1 = plt.subplots(1, 2, sharey=False)

    newpts=self.fig.ginput(0,0, mouse_add=1, mouse_pop=3, mouse_stop=2)#(4,-1) #list 

    pts = np.array(newpts).round().astype(int)
    self.coordinates.append(pts)


    self.im1=self.ax1.imshow(self.tmp)

































