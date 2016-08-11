import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

class Visualization():
    def __init__(self,colorNum,firstfrm):
        self.color = np.array([np.random.randint(0,255) \
                           for _ in range(3*int(colorNum))])\
                           .reshape(int(colorNum),3)


        # color = np.array([np.random.randint(0,255) for _ in range(3*int(NumGoodsampleSameDir))]).reshape(NumGoodsampleSameDir,3)
        
        self.nrows = int(np.size(firstfrm,0))
        self.ncols = int(np.size(firstfrm,1))
        self.fig = plt.figure('vis')
        self.axL = plt.subplot(1,1,1)
        self.im  = plt.imshow(np.zeros([self.nrows,self.ncols,3]))
        plt.axis('off')
        plt.ion()

    def visualize_trj(self,labinf,vcxtrj, vcytrj,frame,frame_idx,DataPathobj,VCobj,useVirtualCenter):
        dots = []
        annos = []
        line_exist = 0
        for k in labinf:
            # print "x,y",vcxtrj[k][-1],vcytrj[k][-1]
            if useVirtualCenter:
                xx = np.array(vcxtrj[k])[~np.isnan(vcxtrj[k])]
                yy = np.array(vcytrj[k])[~np.isnan(vcytrj[k])]
            else:
                # xx = np.array(vcxtrj[k]).reshape((1,-1))
                # yy = np.array(vcytrj[k]).reshape((1,-1))
                xx = vcxtrj[k]
                yy = vcytrj[k]
            # if VCobj.VC_filter(vcxtrj[k],vcytrj[k]):
            """there exsits nan's!!!,see why in the future"""

            if len(xx)>0:
                if useVirtualCenter:
                    lines = self.axL.plot(xx,yy,color = (self.color[k-1].T)/255.,linewidth=2)
                    line_exist = 1
                else:
                    """only draw the last 10 points"""
                    # dots.append(self.axL.scatter(xx[-20:],yy[-20:], s=10, color=(self.color[k-1].T)/255.,edgecolor='none')) 
                    dots.append(self.axL.scatter(xx,yy, s=10, color=(self.color[k-1].T)/255.,edgecolor='none')) 
                    
                # annos.append(plt.annotate(str(k),(xx[-1],yy[-1]),fontsize=11))
                # if xx[-1]<=0 or yy[-1]<=0:
                #     pdb.set_trace()


        self.im.set_data(frame[:,:,::-1])
        self.fig.canvas.draw()
        # plt.pause(0.00001) 

        plt.title('frame '+str(frame_idx))
        name = os.path.join(DataPathobj.visResultPath,str(frame_idx).zfill(6)+'.jpg')
        # plt.savefig(name) ##save figure


        """create fake GT"""

        plt.draw()  
        plt.show()
        plt.pause(0.0001)
        # plt.waitforbuttonpress()

        # image2gif = Figtodat.fig2img(fig)
        # images2gif.append(image2gif)

        while line_exist:
            try:
                self.axL.lines.pop(0)
            except:
                line_exist = 0
        for i in dots:
            i.remove()
        for anno in annos:
            anno.remove()    
