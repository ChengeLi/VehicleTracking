# ============================================================
# This function ransforms trj representation into dictionary formats
# to make constructing trj object more conveniently  TrjObj()

# ==============fix me ...==...
import cv2
import os
import pdb
import pickle
import numpy as np
import glob as glob
from scipy.io import loadmat,savemat
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

def Virctr(x,y):
    '''
    calculate virtual center, and remove out lier
    '''    
    if len(x)<3:
        vcx = np.mean(x)
        vcy = np.mean(y)
    else:
        mx = np.mean(x)
        my = np.mean(y)
        sx = np.std(x)
        sy = np.std(y)
        idx = ((x-mx)<2*sx)&((y-my)<2*sy)
        vcx = np.mean(x[idx])
        vcy = np.mean(y[idx])
    return vcx,vcy




def get_XYT_inDic(matfiles,frame_idx, isClustered, trunclen, isVisualize, axL, im, image_listing):  # get the time dictionary, such as vctime
    # applicable to both clustered and non-clustered trj datas
    # If non-clustered trjs, the input mlabels are just the trj ID (mask)
    
    lasttrunkTrjFile = loadmat(matfiles[-1]) ##the last one, to get the max index number
    IDintrunklast = lasttrunkTrjFile['mask'][0] # 26 trjs 
    lastmlabels = np.ones(max(IDintrunklast)+1)*-1  #initial to be -1
    color = np.array([np.random.randint(0,255) \
               for _ in range(3*int(max(lastmlabels)))])\
               .reshape(int(max(lastmlabels)),3)

    # initialization
    vcxtrj = {} ##dictionary
    vcytrj = {}
    vctime = {}

    for i in np.unique(lastmlabels): 
        vcxtrj[i]=[]
        vcytrj[i]=[]
        vctime[i]=[]


    
    notconnectedLabel =[]
    while frame_idx < 1801:
        if (frame_idx % trunclen == 0):
            trunkTrjFile = loadmat(matfiles[frame_idx//trunclen])
            xtrj = csr_matrix(trunkTrjFile['x_re'], shape=trunkTrjFile['x_re'].shape).toarray()
            ytrj = csr_matrix(trunkTrjFile['y_re'], shape=trunkTrjFile['y_re'].shape).toarray()
            IDintrunk = trunkTrjFile['mask'][0]
            Nsample = trunkTrjFile['x_re'].shape[0] # num of trjs in this trunk
            fnum   = trunkTrjFile['x_re'].shape[1] # 600
            ttrj = csr_matrix(trunkTrjFile['Ttracks'], shape=trunkTrjFile['Ttracks'].shape).toarray()

            trk = np.zeros([Nsample,fnum,3])
            for i in range(Nsample):  # for the ith sample
                trk[i,:,0] = xtrj[i,:]
                trk[i,:,1] = ytrj[i,:]
                trk[i,:,2] = ttrj[i,:]

            startT = np.ones([Nsample,1])*-999
            endT = np.ones([Nsample,1])*-999

            for i in range(Nsample):  # for the ith sample## get the time T (where the pt appears and disappears)
                havePt  = np.array(np.where(xtrj[i,:]>0))[0]
                if len(havePt)!=0:
                    startT[i] = int ( min(havePt)+(frame_idx/trunclen*trunclen) )
                    endT[i]   = int ( max(havePt)+(frame_idx/trunclen*trunclen) )
                    # only for check, can delete trj
                    if startT[i]!=np.min(ttrj[i,havePt])or endT[i]!=np.max(ttrj[i,havePt]):
                        print "wrong time===========!!!, want to delete this trj?"
                        pdb.set_trace()
            
            #  get the mlabels
            if isClustered:
                lrsl = '../DoT/CanalSt@BaxterSt-96.106/finalresult/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/priorssc5030' 
                mask = loadmat(lrsl)['mask'][0] # labeled trjs' indexes
                labels = loadmat(lrsl)['label'][0]

            #build PtsInCurFrm trj labels (-1 : not interest PtsInCurFrm)
            for idx,ID in enumerate(mask):  # ID=mask[idx], the content, trj ID
                mlabels[int(ID)] = labels[int(idx)]
            else:
                IDintrunk = trunkTrjFile['mask'][0]
            for idx,ID in enumerate(IDintrunk):  # i=mask[idx], the content
                mlabels[int(ID)] = IDintrunk[int(idx)]


            #  get the vctime
            labinT = list(set(mlabels[IDintrunk])) # label in this trunk
            for k in np.unique(labinT):
                if k !=-1:          
                    t1list=startT[mlabels[IDintrunk]==k]  # consider all IDs in the trunk, not only alive in curFrm
                    t2list=endT[mlabels[IDintrunk]==k]
                    t1 = t1list[t1list!=-999]
                    t2 = t2list[t2list!=-999]

                    if len(t1)*len(t2)!=0:
                        startfrm=min(t1[t1!=-999]) # earliest appering time in this trj group
                        endfrm=max(t2[t2!=-999])   # latest disappering time in this trj group
                    else:
                        pdb.set_trace()
                        print "!!!!error!!!!!!there are no trjs in class", str(k)
                        startfrm=-888
                        endfrm=-888

                    if not vctime[k]:
                        vctime[k] =  [int(startfrm),int(endfrm)]
                    else:
                        lastendfrm = vctime[k][-1]
                        laststartfrm = vctime[k][-2]
                        if int(startfrm) == lastendfrm+1:
                            vctime[k] = [laststartfrm, int(endfrm)]
                        else:
                            print k
                            print "========same class trjs not overlapping, disconnected==============!"
                            notconnectedLabel.append(k)
                            vctime[k].append(int(startfrm))
                            vctime[k].append(int(endfrm))
                            pdb.set_trace() 

        


        # current frame index is: (frame_idx%trunclen)
        PtsInCurFrm = trk[:,:,0][:,frame_idx%trunclen]!=0 # in True or False, PtsInCurFrm appear in this frame,i.e. X!=0
        IDinCurFrm = IDintrunk[PtsInCurFrm] #select IDs in this frame

        labinf = list(set(mlabels[IDinCurFrm])) # label in current frame
        for k in np.unique(labinf):
            if k !=-1:
                x = trk[:,:,0].T[frame_idx%trunclen][((mlabels==k)[IDintrunk])&PtsInCurFrm]
                y = trk[:,:,1].T[frame_idx%trunclen][((mlabels==k)[IDintrunk])&PtsInCurFrm]
                
                if isClustered:
                    vx,vy = Virctr(x,y) # find virtual center
                else:
                    vx = x
                    vy = y            
                vcxtrj[k].append(vx) 
                vcytrj[k].append(vy)

        if isVisualize:
            # Get the frame and visualize!
            # ret, frame[:] = cam.read()
            tmpName= image_listing[frame_idx]
            frame=cv2.imread(tmpName)
            if (frame_idx == 601):
                pdb.set_trace()

            visualize_trj(axL,im,labinf,vcxtrj, vcytrj,frame)
            

        
        if frame_idx>599 and (frame_idx % trunclen == 0):
            savePath = "../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/"
            savenameT = os.path.join(savePath,'vctime_'+str(frame_idx/trunclen).zfill(3))+'.p'
            savenameX = os.path.join(savePath,'vcxtrj_'+str(frame_idx/trunclen).zfill(3))+'.p'
            savenameY = os.path.join(savePath,'vcytrj_'+str(frame_idx/trunclen).zfill(3))+'.p'
            print "Save the dictionary into a pickle file, trunk:", str(frame_idx/trunclen)
            pickle.dump( vctime, open( savenameT, "wb" ) )
            pickle.dump( vcxtrj, open( savenameT, "wb" ) )
            pickle.dump( vcytrj, open( savenameT, "wb" ) )

        frame_idx = frame_idx+1
        # end of while loop


# ====fix me 
def visualize_trj(axL,im, labinf,vcxtrj, vcytrj,frame):
    dots = []
    for k in np.unique(labinf):
        if k !=-1:
            lines = axL.plot(vcxtrj[k],vcytrj[k],color = (color[k-1].T)/255.,linewidth=2)
            line_exist = 1
            # dots.append(axL.scatter(vx, vy, s=50, color=(color[k-1].T)/255.,edgecolor='black')) 
            # dots.append(axL.scatter(x, y, s=50, color=(color[k-1].T)/255.,edgecolor='none')) 
            # dots.append(axL.scatter(x, y, s=50, color=(1,0,0),edgecolor='none'))
    
    im.set_data(frame[:,:,::-1])
        
    fig.canvas.draw()
    plt.draw()
    plt.pause(0.00001) 
    # pdb.set_trace()

    # name = './canalResult/'+str(frame_idx).zfill(6)+'.jpg'
    
    # cv2.imwrite(name, frame)
    # extent = axL.get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
    # plt.savefig(name,bbox_inches=extent) ##save figure
    # plt.savefig(name) ##save figure

    while line_exist :
        try:
            axL.lines.pop(0)
        except:
            line_exist = 0
    for i in dots:
        i.remove()
    plt.show()






if __name__ == '__main__':
    matfiles = sorted(glob.glob('../DoT/CanalSt@BaxterSt-96.106/adj/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/len50' +'*.mat'))
    # frame_idx = 0
    frame_idx = 600
    trunclen = 600
    isClustered = False


    
    # for visualization
    image_listing = sorted(glob.glob('../DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/*.jpg'))
    # image_listing = sorted(glob('./tempFigs/roi2/*.jpg'))
    firstfrm=cv2.imread(image_listing[0])
    nrows = int(np.size(firstfrm,0))
    ncols = int(np.size(firstfrm,1))
    framenum = int(len(image_listing))
    framerate = 5

    fig = plt.figure(1111111)
    axL = plt.subplot(1,1,1)
    im = plt.imshow(np.zeros([nrows,ncols,3]))
    plt.axis('off')



    get_XYT_inDic(matfiles, frame_idx, isClustered, 600, True, axL, im, image_listing)
