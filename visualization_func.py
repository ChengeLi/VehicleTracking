import cv2,pdb
import csv
import cPickle as pickle
import pprint
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.sparse import csr_matrix
from scipy.io import loadmat,savemat

def Virctr(x,y):
    '''
    calculate virtual center, and remove out lier
    '''    
    if len(x)<3:
        vcx = np.mean(x)
        vcy = np.mean(y)
    else:
        mx  = np.mean(x)
        my  = np.mean(y)
        sx  = np.std(x)
        sy  = np.std(y)
        idx = ((x-mx)<2*sx)&((y-my)<2*sy)
        vcx = np.mean(x[idx])
        vcy = np.mean(y[idx])
    return vcx,vcy




def visualization(image_listing, finalLabel,TrkFilePath):
    trunclen = 600
    # lrsl     = '../DoT/CanalSt@BaxterSt-96.106/finalresult/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/result' 
    # matfiles = sorted(glob('../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'+'*.mat'))
    
    lrsl     = finalLabel
    matfiles = sorted(glob(TrkFilePath+'*.mat'))
    
    mask     = loadmat(lrsl)['mask'][0]
    labels   = loadmat(lrsl)['label'][0]

    trunkTrjFile  = loadmat(matfiles[-1]) ##the last one, to get the max index number
    IDintrunklast = trunkTrjFile['idxtable'][0]
    mlabels       = np.ones(max(IDintrunklast)+1)*-1
    #build PtsInCurFrm trj labels (-1 : not interest PtsInCurFrm)
    for idx,i in enumerate(mask):  # i=mask[idx], the cotent
        mlabels[i] = labels[idx]
    # mlabel: ID --> label

    vcxtrj = {} ##dictionary
    vcytrj = {}

    vctime = {}
    vctime2 = {}

    for i in np.unique(mlabels):  ## there are several PtsInCurFrm contributing to one label i
        vcxtrj[i]=[] # find a virtual center for each label i
        vcytrj[i]=[]
        vctime[i]=[]

        vctime2[i] = [] 

    image_listing = sorted(glob('../DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/*.jpg'))

    firstfrm=cv2.imread(image_listing[0])
    nrows = int(np.size(firstfrm,0))
    ncols = int(np.size(firstfrm,1))
    framenum = int(len(image_listing))
    framerate = 5
    notconnectedLabel= []

    frame_idx = 0


    # fig = plt.figure(1,figsize=[10,12])
    fig = plt.figure(1)
    axL = plt.subplot(1,1,1)
    frame = np.zeros([nrows,ncols,3]).astype('uint8')
    im = plt.imshow(np.zeros([nrows,ncols,3]))
    plt.axis('off')
    color = np.array([np.random.randint(0,255) \
                   for _ in range(3*int(max(labels)))])\
                   .reshape(max(labels),3)

    # cam.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,frame_idx)





    # framenum = 1300 # for testing
    while (frame_idx < framenum):
        print ("frame_idx" , frame_idx)
        if (frame_idx % trunclen == 0):
            print "load file!!-------------------------------------"
            print frame_idx
            

            trunkTrjFile = loadmat(matfiles[frame_idx//trunclen])
            xtrj = csr_matrix(trunkTrjFile['xtracks'], shape=trunkTrjFile['xtracks'].shape).toarray()
            ytrj = csr_matrix(trunkTrjFile['ytracks'], shape=trunkTrjFile['ytracks'].shape).toarray()
            IDintrunk = trunkTrjFile['idxtable'][0]
            sample = trunkTrjFile['xtracks'].shape[0]
            fnum   = trunkTrjFile['xtracks'].shape[1]

            trk = np.zeros([sample,fnum,3])
            startT = np.ones([sample,1])*-999
            endT = np.ones([sample,1])*-999

        

            for i in range(sample):  # for the ith sample
                trk[i,:,0] = xtrj[i,:]
                trk[i,:,1] = ytrj[i,:]
                # trk[i,:,2] = arange(fnum)


                ## get the time T (where the pt appears and disappears)
                havePt  = np.array(np.where(xtrj[i,:]>0))[0]
                if len(havePt)!=0:
                    startT[i] = int ( min(havePt)+(frame_idx/trunclen*trunclen) )
                    endT[i]   = int ( max(havePt)+(frame_idx/trunclen*trunclen) )
       



            # only execute once for a trunk ============================
            labinT = list(set(mlabels[IDintrunk])) # label in this trunk
            dots = []
            for k in np.unique(labinT):
                if k !=-1:          
                    t1list=startT[mlabels[IDintrunk]==k]  # consider all IDs in the trunk, not only alive in curFrm
                    t2list=endT[mlabels[IDintrunk]==k]
                    

                    t1 = t1list[t1list!=-999]
                    t2 = t2list[t2list!=-999]

                    if len(t1)*len(t2)!=0:
                        startfrm=min(t1[t1!=-999])
                        endfrm=max(t2[t2!=-999])
                    else:
                        # pdb.set_trace()
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
                            print "========not connected==============!"
                            print k
                            print frame_idx
                            notconnectedLabel.append(k)
                            vctime[k].append(int(startfrm))
                            vctime[k].append(int(endfrm))

        if !isVideo:
            frame[:,:,:] = cv2.imread(image_listing[frame_idx])
        if isVideo:
            status, frame[:,:,:] = cap.read()

        plt.draw()
        # current frame index is: (frame_idx%trunclen)
        PtsInCurFrm = trk[:,:,0].T[frame_idx%trunclen]!=0 # in True or False, PtsInCurFrm appear in this frame,i.e. X!=0
        IDinCurFrm = IDintrunk[PtsInCurFrm]

        labinf = list(set(mlabels[IDinCurFrm])) # label in current frame
        dots = []
        for k in np.unique(labinf):
            line_exist = 0 ## CG
            if k !=-1:
                x = trk[:,:,0].T[frame_idx%trunclen][((mlabels==k)[IDintrunk])&PtsInCurFrm]
                y = trk[:,:,1].T[frame_idx%trunclen][((mlabels==k)[IDintrunk])&PtsInCurFrm]
                vx,vy = Virctr(x,y) # find virtual center
                
                vcxtrj[k].append(vx) 
                vcytrj[k].append(vy)
                tempxyIDs = np.where(mlabels==k)
                xyIDs = []

                for xxyyiidd in np.array(tempxyIDs)[0]: 
                    if xxyyiidd in IDinCurFrm:
                        xyIDs.append(xxyyiidd)
                
                # lines = axL.plot(vcxtrj[k],vcytrj[k],color = (0,1,0),linewidth=2)
                lines = axL.plot(vcxtrj[k],vcytrj[k],color = (color[k-1].T)/255.,linewidth=2)
                line_exist = 1

                #dots.append(axL.scatter(vx, vy, s=50, color=(color[k-1].T)/255.,edgecolor='black')) 
                # dots.append(axL.scatter(x, y, s=50, color=(color[k-1].T)/255.,edgecolor='none')) 
                # dots.append(axL.scatter(x, y, s=50, color=(1,0,0),edgecolor='none'))
        im.set_data(frame[:,:,::-1])
        
        fig.canvas.draw()
        plt.draw()
        plt.pause(0.0001) 
      
        
        while line_exist :
            try:
                axL.lines.pop(0)
            except:
                line_exist = 0
        

        for i in dots:
            i.remove()
        
        plt.show()
        
        frame_idx = frame_idx+1
      





    for kkk in notconnectedLabel:
        # print vctime[kkk]
        if np.size(vcxtrj[kkk])==vctime[kkk][1]-vctime[kkk][0]+1:
            vctime[kkk] = [vctime[kkk][0], vctime[kkk][1]]













