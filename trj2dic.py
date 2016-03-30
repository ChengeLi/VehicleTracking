# ============================================================
# This function ransforms trj representation into dictionary formats
# to make constructing trj object more conveniently  TrjObj()


# If set isVisualize = True, this program can fully replace the role of visualization.py
# Two options are provided: isClustered or not.
# We can either view non-clustered raw trjs, or clustered final results.

import cv2
import os
import sys
import pdb
import pickle
import numpy as np
import glob as glob
from scipy.io import loadmat,savemat
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from DataPathclass import *
DataPathobj = DataPath(dataSource,VideoIndex)
from parameterClass import *
Parameterobj = parameter(dataSource,VideoIndex)




def Virctr(x,y):
    '''
    calculate virtual center, and remove outlier
    '''
    # x = x[x!=0]
    # y = y[y!=0]

    # if len(x)!=len(y):
    #     return np.nan, np.nan

    # if len(x)<3:
    #     vcx = np.mean(x)
    #     vcy = np.mean(y)
    # else:
    #     mx = np.mean(x)
    #     my = np.mean(y)
    #     sx = np.std(x)
    #     sy = np.std(y)
    #     # idx = ((x-mx)<2*sx)&((y-my)<2*sy)
    #     idx = ((x-mx)<=sx)&((y-my)<=sy)
    #     vcx = np.mean(x[idx])
    #     vcy = np.mean(y[idx])
    if len(x)<3:
        vcx = np.median(x)
        vcy = np.median(y)
    else:
        mx = np.mean(x)
        my = np.mean(y)
        sx = np.std(x)
        sy = np.std(y)

        idx = ((x-mx)<=sx)&((y-my)<=sy)
        vcx = np.median(x[idx])
        vcy = np.median(y[idx])

        """discard very big group???"""
        # if sx>20 or sy>20:
        # if sx>0.8*Parameterobj.nullDist_for_adj or sy>0.8*nullDist_for_adj:
        #     vcx = np.nan
        #     vcy = np.nan
        # else:
        #     # idx = ((x-mx)<2*sx)&((y-my)<2*sy)
        #     idx = ((x-mx)<=sx)&((y-my)<=sy)
        #     vcx = np.median(x[idx])
        #     vcy = np.median(y[idx])
    return vcx,vcy


def VC_filter(vcxtrj,vcytrj):
    Goodvc = True
    if len(vcxtrj)>=2:
        vcxspd = np.abs(np.diff(vcxtrj))
        vcyspd = np.abs(np.diff(vcytrj))
        # pdb.set_trace()
        max_vcxspd = np.max(vcxspd)
        max_vcyspd = np.max(vcyspd)

        if (max_vcxspd>=10) and (max_vcyspd >= 10):
            Goodvc = False

        if (np.sum(vcxspd)<=5) or (np.sum(vcyspd) <= 5):
            # print "static!"
            Goodvc = False

    return Goodvc




def get_XYT_inDic(matfiles,start_frame_idx, isClustered, clustered_result, trunclen, isVisualize,isVideo, dataPath ,isSave, savePath, useVirtualCenter=False):  # get the time dictionary, such as vctime

    if isVisualize:
        if isVideo:    
            video_src   = dataPath
            cap         = cv2.VideoCapture(video_src)
            st,firstfrm = cap.read()
        else:
            image_list = sorted(glob.glob(dataPath + '*.jpg'))
            firstfrm   = cv2.imread(image_list[0])
        
        nrows     = int(np.size(firstfrm,0))
        ncols     = int(np.size(firstfrm,1))
        # plt.ion()
        fig = plt.figure('vis')
        axL = plt.subplot(1,1,1)
        im  = plt.imshow(np.zeros([nrows,ncols,3]))
        plt.axis('off')

    # applicable to both clustered and non-clustered trj datas
    # If non-clustered trjs, the input mlabels are just the trj ID (trjID)
    
    lasttrunkTrjFile = loadmat(matfiles[-1]) ##the last one, to get the max index number
    if len(lasttrunkTrjFile['trjID'])<1:
        lasttrunkTrjFile = loadmat(matfiles[-2])  # in case the last one is empty
    IDintrunklast    = lasttrunkTrjFile['trjID'][0]
    color = np.array([np.random.randint(0,255) \
                       for _ in range(3*int(max(IDintrunklast)))])\
                       .reshape(int(max(IDintrunklast)),3)

    # initialization
    vcxtrj = {} 
    vcytrj = {}
    vctime = {}
    clusterSize = {}

    if isClustered:
        trjID   = np.uint32(loadmat(clustered_result)['trjID'][0]) # labeled trjs' indexes
        mlabels = np.int32(np.ones(max(trjID)+1)*-1)  #initial to be -1
        labels  = loadmat(clustered_result)['label'][0]
        for idx,ID in enumerate(trjID):  # ID=trjID[idx], the content, trj ID
            mlabels[int(ID)] = np.int(labels[int(idx)])

        for i in np.unique(mlabels):  
            vcxtrj[i]=[] 
            vcytrj[i]=[]
            vctime[i]=[] 
            clusterSize[i] = []
    else:
        mlabels = np.int32(np.ones(max(IDintrunklast)+1)*-1)  #initial to be -1
        for i in range(max(IDintrunklast)+1): 
            vcxtrj[i]=[]
            vcytrj[i]=[]
            vctime[i]=[]

    global notconnectedLabel
    notconnectedLabel =[]
    CrossingClassLbel = [] # class labels that go across 2 trunks
    frame_idx = start_frame_idx
    subsample_frmIdx = np.int(np.floor(frame_idx/subSampRate))

    while frame_idx < np.int(matfiles[-1][-7:-4])*subSampRate*trunclen:
        print("frame {0}\r".format(frame_idx)),
        sys.stdout.flush()
        if subsample_frmIdx%trunclen == 0:
            matidx = np.int(np.floor(subsample_frmIdx/trunclen))
            trunkTrjFile = loadmat(matfiles[matidx])
            xtrj = csr_matrix(trunkTrjFile['xtracks'], shape=trunkTrjFile['xtracks'].shape).toarray()
            ytrj = csr_matrix(trunkTrjFile['ytracks'], shape=trunkTrjFile['ytracks'].shape).toarray()
            # xtrj = csr_matrix(trunkTrjFile['xtracks_warpped'], shape=trunkTrjFile['xtracks'].shape).toarray()
            # ytrj = csr_matrix(trunkTrjFile['ytracks_warpped'], shape=trunkTrjFile['ytracks'].shape).toarray()
            if len(trunkTrjFile['trjID'])==0: ##encounter empty file, move on
                subsample_frmIdx = subsample_frmIdx+trunclen
                continue
            IDintrunk = trunkTrjFile['trjID'][0]
            Nsample   = trunkTrjFile['xtracks'].shape[0] # num of trjs in this trunk
            ttrj      = csr_matrix(trunkTrjFile['Ttracks'], shape=trunkTrjFile['Ttracks'].shape).toarray()
            ttrj[ttrj==np.max(ttrj[:])]=np.nan
            startT = np.int32(np.ones([Nsample,1])*-999)
            endT   = np.int32(np.ones([Nsample,1])*-999)
            for i in range(Nsample):  # for the ith sample## get the time T (where the pt appears and disappears)
                # havePt  = np.array(np.where(xtrj[i,:]!=0))[0]
                # if len(havePt)!=0:
                #     startT[i] = np.int((min(havePt)+(subsample_frmIdx/trunclen*trunclen))*subSampRate)
                #     endT[i]   = np.int((max(havePt)+(subsample_frmIdx/trunclen*trunclen))*subSampRate)
                #     """only for check, can delete trj"""
                #     if startT[i]!=np.nanmin(ttrj[i,:])or endT[i]!=np.nanmax(ttrj[i,:]):
                #         pdb.set_trace()
                #         print "wrong time===========!!!, want to delete this trj?"
                #         # fix me..... delete the trj
                startT[i] =np.nanmin(ttrj[i,:])
                endT[i]   =np.nanmax(ttrj[i,:])

            #  get the mlabels for unclustered trjs, just the original global ID
            if not isClustered:
                for idx,ID in enumerate(IDintrunk):  
                    mlabels[int(ID)] = np.int(IDintrunk[int(idx)])

            #  get the vctime
            labinT = mlabels[IDintrunk] # label in this trunk
            for k in np.unique(labinT):
                k = np.int(k)
                if k !=-1:      
                    t1list = startT[labinT==k]  # consider all IDs in the trunk, not only alive in curFrm
                    t2list = endT[labinT==k]
                    t1 = t1list[t1list!=-999]
                    t2 = t2list[t2list!=-999]
                    if len(t1)*len(t2)!=0:
                        startfrm = np.int(np.min(t1))   # earliest appering time in this trj group
                        endfrm   = np.int(np.max(t2))   # latest disappering time in this trj group
                    else:
                        print "!!!!error!!!!!!there are no trjs in class", str(k)
                        print "It's Ok to skip these...Now only consider left lane"
                        pdb.set_trace()
                        startfrm =-888
                        endfrm   =-888
                        continue

                    if not vctime[k]:
                        vctime[k] =  [int(startfrm),int(endfrm)]
                    else:
                        print k
                        lastendfrm = vctime[k][-1]
                        laststartfrm = vctime[k][-2]
                        if int(startfrm) == lastendfrm+1*subSampRate:
                            vctime[k] = [laststartfrm, int(endfrm)]
                            CrossingClassLbel.append(k)
                        else:
                            print k
                            print "========same class trjs not overlapping, disconnected==============!"
                            # pdb.set_trace()
                            notconnectedLabel.append(k)
                            vctime[k].append(int(startfrm))
                            vctime[k].append(int(endfrm))
 

        # try: # protect from individable start_frame_idx, continue adding from idx untill loading xtrj first
        #     print xtrj.keys()[0]
        # except:
        #     pdb.set_trace()
        #     subsample_frmIdx   += 1
        #     frame_idx = subsample_frmIdx*subSampRate
        #     continue
        PtsInCurFrm = xtrj[:,subsample_frmIdx%trunclen]!=0 # in True or False, PtsInCurFrm appear in this frame,i.e. X!=0
        IDinCurFrm  = IDintrunk[PtsInCurFrm] #select IDs in this frame
        labinf      = mlabels[IDinCurFrm] # label in current frame
        # print "labinf: ",labinf
        for k in np.unique(labinf):
            if k != -1:
                x = xtrj[PtsInCurFrm,subsample_frmIdx%trunclen][labinf==k]
                y = ytrj[PtsInCurFrm,subsample_frmIdx%trunclen][labinf==k]
                if useVirtualCenter:
                    vx,vy = Virctr(x,y) # find virtual center
                    if np.isnan(vx) or np.isnan(vy):
                        if len(vcxtrj[k])>1:
                            vx = vcxtrj[k][-1]  # duplicate the last (x,y) in label k
                            vy = vcytrj[k][-1]
                            if np.abs(vx-vcxtrj[k][-1])+np.abs(vy-vcytrj[k][-1])>100:
                                vx = vcxtrj[k][-1]  # duplicate the last (x,y) in label k
                                vy = vcytrj[k][-1]

                        # else: append nan's
                        else:
                            pdb.set_trace()
                            # vx= [np.nan]
                            # vy= [np.nan] 
                    vcxtrj[k].extend([vx]) 
                    vcytrj[k].extend([vy])
                else:
                    vx = list(x)
                    vy = list(y)
                    vcxtrj[k].extend(vx) 
                    vcytrj[k].extend(vy)

                # pdb.set_trace()
                # if vx<=0 or vy<=0:  # why exist negative???? 

                if len(x)!=len(y):
                    pdb.set_trace()
                if isClustered:
                    clusterSize[k].extend([len(x)])

        if isVisualize:
            image2gif=[]
            if isVideo:
                trueLoc = start_frame_idx+(trunclen*matidx+(subsample_frmIdx%trunclen))*subSampRate
                cap.set (cv2.cv.CV_CAP_PROP_POS_FRAMES,frame_idx)
                if trueLoc!=frame_idx:
                    pdb.set_trace()

                status, frame = cap.read()
            else:
                frame = cv2.imread(image_list[frame_idx])
            visualize_trj(fig,axL,im,np.unique(labinf)[1:],vcxtrj,vcytrj,frame,color,frame_idx)
            

        if isSave: #not clustered yet
            savenameT = os.path.join(savePath,'vctime_'+str(subsample_frmIdx/trunclen).zfill(3))+'.p'
            savenameX = os.path.join(savePath,'vcxtrj_'+str(subsample_frmIdx/trunclen).zfill(3))+'.p'
            savenameY = os.path.join(savePath,'vcytrj_'+str(subsample_frmIdx/trunclen).zfill(3))+'.p'
            if not isClustered:    
                if subsample_frmIdx>=599 and ((subsample_frmIdx+1) % trunclen == 0):
                    print "Save the dictionary into a pickle file, trunk:", str(subsample_frmIdx/trunclen)
                    save_vctime = {}
                    save_vcxtrj = {}
                    save_vcytrj = {}
                    print "notconnectedLabel:", notconnectedLabel
                    # for i in np.unique(IDintrunk): #for non-clustered, ID is the label
                    for i in np.unique(labinT):
                        save_vctime[i] = np.array(vctime[i])
                        save_vcxtrj[i] = np.array(vcxtrj[i])
                        save_vcytrj[i] = np.array(vcytrj[i])
                    pickle.dump( save_vctime, open( savenameT, "wb" ) )
                    pickle.dump( save_vcxtrj, open( savenameX, "wb" ) )
                    pickle.dump( save_vcytrj, open( savenameY, "wb" ) )

        subsample_frmIdx += 1
        frame_idx = subsample_frmIdx*subSampRate
        # end of while loop

    if isSave and isClustered:
        print "notconnectedLabel:",notconnectedLabel
        print "CrossingClassLbel:",CrossingClassLbel
        savenameT = os.path.join(savePath,'final_vctime.p')
        savenameX = os.path.join(savePath,'final_vcxtrj.p')
        savenameY = os.path.join(savePath,'final_vcytrj.p')
        savenameclusterSize = os.path.join(savePath,'final_clusterSize.p')
        if isClustered: # is clustered
            save_vctime = {}
            save_vcxtrj = {}
            save_vcytrj = {}
            save_clusterSize = {}            
            clean_vctime = {key: value for key, value in vctime.items() if key not in notconnectedLabel}
            clean_vcxtrj = {key: value for key, value in vcxtrj.items() if key not in notconnectedLabel}
            clean_vcytrj = {key: value for key, value in vcytrj.items() if key not in notconnectedLabel}
            clean_clusterSize = {key: value for key, value in clusterSize.items() if key not in notconnectedLabel}

            pickle.dump( clean_vctime, open(savenameT,"wb"))
            pickle.dump( clean_vcxtrj, open(savenameX,"wb"))
            pickle.dump( clean_vcytrj, open(savenameY,"wb"))
            pickle.dump( clean_clusterSize, open(savenameclusterSize,"wb"))

            """duplicate"""
            # for i in np.int32(np.unique(list(set(vctime.keys())-set(notconnectedLabel)))): 
            #     save_vctime[i] = np.array(clean_vctime[i])
            #     save_vcxtrj[i] = np.array(clean_vcxtrj[i])
            #     save_vcytrj[i] = np.array(clean_vcytrj[i])
            #     save_clusterSize[i] = np.array(clean_clusterSize[i])
            


            # pickle.dump( save_vctime, open(savenameT,"wb"))
            # pickle.dump( save_vcxtrj, open(savenameX,"wb"))
            # pickle.dump( save_vcytrj, open(savenameY,"wb"))
            # pickle.dump( save_clusterSize, open(savenameclusterSize,"wb"))
            # pdb.set_trace()

    return images2gif



def visualize_trj(fig,axL,im, labinf,vcxtrj, vcytrj,frame, color,frame_idx):
    plt.ion()
    dots = []
    annos = []
    line_exist = 0
    for k in labinf:
       # print "x,y",vcxtrj[k],vcytrj[k]
        if useVirtualCenter:
            xx = np.array(vcxtrj[k])[~np.isnan(vcxtrj[k])]
            yy = np.array(vcytrj[k])[~np.isnan(vcytrj[k])]
        else:
            # xx = np.array(vcxtrj[k]).reshape((1,-1))
            # yy = np.array(vcytrj[k]).reshape((1,-1))
            xx = vcxtrj[k]
            yy = vcytrj[k]
        # if VC_filter(vcxtrj[k],vcytrj[k]):
        """there exsits nan's!!!,see why in the future"""
        # print "xx",xx
        # print "yy", yy
        # print "label now = ", k 
        if len(xx)>0:
            if useVirtualCenter:
                lines = axL.plot(xx,yy,color = (color[k-1].T)/255.,linewidth=2)
                line_exist = 1
            else:
                dots.append(axL.scatter(xx,yy, s=10, color=(color[k-1].T)/255.,edgecolor='none')) 
            # annos.append(plt.annotate(str(k),(xx[-1],yy[-1])))


    im.set_data(frame[:,:,::-1])
    fig.canvas.draw()
    # plt.draw()
    plt.pause(0.00001) 
    # plt.title('frame '+str(frame_idx))
    # name = os.path.join(DataPathobj.visResultPath,str(frame_idx).zfill(6)+'.jpg')
    # plt.savefig(name) ##save figure
    """sort the annotation list base dn x location. from left to right"""
    annolist = sorted(annos, key=lambda x: x.xy[0], reverse=False) 
    
    # for jj in range(len(annolist)):
    #     print np.int(annolist[jj].get_text())

    plt.draw()
    plt.show()

    # image2gif = Figtodat.fig2img(fig)
    # images2gif.append(image2gif)

    while line_exist:
        try:
            axL.lines.pop(0)
        except:
            line_exist = 0
    for i in dots:
        i.remove()
    for anno in annos:
        anno.remove()    
    plt.draw()
    plt.show()
    

def prepare_data_to_vis(isVideo,isClustered):
    global subSampRate
    subSampRate = np.int(DataPathobj.cap.get(cv2.cv.CV_CAP_PROP_FPS)/Parameterobj.targetFPS)
    matfiles = sorted(glob.glob(os.path.join(DataPathobj.smoothpath,'klt*.mat')))
    """to visulize raw klt"""
    # matfiles = sorted(glob.glob(os.path.join(DataPathobj.kltpath,'klt*.mat')))
    if Parameterobj.useWarpped:
        clustered_result_files = sorted(glob.glob(os.path.join(DataPathobj.unifiedLabelpath,'usewarpped_*'+Parameterobj.clustering_choice+'*.mat')))
    else:
        clustered_result_files = sorted(glob.glob(os.path.join(DataPathobj.unifiedLabelpath,'Complete*'+Parameterobj.clustering_choice+'*.mat')))
        """to visulize the connected component"""
        # clustered_result_files = sorted(glob.glob(os.path.join(DataPathobj.adjpath,'*.mat')))

    savePath = DataPathobj.dicpath
    result_file_Ind  = 0 # use the clustered result for the 2nd truncs(26-50)
    if isClustered:
        clustered_result = clustered_result_files[result_file_Ind]
    else:
        clustered_result =[]
    if isVideo:
        # dataPath = os.path.join(DataPathobj.sysPathHeader,'My Book/CUSP/AIG/DoT/Convert3/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms.avi')
        dataPath = DataPathobj.video
    else:
        dataPath = '../DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'

    return matfiles,dataPath,clustered_result,savePath,result_file_Ind



if __name__ == '__main__':
    isVideo = True
    trunclen         = Parameterobj.trunclen
    isClustered      = True
    isVisualize      = True
    useVirtualCenter = False
    isSave           = False
    matfiles,dataPath,clustered_result, savePath,result_file_Ind = prepare_data_to_vis(isVideo,isClustered)
    # start_frame_idx = (np.int(matfiles[result_file_Ind*25][-7:-4])-1)*trunclen #start frame_idx
    start_frame_idx = 0
    # start_frame_idx = trunclen*subSampRate*6
    print "start_frame_idx: ",start_frame_idx
    # matfiles = matfiles[result_file_Ind*25:(result_file_Ind+1)*25]
    get_XYT_inDic(matfiles,start_frame_idx, isClustered, clustered_result, trunclen, isVisualize,isVideo, dataPath ,isSave, savePath, useVirtualCenter=useVirtualCenter)









