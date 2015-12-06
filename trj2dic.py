# ============================================================
# This function ransforms trj representation into dictionary formats
# to make constructing trj object more conveniently  TrjObj()


# If set isVisualize = True, this program can fully replace the role of visualization.py
# Two options are provided: isClustered or not.
# We can either view non-clustered raw trjs, or clustered final results.

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
        # idx = ((x-mx)<2*sx)&((y-my)<2*sy)
        idx = ((x-mx)<sx)&((y-my)<sy)
        vcx = np.mean(x[idx])
        vcy = np.mean(y[idx])
    return vcx,vcy




def get_XYT_inDic(matfiles,start_frame_idx, isClustered, clustered_result, trunclen, isVisualize,isVideo, dataPath ,isSave, savePath, useVirtualCenter=False):  # get the time dictionary, such as vctime

    if isVideo:    
        video_src   = dataPath
        cap         = cv2.VideoCapture(video_src)
        st,firstfrm = cap.read()
    else:
        image_list = sorted(glob.glob(dataPath + '*.jpg'))
        firstfrm   = cv2.imread(image_list[0])
    

    nrows     = int(np.size(firstfrm,0))
    ncols     = int(np.size(firstfrm,1))
    fig = plt.figure('vis')
    axL = plt.subplot(1,1,1)
    im  = plt.imshow(np.zeros([nrows,ncols,3]))
    plt.axis('off')




    # applicable to both clustered and non-clustered trj datas
    # If non-clustered trjs, the input mlabels are just the trj ID (trjID)
    
    lasttrunkTrjFile = loadmat(matfiles[-1]) ##the last one, to get the max index number
    IDintrunklast    = lasttrunkTrjFile['trjID'][0] 
    color            = np.array([np.random.randint(0,255) \
                       for _ in range(3*int(max(IDintrunklast)))])\
                       .reshape(int(max(IDintrunklast)),3)

    # initialization
    vcxtrj = {} ##dictionary
    vcytrj = {}
    vctime = {}
    

    if isClustered:
        trjID   = np.uint32(loadmat(clustered_result)['trjID'][0]) # labeled trjs' indexes
        mlabels = np.int32(np.ones(max(trjID)+1)*-1)  #initial to be -1
        labels  = loadmat(clustered_result)['label'][0]
        for idx,ID in enumerate(trjID):  # ID=trjID[idx], the content, trj ID
            mlabels[int(ID)] = np.int32(labels[int(idx)])

        for i in np.unique(mlabels):  
            vcxtrj[i]=[] 
            vcytrj[i]=[]
            vctime[i]=[] 
    else:
        mlabels = np.int32(np.ones(max(IDintrunklast)+1)*-1)  #initial to be -1
        for i in range(max(IDintrunklast)+1): 
            vcxtrj[i]=[]
            vcytrj[i]=[]
            vctime[i]=[]

    global notconnectedLabel
    notconnectedLabel =[]
    frame_idx = start_frame_idx
    while frame_idx < np.int32(matfiles[-1][-7:-4])*600:
        print "frame = ", str(frame_idx)
        if (frame_idx % trunclen == 0):
            trunkTrjFile = loadmat(matfiles[(frame_idx)/trunclen])
            xtrj         = csr_matrix(trunkTrjFile['xtracks'], shape=trunkTrjFile['xtracks'].shape).toarray()
            ytrj         = csr_matrix(trunkTrjFile['ytracks'], shape=trunkTrjFile['ytracks'].shape).toarray()
            IDintrunk    = trunkTrjFile['trjID'][0]
            Nsample      = trunkTrjFile['xtracks'].shape[0] # num of trjs in this trunk
            fnum         = trunkTrjFile['xtracks'].shape[1] # 600
            ttrj         = csr_matrix(trunkTrjFile['Ttracks'], shape=trunkTrjFile['Ttracks'].shape).toarray()
            # pdb.set_trace()

            startT = np.int32(np.ones([Nsample,1])*-999)
            endT   = np.int32(np.ones([Nsample,1])*-999)
            for i in range(Nsample):  # for the ith sample## get the time T (where the pt appears and disappears)
                havePt  = np.array(np.where(xtrj[i,:]!=0))[0]
                if len(havePt)!=0:
                    startT[i] = np.int32( min(havePt)+(frame_idx/trunclen*trunclen) )
                    endT[i]   = np.int32( max(havePt)+(frame_idx/trunclen*trunclen) )
                    """only for check, can delete trj"""
                    if startT[i]!=np.nanmin(ttrj[i,:])or endT[i]!=np.nanmax(ttrj[i,:]):
                        print "wrong time===========!!!, want to delete this trj?"
                        # fix me..... delete the trj
                        pdb.set_trace()
            #  get the mlabels for unclustered trjs, just the original global ID
            if not isClustered:
                for idx,ID in enumerate(IDintrunk):  
                    mlabels[int(ID)] = np.int32(IDintrunk[int(idx)])


            #  get the vctime
            labinT = list(set(mlabels[IDintrunk])) # label in this trunk

            for k in np.unique(labinT):
                k = np.int32(k)
                if k !=-1:      
                    t1list = startT[mlabels[IDintrunk]==k]  # consider all IDs in the trunk, not only alive in curFrm
                    t2list = endT[mlabels[IDintrunk]==k]
                    t1     = t1list[t1list!=-999]
                    t2     = t2list[t2list!=-999]
                    if len(t1)*len(t2)!=0:
                        startfrm = np.int32(min(t1[t1!=-999])) # earliest appering time in this trj group
                        endfrm   = np.int32(max(t2[t2!=-999]))   # latest disappering time in this trj group
                    else:
                        print "!!!!error!!!!!!there are no trjs in class", str(k)
                        print "It's Ok to skip these...Now only consider left lane"
                        startfrm =-888
                        endfrm   =-888
                        continue

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
                            # pdb.set_trace() 
        # current frame index is: (frame_idx%trunclen)
        PtsInCurFrm = xtrj[:,frame_idx%trunclen]!=0 # in True or False, PtsInCurFrm appear in this frame,i.e. X!=0
        IDinCurFrm  = IDintrunk[PtsInCurFrm] #select IDs in this frame
        labinf      = list(set(mlabels[IDinCurFrm])) # label in current frame
        # print "labinf: ",labinf

        for k in np.unique(labinf):
            if k != -1:
                x = xtrj.T[frame_idx%trunclen][(mlabels[IDintrunk]==k)&PtsInCurFrm]
                y = ytrj.T[frame_idx%trunclen][(mlabels[IDintrunk]==k)&PtsInCurFrm]
                
                if useVirtualCenter:
                    vx,vy = Virctr(x,y) # find virtual center
                else:
                    vx = x
                    vy = y
                # if vx<0 or vy<0:
                #     continue
                vcxtrj[k].append(vx) 
                vcytrj[k].append(vy)

        if isVisualize:
            if isVideo:
                cap.set (cv2.cv.CV_CAP_PROP_POS_FRAMES,frame_idx)
                status, frame = cap.read()
            else:
                frame   = cv2.imread(image_list[frame_idx])
            pdb.set_trace()
            visualize_trj(fig,axL,im,labinf,vcxtrj,vcytrj,frame, color,frame_idx,start_frame_idx)
            

        if isSave: #not clustered yet
            savenameT = os.path.join(savePath,'vctime_'+str(frame_idx/trunclen).zfill(3))+'.p'
            savenameX = os.path.join(savePath,'vcxtrj_'+str(frame_idx/trunclen).zfill(3))+'.p'
            savenameY = os.path.join(savePath,'vcytrj_'+str(frame_idx/trunclen).zfill(3))+'.p'
            if not isClustered:    
                if frame_idx>=599 and ((frame_idx+1) % trunclen == 0):
                    print "Save the dictionary into a pickle file, trunk:", str(frame_idx/trunclen)
                    save_vctime = {}
                    save_vcxtrj = {}
                    save_vcytrj = {}
                    print "notconnectedLabel:", notconnectedLabel
                    for i in np.unique(IDintrunk): 
                        save_vctime[i] = np.array(vctime[i])
                        save_vcxtrj[i] = np.array(vcxtrj[i])
                        save_vcytrj[i] = np.array(vcytrj[i])
                    pickle.dump( save_vctime, open( savenameT, "wb" ) )
                    pickle.dump( save_vcxtrj, open( savenameX, "wb" ) )
                    pickle.dump( save_vcytrj, open( savenameY, "wb" ) )

        frame_idx = frame_idx+1
        # end of while loop

    if isSave and isClustered:
        print "notconnectedLabel:",notconnectedLabel
        savenameT = os.path.join(savePath,'final_vctime.p')
        savenameX = os.path.join(savePath,'final_vcxtrj.p')
        savenameY = os.path.join(savePath,'final_vcytrj.p')
        if isClustered: # is clustered
            save_vctime = {}
            save_vcxtrj = {}
            save_vcytrj = {}            
            clean_vctime = {key: value for key, value in vctime.items() if key not in notconnectedLabel}
            clean_vcxtrj = {key: value for key, value in vcxtrj.items() if key not in notconnectedLabel}
            clean_vcytrj = {key: value for key, value in vcytrj.items() if key not in notconnectedLabel}

            for i in np.int32(np.unique(list(set(vctime.keys())-set(notconnectedLabel)))): 
                save_vctime[i] = np.array(clean_vctime[i])
                save_vcxtrj[i] = np.array(clean_vcxtrj[i])
                save_vcytrj[i] = np.array(clean_vcytrj[i])
            pickle.dump( save_vctime, open( savenameT, "wb" ) )
            pickle.dump( save_vcxtrj, open( savenameX, "wb" ) )
            pickle.dump( save_vcytrj, open( savenameY, "wb" ) )


def visualize_trj(fig,axL,im, labinf,vcxtrj, vcytrj,frame, color,frame_idx,start_frame_idx):
    dots       = []
    line_exist = 0

    # if frame_idx == 5:
    #     pdb.set_trace()

    for k in np.unique(labinf):
        if k !=-1:
            # if (len(list(vcxtrj[k][frame_idx-start_frame_idx]))==1) and (len(np.array(vcytrj[k][frame_idx-start_frame_idx]))==1): #only the virtual center
            print "x,y",vcxtrj[k],vcytrj[k]
            pdb.set_trace()
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




def prepare_data_to_vis(isAfterWarpping,isLeft,isVideo, dataSource):
    if isAfterWarpping:
        if isLeft:
            matPath         = '../DoT/CanalSt@BaxterSt-96.106/leftlane/'
            matfiles        = sorted(glob.glob(matPath +'warpped_'+'*.mat'))
            left_image_list = sorted(glob.glob('../DoT/CanalSt@BaxterSt-96.106/leftlane/img/*.jpg'))
            image_list      = left_image_list
            # final result for vis
            clustered_result = '../DoT/CanalSt@BaxterSt-96.106/leftlane/result/final_warpped_left'
            savePath         = "../DoT/CanalSt@BaxterSt-96.106/leftlane/result/"

        else:
            matPath          = '../DoT/CanalSt@BaxterSt-96.106/rightlane/'
            matfiles         = sorted(glob.glob(matPath +'warpped_'+'*.mat'))
            right_image_list = sorted(glob.glob('../DoT/CanalSt@BaxterSt-96.106/rightlane/img/*.jpg'))
            image_list       = right_image_list
            # final result for vis
            clustered_result = '../DoT/CanalSt@BaxterSt-96.106/rightlane/result/final_warpped_right'
            savePath         = "../DoT/CanalSt@BaxterSt-96.106/rightlane/result/"
    else:
        if dataSource == 'DoT':
            """Linux Canal"""
            matfiles               = sorted(glob.glob('/media/My Book/CUSP/AIG/DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/klt/filtered/len' +'*.mat'))
            clustered_result_files = sorted(glob.glob('/media/My Book/CUSP/AIG/DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'+'*mat'))
            savePath               = '/media/My Book/CUSP/AIG/DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'
            result_file_Ind        = 0 # use the clustered result for the 2nd truncs(26-50)
            clustered_result       = clustered_result_files[result_file_Ind]
            """Mac Canal"""
            # matfiles   = sorted(glob.glob('../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/filtered/len' +'*.mat'))
            # clustered_result       = '../DoT/CanalSt@BaxterSt-96.106/finalresult/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/priorssc5030' 
            # savePath   = '../DoT/CanalSt@BaxterSt-96.106/dic/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'        
            
            if isVideo:
                dataPath = '../DoT/Convert3/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms.avi'
            else:
                dataPath = '../DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/'

        if dataSource == 'Johnson':
            """Jay & Johnson"""
            matfiles         = sorted(glob.glob('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/klt/filtered/' +'*.mat'))
            dataPath         = '/media/My Book/CUSP/AIG/Jay&Johnson/roi2/imgs/'
            clustered_result = '/media/My Book/CUSP/AIG/Jay&Johnson/roi2/Complete_result'
            savePath         = '/media/My Book/CUSP/AIG/Jay&Johnson/roi2/dic/'
            result_file_Ind  = 0 # use complete result

    return matfiles,dataPath,clustered_result,savePath,result_file_Ind



if __name__ == '__main__':
# def trj2dic_main(isVideo, dataSource):
    # isVideo    = True
    # dataSource = 'DoT'
    isVideo    = False
    dataSource = 'Johnson'

    trunclen         = 600
    isClustered      = True
    isAfterWarpping  = False
    isVisualize      = False
    # isVisualize = True
    useVirtualCenter = True
    isLeft           = False
    isSave           = True
    matfiles,dataPath,clustered_result, savePath,result_file_Ind = prepare_data_to_vis(isAfterWarpping,isLeft,isVideo, dataSource)
    # start_frame_idx = (np.int32(matfiles[result_file_Ind*25][-7:-4])-1)*600 #start frame_idx
    start_frame_idx = 0
    print "start_frame_idx: ",start_frame_idx
    matfiles        = matfiles[result_file_Ind*25:(result_file_Ind+1)*25]
    get_XYT_inDic(matfiles,start_frame_idx, isClustered, clustered_result, trunclen, isVisualize,isVideo, dataPath ,isSave, savePath, useVirtualCenter=useVirtualCenter)
