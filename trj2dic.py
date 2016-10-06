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

# from DataPathclass import *
# DataPathobj = DataPath(dataSource,VideoIndex)
# from parameterClass import *
# Parameterobj = parameter(dataSource,VideoIndex)



from utilities.VCclass import VirtualCenter
from utilities.videoReading import videoReading
from utilities.VisualizationClass import Visualization


"""global parameters"""
isVideo = True
isClustered      = True
isVisualize      = False
useVirtualCenter = True
isSave           = True
createGT = False
if createGT:
    isClustered = False
    useVirtualCenter = False

useCC = False
useKcenter = False



class fakeGTfromAnnotation():
    """record the annotations, any information from that????"""
    def __init__(self, matidx):
        self.annotation_file = open(DataPathobj.DataPath+'/'+str(matidx).zfill(3)+'annotation_file.txt','wb')
        self.global_annolist = []

        annotation_file.write('frame:')
        annotation_file.write(str(frame_idx))
        annotation_file.write('\n')

        # within visualization part
        """sort the annotation list base dn x location. from left to right"""
        if createGT:
            # annolist = sorted(annos, key=lambda x: sqrt(x.xy[0]**2+x.xy[1]**2), reverse=False) 
            annolist = sorted(annos, key=lambda x: x.xy[0], reverse=False)
            
            for jj in range(len(annolist)):
                annotation_file.write(str(np.int(annolist[jj].get_text())))
                global_annolist.append(np.int(annolist[jj].get_text())) 
                # annotation_file.write(str(annolist[jj].xy))
                annotation_file.write('\n')



        # self.annotation_file.close()
        # pickle.dump(global_annolist,open(DataPathobj.DataPath+'/'+str(matidx).zfill(3)+'global_annolist.p','wb'))

class trjEachTrunc(object):
    """trj in each truncation"""
    def __init__(self, matidx, trj2dicObj):
        """get vcxtrj, vcytrj, vctime in dictionary format"""
        self.trunkTrjFile = loadmat(trj2dicObj.matfiles[matidx])
        if len(self.trunkTrjFile['trjID'])==0: ##encounter empty file, move on
            # subsample_frmIdx = subsample_frmIdx+Parameterobj.trunclen
            # continue
            # pdb.set_trace()
            print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!error'

        trj2dicObj.checkFileExist(matidx)

        self.xtrj = csr_matrix(self.trunkTrjFile['xtracks'], shape=self.trunkTrjFile['xtracks'].shape).toarray()
        self.ytrj = csr_matrix(self.trunkTrjFile['ytracks'], shape=self.trunkTrjFile['ytracks'].shape).toarray()
        # self.xtrj = csr_matrix(self.trunkTrjFile['xtracks_warpped'], shape=self.trunkTrjFile['xtracks'].shape).toarray()
        # self.ytrj = csr_matrix(self.trunkTrjFile['ytracks_warpped'], shape=self.trunkTrjFile['ytracks'].shape).toarray()
        self.IDintrunk = self.trunkTrjFile['trjID'][0]

            
        """as we deleted small isolated connected component trjs in trjcluster"""
        self.adjFile = loadmat(trj2dicObj.adjFiles[matidx])
        self.non_isolatedCC = self.adjFile['non_isolatedCCupup']
        self.xtrj = self.xtrj[self.non_isolatedCC[0,:],:]
        self.ytrj = self.ytrj[self.non_isolatedCC[0,:],:]
        self.IDintrunk = self.IDintrunk[self.non_isolatedCC[0,:]]
        self.Nsample   = self.xtrj.shape[0] # num of trjs in this trunk

        #  get the mlabels for unclustered trjs, just the original global ID
        if not isClustered:
            for idx,ID in enumerate(self.IDintrunk):  
                trj2dicObj.mlabels[int(ID)] = np.int(self.IDintrunk[int(idx)])
        self.labinT = trj2dicObj.mlabels[self.IDintrunk] # label in this trunk
       
        """get vctime trunk-wise, one vctime per trunk"""
        # self.ttrj      = csr_matrix(trunkTrjFile['Ttracks'], shape=trunkTrjFile['Ttracks'].shape).toarray()
        # self.ttrj = self.ttrj[self.non_isolatedCC,:]
        # self.ttrj[self.ttrj==np.max(self.ttrj[:])]=np.nan
        # self.startT = np.int32(np.ones([self.Nsample,1])*-999)
        # self.endT   = np.int32(np.ones([self.Nsample,1])*-999)
        # for i in range(self.Nsample):  # for the ith sample## get the time T (where the pt appears and disappears)
        #     self.startT[i] =np.nanmin(self.ttrj[i,:])
        #     self.endT[i]   =np.nanmax(self.ttrj[i,:])

class trj2dic(object):
    """labels are already available, group trajectories with the same labels"""
    """convert into dictionary format"""

    def __init__(self):
        self.prepare_input_data()
        # initialization
        self.vcxtrj = {} 
        self.vcytrj = {}
        self.vctime = {}
        self.clusterSize = {}
        self.vctime2 = {} #for test

        # applicable to both clustered and non-clustered trj datas
        # If non-clustered trjs, the input mlabels are just the trj ID (trjID)
        lasttrunkTrjFile = loadmat(self.matfiles[-1]) ##the last one, to get the max index number
        if len(lasttrunkTrjFile['trjID'])<1:
            lasttrunkTrjFile = loadmat(matfiles[-2])  # in case the last one is empty
        IDintrunklast    = lasttrunkTrjFile['trjID'][0]
        if isClustered:
            self.trjID   = np.uint32(loadmat(self.clustered_result)['trjID'][0]) # labeled trjs' indexes
            self.mlabels = np.int32(np.ones(max(self.trjID)+1)*-1)  #initial to be -1
            labels  = loadmat(self.clustered_result)['label'][0]
            for idx,ID in enumerate(self.trjID):  # ID=trjID[idx], the content, trj ID
                self.mlabels[int(ID)] = np.int(labels[int(idx)])
            for i in np.unique(self.mlabels):  
                self.vcxtrj[i]=[] 
                self.vcytrj[i]=[]
                self.vctime[i]=[] 
                self.vctime2[i] = []
                self.clusterSize[i] = []
        else:
            self.mlabels = np.int32(np.ones(max(IDintrunklast)+1)*-1)  #initial to be -1
            for i in range(max(IDintrunklast)+1): 
                self.vcxtrj[i]=[]
                self.vcytrj[i]=[]
                self.vctime[i]=[]
                self.vctime2[i] = []
        self.notconnectedLabel =[]
        self.CrossingClassLbel = [] # class labels that go across 2 trunks

    def prepare_input_data(self):
        self.matfiles = sorted(glob.glob(os.path.join(DataPathobj.smoothpath,'*.mat')))
        """to visulize raw klt"""
        # self.matfiles = sorted(glob.glob(os.path.join(DataPathobj.kltpath,'*.mat')))
        """to visulize the connected component"""
        if useCC:
            clustereFileName = 'concomp*'
        else:
            clustereFileName = 'Complete_result*'
        
        if Parameterobj.useWarpped:
            clustereFileName = 'usewarpped_'+clustereFileName
            self.adjFiles = sorted(glob.glob(DataPathobj.adjpath +'*usewarpped_*.mat'))
        else:
            self.adjFiles = sorted(glob.glob(DataPathobj.adjpath +'normalize*.mat'))

        clustereFileName = clustereFileName[:-1]+Parameterobj.clustering_choice+'*'
        clustered_result_files = sorted(glob.glob(os.path.join(DataPathobj.unifiedLabelpath,clustereFileName)))
        self.savePath = DataPathobj.dicpath
        self.result_file_Ind  = 0 # use the clustered result for the 2nd truncs(26-50)
        if isClustered:
            self.clustered_result = clustered_result_files[self.result_file_Ind]
        else:
            self.clustered_result =[]
        if isVideo:
            self.dataPath = DataPathobj.video
        else:
            self.dataPath = DataPathobj.imagePath


    def checkFileExist(self,matidx):
        if useCC:
            assert matidx<=len(self.adjFiles), "already used up the adj files"
        else:
            assert matidx<=len(sorted(glob.glob(DataPathobj.sscpath +'*0*.mat'))), "already used up the ssc files"


    def KCenter(self):
        """try k-center algo to see whether output is similar with initial groups"""
        sys.path.insert(-1,'/Users/Chenge/Desktop/k-center-problem-master/k_center')
        from k_center import *
        points = []
        for mm in np.array(range(xtrj.shape[0]))[PtsInCurFrm]:
            xx = xtrj[mm,subsample_frmIdx%Parameterobj.trunclen]
            yy = ytrj[mm,subsample_frmIdx%Parameterobj.trunclen]
            points = points+[Point(xx, yy)]
        
        k_center = KCenter(points)
        k_center = KCenter_by_threshold(points)

        locations = k_center.furtherst_first(10, start_location=points[0])
        if isVisualize:
            print locations
        k_center.chenge_plot_point(locations)

        threshold = 50
        locations2 = k_center.chenge_given_threshold(threshold, start_location=points[0])
        k_center.chenge_plot_point(locations2,axL)

    def getVCtime(self, trjEachTruncObj):
        for k in np.unique(trjEachTruncObj.labinT):
            k = np.int(k)
            if k !=-1:      
                t1list = trjEachTruncObj.startT[trjEachTruncObj.labinT==k]  # consider all IDs in the trunk, not only alive in curFrm
                t2list = trjEachTruncObj.endT[trjEachTruncObj.labinT==k]
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

                if not self.vctime[k]:
                    self.vctime[k] =  [int(startfrm),int(endfrm)]
                else:
                    print k," this class has already appeared."
                    lastendfrm = self.vctime[k][-1]
                    laststartfrm = self.vctime[k][-2]
                    if int(startfrm) == lastendfrm+1*subSampRate:
                        print k,"========same class trjs connected==============!"
                        self.vctime[k] = [laststartfrm, int(endfrm)]
                        self.CrossingClassLbel.append(k)

                    else:
                        print "========same class trjs not overlapping, disconnected==============!"
                        self.notconnectedLabel.append(k)
                        # self.vctime[k].append(int(startfrm))
                        # self.vctime[k].append(int(endfrm))
                        self.vctime[k].append(-2000)
                        self.vctime[k].append(-2000)


    def getVCtrjs(self, frame_idx, trjEachTruncObj,VCobj):
        # self.getVCtime()
        subsample_frmIdx = np.int(np.floor(frame_idx/subSampRate))
        PtsInCurFrm = trjEachTruncObj.xtrj[:,subsample_frmIdx%Parameterobj.trunclen]!=0 # in True or False, PtsInCurFrm appear in this frame,i.e. X!=0 
        # PtsInCurFrm = trjEachTruncObj.xtrj[:,floor(float(subsample_frmIdx)/Parameterobj.trunclen)]!=0  ## this is wrong!
        IDinCurFrm  = trjEachTruncObj.IDintrunk[PtsInCurFrm] #select IDs in this frame
        self.labinf      = self.mlabels[IDinCurFrm] # label in current frame
        # print "labinf: ",self.labinf
        for k in np.unique(self.labinf):
            if k != -1:
                x = trjEachTruncObj.xtrj[PtsInCurFrm,subsample_frmIdx%Parameterobj.trunclen][self.labinf==k]
                y = trjEachTruncObj.ytrj[PtsInCurFrm,subsample_frmIdx%Parameterobj.trunclen][self.labinf==k]
                if useVirtualCenter:
                    VCobj.getVC(x[x!=0],y[y!=0])
                    vx,vy =  VCobj.vcx, VCobj.vcy# find virtual center; exist zero points.. don't know why
                    if np.isnan(vx) or np.isnan(vy):
                        if len(vcxtrj[k])>1:
                            vx = vcxtrj[k][-1]  # duplicate the last (x,y) in label k
                            vy = vcytrj[k][-1]
                            if np.abs(vx-vcxtrj[k][-1])+np.abs(vy-vcytrj[k][-1])>100:
                                vx = vcxtrj[k][-1]  # duplicate the last (x,y) in label k
                                vy = vcytrj[k][-1]

                        # else append nan's
                        else:
                            pdb.set_trace()
                            # vx= [np.nan]
                            # vy= [np.nan]
                    self.vcxtrj[k].extend([vx]) 
                    self.vcytrj[k].extend([vy])
                else:
                    vx = list(x)
                    vy = list(y)
                    self.vcxtrj[k].extend(vx) 
                    self.vcytrj[k].extend(vy)
                    # if np.sum(x<=0)>0 or np.sum(y<=0)>0:  # why exist negative???? 
                    #     pdb.set_trace()

                """get vctime with the X and Y"""
                self.vctime2[k].extend([frame_idx])
                # if len( vctime2[k])!=len( vcxtrj[k]):
                #     print "frame_idx=",frame_idx
                #     print "k=",k
                #     print "vctime2[k]", vctime2[k]
                #     print "vcxtrj[k]", vcxtrj[k]
                if isClustered:
                    self.clusterSize[k].extend([len(x)])

    def get_XYT_inDic(self,start_frame_idx,videoReadingObj,VCobj):  # get the time dictionary, such as vctime
        frame_idx = start_frame_idx
        subsample_frmIdx = np.int(np.floor(frame_idx/subSampRate))
        
        if isVisualize:
            if isVideo:
                firstframe = videoReadingObj.getFrame_loopy()
                videoReadingObj.reset(DataPathobj.video) ##after reading the first frm, reset to index 0

                #read the buffer till start_frame_idx
                videoReadingObj.readBuffer(start_frame_idx)
            else:
                image_list = sorted(glob.glob(dataPath + '*.jpg'))
                firstframe = cv2.imread(image_list[0])            
            
            visualizationObj = Visualization(max(self.mlabels),firstframe)

        """initialize """
        while frame_idx < np.int(self.matfiles[-1][-7:-4])*subSampRate*Parameterobj.trunclen:
            print("frame {0}\r".format(frame_idx))
            sys.stdout.flush()

            matidx = np.int(np.floor(subsample_frmIdx/Parameterobj.trunclen))
            trjEachTruncObj = trjEachTrunc(matidx, self)
            self.getVCtrjs(frame_idx,trjEachTruncObj,VCobj)
            """visualize trj here"""
            if isVisualize:
                if isVideo:
                    frame = videoReadingObj.getFrame_loopy()
                else:
                    frame = cv2.imread(image_list[frame_idx])
                visualizationObj.visualize_trj(np.unique(self.labinf)[:],self.vcxtrj, self.vcytrj,frame,frame_idx,DataPathobj,VCobj,useVirtualCenter)

            """saving"""
            if isSave and not isClustered:
                if subsample_frmIdx>=599 and ((subsample_frmIdx+1) % Parameterobj.trunclen == 0):
                    self.saveTrjDict_nonClustered(trjEachTruncObj,matidx)
                    self.clean_after_save()

            if isSave and isClustered:
                # if frame_idx == np.int(self.matfiles[-1][-7:-4])*subSampRate*Parameterobj.trunclen - subSampRate: #to the end
                if subsample_frmIdx>=599 and ((subsample_frmIdx+1) % Parameterobj.trunclen == 0):
                    print 'frame_idx ',frame_idx,' subsample_frmIdx ',subsample_frmIdx
                    print "finished all files... save all!"
                    self.saveTrjDict(matidx)
                    self.clean_after_save()

            subsample_frmIdx += 1
            frame_idx = subsample_frmIdx*subSampRate


                
    def clean_after_save(self):
        print "!===========cleaning, re-initialization ==========!"
        self.__init__()


    def saveTrjDict_nonClustered(self, trjEachTruncObj,matidx):
        savenameT = os.path.join(self.savePath,'vctime_'+str(matidx).zfill(3))+'.p'
        savenameX = os.path.join(self.savePath,'vcxtrj_'+str(matidx).zfill(3))+'.p'
        savenameY = os.path.join(self.savePath,'vcytrj_'+str(matidx).zfill(3))+'.p'
        print "Save the dictionary into a pickle file, trunk:", str(matidx)
        save_vctime = {}
        save_vcxtrj = {}
        save_vcytrj = {}
        save_vctime2 = {}
        print "notconnectedLabel:", self.notconnectedLabel
        for i in np.unique(trjEachTruncObj.labinT):# same with np.unique(trjEachTruncObj.IDintrunk), for non-clustered, ID is the label
            save_vctime[i] = np.array(self.vctime[i])
            save_vcxtrj[i] = np.array(self.vcxtrj[i])
            save_vcytrj[i] = np.array(self.vcytrj[i])
            save_vctime2[i] = np.array(self.vctime2[i])

        pickle.dump( save_vctime, open( savenameT, "wb" ) )
        pickle.dump( save_vcxtrj, open( savenameX, "wb" ) )
        pickle.dump( save_vcytrj, open( savenameY, "wb" ) )
        pickle.dump( save_vctime2, open(os.path.join(self.savePath,'vctime_consecutive_frame'+str(matidx).zfill(3)+'.p'),'wb'))

    def saveTrjDict(self,matidx):
        """must save the virtual center, since the class size is changing"""
        print "notconnectedLabel:",self.notconnectedLabel
        print "CrossingClassLbel:",self.CrossingClassLbel

        if useCC:
            """ save CC_ connected component"""
            savenameT = os.path.join(self.savePath,'CC_final_vctime.p')
            savenameX = os.path.join(self.savePath,'CC_final_vcxtrj.p')
            savenameY = os.path.join(self.savePath,'CC_final_vcytrj.p')
            savenameclusterSize = os.path.join(self.savePath,'CC_final_clusterSize.p')
            savenameT_consecutive = os.path.join(self.savePath,'CC_final_vctime_consecutive_frame.p')
            savename_cross = os.path.join(self.savePath,'CC_CrossingClassLbel.p')
        else:
            # savenameT = os.path.join(self.savePath,'final_vctime.p')
            # savenameX = os.path.join(self.savePath,'final_vcxtrj.p')
            # savenameY = os.path.join(self.savePath,'final_vcytrj.p')
            # savenameclusterSize = os.path.join(self.savePath,'final_clusterSize.p')
            # savenameT_consecutive = os.path.join(self.savePath,'final_vctime_consecutive_frame.p')
            # savename_cross = os.path.join(self.savePath,'CrossingClassLbel.p')
            ## too large to save all, still save in truncations
            savenameT = os.path.join(self.savePath,'final_vctime'+str(matidx).zfill(3)+'.p')
            savenameX = os.path.join(self.savePath,'final_vcxtrj'+str(matidx).zfill(3)+'.p')
            savenameY = os.path.join(self.savePath,'final_vcytrj'+str(matidx).zfill(3)+'.p')
            savenameclusterSize = os.path.join(self.savePath,'final_clusterSize'+str(matidx).zfill(3)+'.p')
            savenameT_consecutive = os.path.join(self.savePath,'final_vctime_consecutive_frame'+str(matidx).zfill(3)+'.p')
            savename_cross = os.path.join(self.savePath,'CrossingClassLbel'+str(matidx).zfill(3)+'.p')

        if isClustered: # is clustered
            save_vctime = {}
            save_vcxtrj = {}
            save_vcytrj = {}
            save_clusterSize = {}            
            clean_vctime = {key: value for key, value in self.vctime.items() if key not in self.notconnectedLabel and key!=-1}
            clean_vcxtrj = {key: value for key, value in self.vcxtrj.items() if key not in self.notconnectedLabel and key!=-1}
            clean_vcytrj = {key: value for key, value in self.vcytrj.items() if key not in self.notconnectedLabel and key!=-1}
            clean_clusterSize = {key: value for key, value in self.clusterSize.items() if key not in self.notconnectedLabel and key!=-1}
            
            clean_vctime2 = {}
            """is the same with clean_vctime"""
            # clean_vctime2 = {key: [min(value),max(value)] for key, value in vctime2.items() if key!=-1}
            clean_vctime2 = {key: value for key, value in self.vctime2.items() if key not in self.notconnectedLabel and key!=-1}
            pickle.dump( clean_vctime, open(savenameT,"wb"))
            pickle.dump( clean_vctime2, open(savenameT_consecutive,'wb'))
            pickle.dump(self.CrossingClassLbel, open(savename_cross,'wb'))
            pickle.dump( clean_vcxtrj, open(savenameX,"wb"))
            pickle.dump( clean_vcytrj, open(savenameY,"wb"))
            pickle.dump( clean_clusterSize, open(savenameclusterSize,"wb"))

# if __name__ == '__main__':
def dic_main(dataSource,VideoIndex):
    import DataPathclass 
    global DataPathobj
    DataPathobj = DataPathclass.DataPath(dataSource,VideoIndex)
    import parameterClass 
    global Parameterobj
    Parameterobj = parameterClass.parameter(dataSource,VideoIndex)
    global subSampRate
    # subSampRate = int(np.round(DataPathobj.cap.get(cv2.cv.CV_CAP_PROP_FPS)/Parameterobj.targetFPS))
    subSampRate = int(30.0/Parameterobj.targetFPS)


    existingFiles = sorted(glob.glob(DataPathobj.dicpath+'*final_vcxtrj*.p'))
    existingFileNames = []
    for jj in range(len(existingFiles)):
        existingFileNames.append(int(existingFiles[jj][-5:-2])) # files starts from 0
    
    if len(existingFileNames)>0:
        print "alredy processed from 0 to ", str( max(existingFileNames))
        start_frame_idx = max(existingFileNames)*3600+3600
        # print "processing", str(15)
        # start_frame_idx = (15-1)*3600+3600
    else:
        start_frame_idx = 0
    print "start_frame_idx: ",start_frame_idx

    trj2dicObj = trj2dic()
    VCobj = VirtualCenter()
    videoReadingObj = videoReading(DataPathobj.video,subSampRate)
    trj2dicObj.get_XYT_inDic(start_frame_idx,videoReadingObj,VCobj)







