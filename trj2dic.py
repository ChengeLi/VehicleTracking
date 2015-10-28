# ============================================================
# This function ransforms trj representation into dictionary formats
# to make constructing trj object more conveniently  TrjObj()

# ==============fix me ...==...
import pdb
import numpy as np
import glob as glob
from scipy.io import loadmat,savemat
from scipy.sparse import csr_matrix
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




def get_XYT_inDic(trunkTrjFile, mlabels,frame_idx, isClustered = True, trunclen = 600):  # get the time dictionary, such as vctime
    # applicable to both clustered and non-clustered trj datas
    # If non-clustered trjs, the input mlabels are just the trj ID (mask)


    # trunkTrjFile = loadmat(matfiles[frame_idx//trunclen])
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

    # initialization
    vcxtrj = {} ##dictionary
    vcytrj = {}
    vctime = {}

    for i in np.unique(mlabels): 
        vcxtrj[i]=[]
        vcytrj[i]=[]
        vctime[i]=[]

    startT = np.ones([Nsample,1])*-999
    endT = np.ones([Nsample,1])*-999

    for i in range(Nsample):  # for the ith sample## get the time T (where the pt appears and disappears)
        havePt  = np.array(np.where(xtrj[i,:]>0))[0]
        if len(havePt)!=0:
            startT[i] = int ( min(havePt)+(frame_idx/trunclen*trunclen) )
            endT[i]   = int ( max(havePt)+(frame_idx/trunclen*trunclen) )
            # only for check, can delete ttrj
            if startT[i]!=np.min(ttrj[i,havePt]) or endT[i]!=np.max(ttrj[i,havePt]):
                print "wrong time===========!!!, want to delete this trj?"
                pdb.set_trace()
                
                
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
                    print frame_idx
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

    return vcxtrj, vcytrj, vctime, trk



if __name__ == '__main__':
    matfiles = sorted(glob.glob('../DoT/CanalSt@BaxterSt-96.106/adj/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/len50' +'*.mat'))
    frame_idx = 0
    trunclen = 600
    isClustered = True
    trunkTrjFile = loadmat(matfiles[frame_idx//trunclen])

    if isClustered:
        lrsl = '../DoT/CanalSt@BaxterSt-96.106/finalresult/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/priorssc5030' 
        mask = loadmat(lrsl)['mask'][0] # labeled trjs' indexes
        labels = loadmat(lrsl)['label'][0]

        lasttrunkTrjFile = loadmat(matfiles[-1]) ##the last one, to get the max index number
        # IDintrunklast = lasttrunkTrjFile['idxtable'][0]  #last trunk (3rd mat) has 3603 trjs #use original
        IDintrunklast = lasttrunkTrjFile['mask'][0] # 26 trjs 
        mlabels = np.ones(max(IDintrunklast)+1)*-1  #initial to be -1
    
        #build PtsInCurFrm trj labels (-1 : not interest PtsInCurFrm)
        for idx,ID in enumerate(mask):  # i=mask[idx], the content
            mlabels[int(ID)] = labels[int(idx)]
    else:
        IDintrunk = trunkTrjFile['mask'][0]
        mlabels = IDintrunk

    
    vcxtrj, vcytrj, vctime, trk = get_XYT_inDic(trunkTrjFile, mlabels, frame_idx, isClustered = True)

