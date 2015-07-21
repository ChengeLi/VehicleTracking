from scipy.io import loadmat,savemat
import pdb,glob
num = 0
atmp = []
flab = []  #final labels
fmask = [] #final mask
fTtracks = {}

inifilename = 'HR'
# matfiles = sorted(glob.glob('./mat/20150222_Mat/labels/'+inifilename+'*.mat'))
matfiles = sorted(glob.glob('./mat/20150222_Mat/labels/'+inifilename+'_label_withT_'+'*.mat'))

for matidx in range(len(matfiles)-1): 
    #pdb.set_trace() 
    if matidx == 0:
        L1 = loadmat(matfiles[matidx])['label'][0]
        M1 = loadmat(matfiles[matidx])['mask'][0]
        Ttracks1 = loadmat(matfiles[matidx])['Ttracks']
    else:
        L1 = L2
        M1 = M2
        Ttracks1 = Ttracks2

    L2 = loadmat(matfiles[matidx+1])['label'][0]
    M2 = loadmat(matfiles[matidx+1])['mask'][0]
    Ttracks2 = loadmat(matfiles[matidx+1])['Ttracks']

    labelnum = max(L1)
    L2[:] = L2 + labelnum+1 # to make sure there're no duplicate labels

    commonidx = np.intersect1d(M1,M2)  #trajectories existing in both 2 trucations
    
    print('L1 : {0}, L2 : {1} ,common term : {2}').format(len(np.unique(L1)),len(np.unique(L2)),len(commonidx))

    
    # tracks
    # final track fTracks[ID]=[startFrm, startFrm+1,.....,endFrm]
    for ii in np.union1d(M1, M2):
        track_pre = Ttracks1[ np.where(M1 == ii)]
        track_post = Ttracks2[ np.where(M2 == ii)]
        trackcombo = list(track_pre[np.where(track_pre>0)]) + list(track_post[np.where(track_post>0)])
        fTtracks[ii] = list(trackcombo)





    #  duplicate
    # for i in commonidx:
    #     print i 
    #     label1 = L1[np.where((M1 == i)!=0)[0][0]]
    #     print('\nlabel 1 : {0}').format(label1)
    #     idx1 = np.where(L1==label1)
    #     print('label 1 group : {0}').format(M1[idx1])
    #     label2 = L2[np.where((M2 == i)!=0)[0][0]]
    #     print('label 2 : {0}').format(label2)
    #     idx2 = np.where(L2==label2)
    #     print('label 2 group : {0}\n\n').format(M2[idx2])

    for i in commonidx:
        track_pre = Ttracks1[ np.where(M1 == i)]
        track_post = Ttracks2[ np.where(M2 == i)]

        track = list(track_pre[np.where(track_pre>0)]) + list(track_post[np.where(track_post>0)])


        if i not in atmp:
        
            label1 = L1[np.where((M1 == i)!=0)[0][0]]
            label2 = L2[np.where((M2 == i)!=0)[0][0]]
            idx1 = np.where(L1==label1)
            idx2 = np.where(L2==label2)
            tmp1 = np.intersect1d(M1[idx1],commonidx) # appear in both trunks and label is label1 
            tmp2 = np.intersect1d(M2[idx2],commonidx)

            L1idx =list(idx1[0])
            L2idx =list(idx2[0])
            # L1idx_TEST =list(idx1[0])
            # L2idx_TEST =list(idx2[0])


            diff1 = np.setdiff1d(tmp1,tmp2)  # this difference only accounts for elements in tmp1
            diff = np.union1d(np.setdiff1d(tmp1,np.intersect1d(tmp1,tmp2)) , np.setdiff1d(tmp2,np.intersect1d(tmp1,tmp2) ))
            # pdb.set_trace()

            atmp = np.unique(np.hstack((tmp1,tmp2)))


            # this chunk is useless, it doesn't change the L1 or L2 at all. 
            # In the end L1_TEST == L1. 
            # if len(diff)>0:
            #     for jj in diff:
            #         if jj in tmp1:  ## in the same group in trunk1 but different groups in trunk 2
            #             # pdb.set_trace()
            #             wlab = L1[np.where((M1 == jj )!=0)[0][0]] # wrong labels
            #             widx = np.where(L1==wlab) #idx of wrong labels
            #             L1[widx] = label1
            #             L1idx.append(np.where(M1 == jj )[0][0])
                        
            #         else: # jj  in tmp2:
            #             # pdb.set_trace()
            #             wlab = L2[np.where((M2 == jj )!=0)[0][0]]
            #             widx = np.where(L2==wlab)
            #             L2[widx] = label2
            #             L2idx.append(np.where(M2 == jj )[0][0])




            # L1_TEST = L1
            # L2_TEST = L2
            L1[L1idx] = label1  ## keep the first appearing label
            L2[L2idx] = label1   
            # L1_TEST[L1idx_TEST] = label1  
            # L2_TEST[L2idx_TEST] = label1 
            # pdb.set_trace()  
            # print sum(L1 == L1_TEST)== len(L1)
            # print sum(L2 == L2_TEST)== len(L2)


            # change the time frame 




 


    if matidx == 0 :
        flab[:] = flab + list(L1) 
        fmask[:] = fmask + list(M1)
        


    flab[:] = flab +list(L2)
    fmask[:] = fmask + list(M2)


    fTtracks.update(fTtracks)


#pdb.set_trace()
#== eliminate duplicate part == 
   
data = np.vstack((fmask,flab))
result = data[:,data[0].argsort()]
labels = list(result[1])
mask   = list(result[0])
dpidx = np.where(np.diff(sorted(mask)) == 0)[0]  #find duplicate ID in mask
    
for k in dpidx: #dpidx[::-1]:
    labels.pop(k)
    mask.pop(k)
 
result = {}
result['label']   = labels
result['mask']= mask
# result['Ttracks'] = fTtracks


savename = './mat/20150222_Mat/finalresult/'+inifilename
savemat(savename,result)

# why not working???!!!!
# savename2 = './mat/20150222_Mat/finalresult/'+inifilename+'Ttracks'
# result2 = {}
# result2['Ttracks'] = fTtracks
# savemat(savename2,result2)


pickle.dump( fTtracks, open( "./mat/20150222_Mat/finalresult/HRTtracks.p", "wb" ) )















