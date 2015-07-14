from scipy.io import loadmat,savemat
import cv2,pickle,pdb,glob
from scipy.sparse import csr_matrix

# video_src = '/home/andyc/Videos/video0222.mp4'
video_src = '../VideoData/video0222.mp4'
trunclen = 600

inifilename = 'HR'

lrsl = './mat/20150222_Mat/finalresult/'+inifilename
mask = loadmat(lrsl)['mask'][0]
labels = loadmat(lrsl)['label'][0]
matfiles = sorted(glob.glob('./mat/20150222_Mat/'+inifilename+'*.mat'))
ptstrj = loadmat(matfiles[-1]) ##the last one, to get the max label number
ptsidx = ptstrj['idxtable'][0]



mlabels = np.ones(max(ptsidx)+1)*-1
#build pts trj labels (-1 : not interest pts)
for idx,i in enumerate(mask):  # i=mask[idx], the cotent
    mlabels[i] = labels[idx]

vcxtrj = {} ##dictionary
vcytrj = {}

vctime = {}

for i in np.unique(mlabels):
    vcxtrj[i]=[]
    vcytrj[i]=[]
    vctime[i]=[]

cam = cv2.VideoCapture(video_src)
nrows = cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
ncols = cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
framenum  = int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

framerate = int (cam.get(cv2.cv.CV_CAP_PROP_FPS))
frame_idx = 0

plt.figure(1,figsize=[10,12])
axL = plt.subplot(1,1,1)
frame = np.zeros([nrows,ncols,3]).astype('uint8')
im = plt.imshow(np.zeros([nrows,ncols,3]))
axis('off')
color = array([random.randint(0,255) \
               for _ in range(3*int(max(labels)))])\
               .reshape(max(labels),3)

cam.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,frame_idx)

def Virctr(x,y):
    '''
    calculate virtual center, and remove out lier
    '''    
    if len(x)<3:
        vcx = mean(x)
        vcy = mean(y)
    else:
        mx = np.mean(x)
        my = np.mean(y)
        sx = np.std(x)
        sy = np.std(y)
        idx = ((x-mx)<2*sx)&((y-my)<2*sy)
        vcx = np.mean(x[idx])
        vcy = np.mean(y[idx])
    return vcx,vcy


while (frame_idx < 3000): #framenum):

    if (frame_idx % trunclen == 0):
        
        ptstrj = loadmat(matfiles[frame_idx//trunclen])
        xtrj = csr_matrix(ptstrj['xtracks'], shape=ptstrj['xtracks'].shape).toarray()
        ytrj = csr_matrix(ptstrj['ytracks'], shape=ptstrj['ytracks'].shape).toarray()
        ptsidx = ptstrj['idxtable'][0]
        sample = ptstrj['xtracks'].shape[0]
        fnum   = ptstrj['xtracks'].shape[1]

        trk = np.zeros([sample,fnum,3])
        startT = np.ones([sample,1])*-999
        endT = np.ones([sample,1])*-999

        for i in range(sample):
            trk[i,:,0] = xtrj[i]
            trk[i,:,1] = ytrj[i]
            # trk[i,:,2] = arange(fnum)


            ## get the time T (where the pt appears and disappears)
            
            havePt  = find(xtrj[i,:]>0)
            if havePt!=[]:
                startT[i] = min(havePt)+(frame_idx/trunclen*trunclen) 
                endT[i]   = max(havePt)+(frame_idx/trunclen*trunclen) 
   




    print frame_idx
    ret, frame[:] = cam.read()
    # plt.draw()
    pts = trk[:,:,0].T[frame_idx%trunclen]!=0 # pts appear in this frame,i.e. X!=0


    # pdb.set_trace()

    labinf = list(set(mlabels[ptsidx[pts]])) # label in current frame
    dots = []
    for k in labinf:
        if k !=-1:
            x = trk[:,:,0].T[frame_idx%trunclen][((mlabels==k)[ptsidx])&pts]
            y = trk[:,:,1].T[frame_idx%trunclen][((mlabels==k)[ptsidx])&pts]
            vx,vy = Virctr(x,y) # find virtual center
            
            vcxtrj[k].append(vx) 
            vcytrj[k].append(vy)

            t1=startT[((mlabels==k)[ptsidx])&pts]
            t2=endT[((mlabels==k)[ptsidx])&pts]

            t1=min(t1)
            t2=max(t2)


            vctime[k]=[int(t1),int(t2)]

            #lines = axL.plot(vcxtrj[k],vcytrj[k],color = (0,1,0),linewidth=2)
            lines = axL.plot(vcxtrj[k],vcytrj[k],color = (color[k-1].T)/255.,linewidth=2)
            line_exist = 1
            #dots.append(axL.scatter(vx, vy, s=50, color=(color[k-1].T)/255.,edgecolor='black')) 
            # dots.append(axL.scatter(x, y, s=50, color=(color[k-1].T)/255.,edgecolor='none')) 
            #dots.append(axL.scatter(x, y, s=50, color=(1,0,0),edgecolor='none'))
                                         
    im.set_data(frame[:,:,::-1])
    # plt.draw()
    # plt.show()



    # name = '/home/andyc/image/AIG/HR/'+str(frame_idx).zfill(6)+'.jpg'
    # name = '../Image/'+str(frame_idx).zfill(6)+'.jpg'

    # savefig(name) ##save figure
   
    
    while line_exist :
        try:
            axL.lines.pop(0)
        except:
            line_exist = 0
    plt.show()
    

    for i in dots:
        i.remove()
    plt.show()

    frame_idx = frame_idx+1
  
    #pdb.set_trace()







# save the tracks


# savename = './mat/'+'vcxtrj'  ## mat not working??
# savemat(savename,vcxtrj)

# savename = './mat/'+'vcytrj.mat'
# savemat(savename,vcytrj)




import csv
writer = csv.writer(open('./mat/20150222_Mat/vcxtrj.csv', 'wb'))
for key, value in vcxtrj.items():
   writer.writerow([key, value])

writer = csv.writer(open('./mat/20150222_Mat/vcytrj.csv', 'wb'))
for key, value in vcytrj.items():
   writer.writerow([key, value])

writer = csv.writer(open('./mat/20150222_Mat/vctime.csv', 'wb'))
for key, value in vctime.items():
   writer.writerow([key,value])



import cPickle as pickle
import pprint

 # Save a dictionary into a pickle file.
pickle.dump( vctime, open( "./mat/20150222_Mat/vctime.p", "wb" ) )
pickle.dump( vcxtrj, open( "./mat/20150222_Mat/vcxtrj.p", "wb" ) )
pickle.dump( vcytrj, open( "./mat/20150222_Mat/vcytrj.p", "wb" ) )



# load and check
test_vctime = pickle.load( open( "./mat/20150222_Mat/tracksPickleBKUP/vctime.p", "rb" ) )
print test_vctime == vctime  # true

test_vcxtrj = pickle.load( open( "./mat/20150222_Mat/tracksPickleBKUP/vcxtrj.p", "rb" ) )
print test_vcxtrj == vcxtrj   

test_vcytrj = pickle.load( open( "./mat/20150222_Mat/tracksPickleBKUP/vcytrj.p", "rb" ) )
print test_vcytrj == vcytrj  

# for key, value in test_vcxtrj.iteritems():
#     pprint.pprint(key)
#     print vcxtrj[key] == test_vcxtrj[key]




# consider the pair-wise relationship between each two cars
class Objects():
    def __init__(self,vcxtrj,vcytrj,vctime):
        self.ptsTrj= {}
        self.pts = []
        self.Trj = [] #[x,y]
        self.Trj_with_ID = [] # [ID,x,y]
        self.xTrj = vcxtrj # x
        self.yTrj = vcytrj  #y
        self.frame = vctime #current frm number
        self.vel = [] 
        self.pos = [] 
        self.status = 1   # 1: alive  2: dead
        self.globalID = sorted(vctime.keys())
        self.dir = {} # directions 0 or 1
        self.bad_IDs = []


        for key, val in vctime.iteritems():
            if val ==[] or val[1]-val[0] <= 5*3:
                self.bad_IDs.append(key)



        for key, value in vcxtrj.iteritems():
            x_location = vcxtrj[key]
            y_location = vcytrj[key]
            for ii in range(size(value)):
                self.Trj.append([x_location[ii],y_location[ii]]) 
                self.Trj_with_ID.append([key,x_location[ii],y_location[ii]])

        for key in vctime.iterkeys():
            if abs(((np.asarray(self.yTrj[key][1:])-np.asarray(self.yTrj[key][:-1]))>=0).sum() - (size(self.yTrj[key])-1))<=5:
                self.dir[key] = 1
            elif abs(((np.asarray(self.yTrj[key][1:])-np.asarray(self.yTrj[key][:-1]))<=0).sum() - (size(self.yTrj[key])-1))<=5:
                self.dir[key] = 0
            else: 
                self.dir[key] = 999
                self.bad_IDs.append(key)



obj_pair = Objects(test_vcxtrj,test_vcytrj,test_vctime)
test_vctime = {key: value for key, value in test_vctime.items() 
             if key not in obj_pair.bad_IDs}
test_vcxtrj = {key: value for key, value in test_vcxtrj.items() 
             if key not in obj_pair.bad_IDs}
test_vcytrj = {key: value for key, value in test_vcytrj.items() 
             if key not in obj_pair.bad_IDs}
# rebuild this object using filtered data, should be no bad_IDs
obj_pair = Objects(test_vcxtrj,test_vcytrj,test_vctime)
print obj_pair.bad_IDs == []


# pickle.dump( obj_pair, open( "./mat/20150222_Mat/obj_pair.p", "wb" ) )
# test_obj = pickle.load(open("./mat/20150222_Mat/obj_pair.p", "rb" ))






chunk_len = framerate*40 # 40s
chunk_center = range(1/2*chunk_len,3000,chunk_len)  #change 1000 to the Framenumber
chunk_center = chunk_center [1:]

# temp_vcxtrj = {}
# temp_vcytrj = {}
# temp_vctime = {}
# for ii in obj_pair.globalID[0:400]:  
#     temp_vcxtrj[ii] = test_vcxtrj[ii]
#     temp_vcytrj[ii] = test_vcytrj[ii]
#     temp_vctime[ii] = test_vctime[ii]

potential_key = []

for ii in range(size(chunk_center,0)):


    for key, value in temp_vctime.iteritems():
        if value!=[]:
            startF = value[0]
            endF = value[1]
            if not(startF >= chunk_center[ii]+1/2*chunk_len or endF < chunk_center[ii]-1/2*chunk_len):
                potential_key.append(int(key))


potential_x = []
potential_y = []

set_x = []
set_y = []
set_frm = []
# get intersection for those IDs
for kk in range(len(potential_key)):
    potential_frm = temp_vctime[potential_key[kk]]
    set_frm.append(set(range(potential_frm[0],potential_frm[1],1)))


    potential_x = temp_vcxtrj[potential_key[kk]]
    set_x.append(set(potential_x))

    potential_y = temp_vcytrj[potential_key[kk]]
    set_y.append(set(potential_y))


parallel_vehicle_ID = []
parallel_vehicle_common = []
parallel_vehicle_x1 = []
parallel_vehicle_x2 = []
parallel_vehicle_y1 = []
parallel_vehicle_y2 = []



for ff in range (size(set_frm)-1):
    z = set_frm[ff]
    for ff2 in range(ff+1, size(set_frm)):
        common_frm = z.intersection(set_frm[ff2])
        if len(common_frm) >= 10:
            common_frm = list(common_frm)
            parallel_vehicle_ID.append([ff, ff2])
            parallel_vehicle_common.append(common_frm)

            parallel_vehicle_x1.append(vcxtrj[ff][range(common_frm[0]-vctime[ff][0],common_frm[-1]-vctime[ff][0],1)])
            parallel_vehicle_x2.append(vcxtrj[ff2][range(common_frm[0]-vctime[ff2][0],common_frm[-1]-vctime[ff2][0],1)])

            parallel_vehicle_y1.append(vcytrj[ff][range(common_frm[0]-vctime[ff][0],common_frm[-1]-vctime[ff][0],1)])
            parallel_vehicle_y2.append(vcytrj[ff2][range(common_frm[0]-vctime[ff2][0],common_frm[-1]-vctime[ff2][0],1)])

 #  still need to debug here!!







# 2. Filter out thoes intersect too short


if common_frm.size








# 3. save





# tracks2015 = pickle.load( open ("/home/chengeli/Downloads/tracks2015.pkl","rb")) 
 





    














