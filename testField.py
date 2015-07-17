




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

        for key, value in vcxtrj.iteritems():
            x_location = vcxtrj[key]
            y_location = vcytrj[key]
            for ii in range(size(value)):
                self.Trj.append([x_location[ii],y_location[ii]]) 
                self.Trj_with_ID.append([key,x_location[ii],y_location[ii]])








test_obj = pickle.load( open( "./mat/20150222_Mat/obj_pair.p", "rb" ) )


badkey3 = []
for key, val in test_obj.frame.iteritems():
    if not val==[]:
        if size(test_obj.xTrj[key])!= val[1]-val[0]+1:
            badkey3.append(key)
       
























