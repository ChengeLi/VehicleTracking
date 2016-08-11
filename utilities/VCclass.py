import numpy as np
class VirtualCenter(object):
    """calculate virtual center, and remove outlier"""
    def getVC(self,x,y):
        assert len(x)==len(y)
        # x = x[x!=0]
        # y = y[y!=0]        
        self.vcx = np.median(x)
        self.vcy = np.median(y)

    def getVC_robust(self, x,y):
        """more robust???"""
        assert len(x)==len(y)
        if len(x)<3:
            self.vcx = np.median(x)
            self.vcy = np.median(y)
        else:
            mx = np.mean(x)
            my = np.mean(y)
            sx = np.std(x)
            sy = np.std(y)

            idx = ((x-mx)<=sx)&((y-my)<=sy)
            self.vcx = np.median(x[idx])
            self.vcy = np.median(y[idx])

        """discard very big group???"""
        # if sx>20 or sy>20:
        # if sx>0.8*Parameterobj.nullDist_for_adj or sy>0.8*Parameterobj.nullDist_for_adj:
        #     vcx = np.nan
        #     vcy = np.nan
        #     pdb.set_trace()
        # else:
        #     # idx = ((x-mx)<2*sx)&((y-my)<2*sy)
        #     idx = ((x-mx)<=sx)&((y-my)<=sy)
        #     vcx = np.median(x[idx])
        #     vcy = np.median(y[idx])

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


