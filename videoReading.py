import cv2

""" video reading class"""
class videoReading(object):
    def __init__(self, videoName, subSampRate):
        """general class used for reading videos"""
        """mainly because the video converstion problem and existing opencv bugs"""
        self.cap = cv2.VideoCapture(videoName)
        self.subSampRate = subSampRate

    def reset(self,videoName):
        self.cap = cv2.VideoCapture(videoName) #reset to the 0 index


    def getFrame_loopy(self):
        """when read video in a loop, every subSampRate frames"""
        status, frame = self.cap.read()  
        for ii in range(self.subSampRate-1):
            status, frameskip = self.cap.read()
        return frame

    def sanityCheck(self):
        """read through all frame index"""
        """may not be consistent with the opencv FramNum"""
        ii =0
        st = True
        while st:
            st, vidFrame = self.cap.read()
            cv2.imshow('',vidFrame)
            ii=ii+1;
        FrameNum_loop = ii
        FrameNum = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        print 'FrameNum_loop',FrameNum_loop,'FrameNum',FrameNum
        print 'fps',int(np.round(cap.get(cv2.cv.CV_CAP_PROP_FPS)))







