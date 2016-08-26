import cv2
import glob as glob

imageList = sorted(glob.glob('/Volumes/Transcend/US-101/US-101-RawVideo-0750am-0805am-Cam1234/camera4/visualization/rawKLTresult/*.jpg'))
firstImg = cv2.imread(imageList[0])
height , width , layers =  firstImg.shape

fourcc = cv2.cv.CV_FOURCC('M','P','4','2')
fps = 10
video = cv2.VideoWriter('~/Desktop/KLTvideo.mp4',fourcc,fps,(width,height))

for ii in range(len(imageList)):
	img = cv2.imread(imageList[ii])
	video.write(img)

cv2.destroyAllWindows()
video.release()