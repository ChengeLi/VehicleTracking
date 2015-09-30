# change the unorderd names into a sequential name listing

from glob import glob
import cv2


# image_listing = glob('../VideoData/Video_20150222/*.jpg')
image_listing = glob('/home/chengeli/CUSP/park/detection_result/Sep7/Sep7_*.jpg')
new_name_listing=[]

# the header length : len("../VideoData/Video_20150222/fc2_save_2015-02-22-161719-")==55
# len("../VideoData/Video Data_20150220/02202015_test1_2015-02-20-153011-")==66
for ii in range(len(image_listing)):
	temp=image_listing[ii][52:-4]
	temp=temp.zfill(7)
	# new_name_listing.append('../VideoData/20150222/'+temp+'.jpg')
	new_name_listing.append('/home/chengeli/CUSP/park/detection_result/temp/'+temp+'.jpg')
	curFrm=cv2.imread(image_listing[ii])
	cv2.imwrite(new_name_listing[ii],curFrm)



















































