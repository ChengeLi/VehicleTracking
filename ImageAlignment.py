import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('../DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/00000601.jpg')
img = np.asarray(img)
rows,cols,ch = img.shape

# img_small = cv2.resize(img,(cols/2,rows/2))  # the resize function is x first, y after

# pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
# pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])


# pts1 = np.float32([[328, 431],[483,435], [26,615],[797,621]])  # beatles 11.jpg
# pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])


pts1 = np.float32([[161, 147],[224,147], [73,519],[399,397]])  # canal st, left lane
# pts1 = np.float32([[229, 164],[340,151], [451,444],[703,331]])  # canal st, right lane

# pts2 = np.float32([[0,0],[350,0],[0,305],[350,305]])
# pts2 = np.float32([[0,0],[350,0],[0,600],[350,600]])
pts2 = np.float32([[0,0],[380,0],[0,380],[380,380]])

M = cv2.getPerspectiveTransform(pts1,pts2)
 
# dst = cv2.warpPerspective(img_small,M,(300,300))
dst = cv2.warpPerspective(img,M,(380,380))
 
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('left Lane Output')
plt.show()
# plt.savefig('./tempFigs/rightlane_per_trans.jpg')
# plt.savefig('./tempFigs/leftlane_per_trans.jpg')


'''
# 2 step way, (not so good)
pts1_step1 = np.float32([[161, 147],[224,147], [132,229],[271,220]])  # canal st
pts2_step1 = np.float32([[0,0],[350,0],[0,200],[350,200]]) 

pts1_step2 = np.float32([[132,229],[271,220],[76,513],[408,410]]) 
pts2_step2 = np.float32([[0,200],[350,200],[0,1000],[350,1000]]) 

M1 = cv2.getPerspectiveTransform(pts1_step1,pts2_step1)
M2 = cv2.getPerspectiveTransform(pts1_step2,pts2_step2)

# dst = cv2.warpPerspective(img_small,M,(300,300))
dst1 = cv2.warpPerspective(img,M1,(350,200))
dst2 = cv2.warpPerspective(img,M2,(350,1000))

plt.subplot(121),plt.imshow(dst1),plt.title('step1')
plt.subplot(122),plt.imshow(dst2),plt.title('step2')
plt.show()


'''



