import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('./11.jpg')
img = np.asarray(img)
rows,cols,ch = img.shape

img_small = cv2.resize(img,(cols/2,rows/2))  # the resize function is x first, y after

# pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
# pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
pts1 = np.float32([[328, 431],[483,435], [26,615],[797,621]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1,pts2)
 
# dst = cv2.warpPerspective(img_small,M,(300,300))
dst = cv2.warpPerspective(img_small,M,(300,int(300*float(cols)/rows)))
 
plt.subplot(121),plt.imshow(img_small),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()




(328, 431)

(483,435)


(26,615)


(797,621)