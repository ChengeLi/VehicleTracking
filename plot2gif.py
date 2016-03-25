import matplotlib.pyplot as plt
import numpy as np
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
import pdb
import cv2

# DRAW A FIGURE WITH MATPLOTLIB

# duration = 2

# fig_mpl, ax = plt.subplots(1,figsize=(5,3), facecolor='white')
# xx = np.linspace(-2,2,200) # the x vector
# zz = lambda d: np.sinc(xx**2)+np.sin(xx+d) # the (changing) z vector
# ax.set_title("Elevation in y=0")
# ax.set_ylim(-1.5,2.5)
# line, = ax.plot(xx, zz(0), lw=3)

filenameList = sorted(glob.glob('/home/chengeli/Downloads/vis/vis/*.jpg'))
duration = len(filenameList)
fig_mpl,ax = plt.subplots(1)#,figsize=(im.height, im.width))

def make_frame_mpl(t):
    # line.set_ydata( zz(2*np.pi*t/duration))  # <= Update the curve
	# im = Image.open(filenameList[int(t)])
	im = cv2.imread(filenameList[int(t)])
	print int(t)
	# original_duration = im.info['duration']
	# frames = [frame.copy() for frame in ImageSequence.Iterator(im)]    
	# plt.imshow(im)
	plt.imshow(im)
	return mplfig_to_npimage(fig_mpl) # RGB image of the figure



# ANIMATE WITH MOVIEPY (UPDATE THE CURVE FOR EACH t). MAKE A GIF.
# import glob as glob
# from PIL import Image, ImageSequence

# filename = sys.argv[1]

fps = 50
animation =mpy.VideoClip(make_frame_mpl, duration=100)
pdb.set_trace()
# animation =mpy.VideoClip(mplfig_to_npimage(fig),duration=duration)
animation.write_gif("~/Desktop/NGSIMvis.gif", fps=fps)


# import numpy as np
# import matplotlib.pyplot as plt
# from moviepy.video.io.bindings import mplfig_to_npimage
# import moviepy.editor as mpy

# fig = plt.figure(facecolor="white") # <- ADDED FACECOLOR FOR WHITE BACKGROUND
# ax = plt.axes()
# x = np.random.randn(10, 1)
# y = np.random.randn(10, 1)
# p = plt.plot(x, y, 'ko')
# time = np.arange(2341973, 2342373)

# last_i = None
# last_frame = None

# def animate(t):
#     global last_i, last_frame

#     i = int(t)
#     if i == last_i:
#         return last_frame

#     xn = x + np.sin(2 * np.pi * time[i] / 10.0)
#     yn = y + np.cos(2 * np.pi * time[i] / 8.0)
#     p[0].set_data(xn, yn)

#     last_i = i
#     last_frame = mplfig_to_npimage(fig)
#     return last_frame

# duration = len(time)
# fps = 15
# animation = mpy.VideoClip(animate, duration=duration)
# animation.write_videofile("test.mp4", fps=fps)




