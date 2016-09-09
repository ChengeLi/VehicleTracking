import glob as glob

dataSource = 'DoT'
# videoList = sorted(glob.glob('/media/My Book/DOT Video/Canal@Baxter_avi/*.avi'))
videoList = sorted(glob.glob('/media/My Book/DOT Video/Canal@Baxter/*.asf'))

# for ii in range(51,len(videoList)):
for ii in range(32,37):
    filename = videoList[ii]
    print filename
    VideoIndex = ii
    execfile('foreground_blob.py')







