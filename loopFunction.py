import glob as glob

dataSource = 'DoT'
# videoList = sorted(glob.glob('/media/My Book/DOT Video/Canal@Baxter_avi/*.avi'))
# videoList = sorted(glob.glob('/media/My Book/DOT Video/Canal@Baxter/*.asf'))

# for ii in range(51,len(videoList)):
#     filename = videoList[ii]
#     print filename
#     VideoIndex = ii
#     execfile('foreground_blob.py')



existingDirList = sorted(glob.glob('/scratch/cl2840/CUSP/2015-06*'))
for ii in range(len(existingDirList)):  
    VideoIndex = ii
    execfile('klt_func.py')




    