#!/usr/bin/env python
import numpy as np
import multiprocessing
import time
import glob as glob
import pdb
import fit_extrapolate
import subprocess
import sys
import trjcluster_func_SBS
import subspace_clustering_merge
import unify_label_func
import trj2dic
import getVehiclesPairs
import os
import shutil
from datetime import datetime

def sort_modified_time(search_dir):
    filetimes = []
    filetimes_raw = []
    # search_dir = "/mydir/"
    # remove anything from the list that is not a file (directories, symlinks)
    # thanks to J.F. Sebastion for pointing out that the requirement was a list 
    # of files (presumably not including directories)  
    files = filter(os.path.isfile, glob.glob(search_dir + "*"))
    files.sort(key=lambda x: os.path.getmtime(x))
    for x in files:
        temp = os.path.getmtime(x)
        filetimes_raw.append(temp)
        filetimes.append(datetime.fromtimestamp(temp).strftime('%Y-%m-%d %H:%M:%S')) 
    return filetimes,filetimes_raw,files


def check_if_old_generated(VideoIndex,folderName, oldFile):
    """some adj files are generated before september, potentially is buggy....!"""
    search_dir = '/media/My Book/DOT Video/'+folderName+'/adj/'
    filetimes,filetimes_raw,files = sort_modified_time(search_dir)
    # filetimes,filetimes_raw = sort_modified_time('/media/My Book/DOT Video/'+folderName+'/unifiedLabel/')

    print "------------------------------------------"
    print VideoIndex,folderName
    # print len(filetimes),len(currentAdj),filetimes[-1],filetimes_raw[-1]
    
    for file,time_raw,time in zip(files,filetimes_raw,filetimes):
        if time_raw<1470000444.26:#before sep
            try:
                os.mkdir(search_dir+'old')
            except:
                pass
            shutil.move(file, search_dir+'old/')
            print 'move',file , time,'into old folder'
            oldFile.append(VideoIndex)
        # print len(filetimes),len(currentAdj),filetimes[-1],filetimes_raw[-1]
    return oldFile



def check_whos_older(VideoIndex,folderName,Timewrong):
    """every file should be generated according to the processing order, if not, delete"""
    search_dir1 = '/media/My Book/DOT Video/'+folderName+'/klt/smooth/'
    search_dir2 = '/media/My Book/DOT Video/'+folderName+'/adj/'
    # search_dir2 = '/media/My Book/DOT Video/'+folderName+'/ssc/'
    filetimes1,filetimes_raw1,files1 = sort_modified_time(search_dir1)
    filetimes2,filetimes_raw2,files2 = sort_modified_time(search_dir2)

    print "------------------------------------------"
    print VideoIndex,folderName
    for time_raw1,file1,filetime1,time_raw2,file2,filetime2 in zip(filetimes_raw1,files1,filetimes1,filetimes_raw2,files2,filetimes2):
        if time_raw1>=time_raw2:
            print filetime1,filetime2
            # print '(re)move latter:', file1, file2
            
            # try:
            #     os.mkdir(search_dir2+'old')
            # except:
            #     pass
            # try:
            #     shutil.move(file2, search_dir2+'old/')
            # except:
            #     print 'delete',file2
            #     os.remove(file2)
            Timewrong.append(VideoIndex)
    

    """recovering"""
    # print "------------------------------------------"
    # print VideoIndex,folderName
    # for time_raw1,file1,time_raw2,file2 in zip(filetimes_raw1,files1,filetimes_raw2,files2):
    #     if time_raw1<=time_raw2:
    #         print '(re)cover latter:', file1
    #         try:
    #             shutil.move(file1, search_dir1+'../')
    #         except:
    #             print 'error recovering'
    return Timewrong


onLocal =[0, 1, 11, 12, 13, 14, 15, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]

def kltStep(dataSource,VideoIndex):
    dataSource = 'DoT'
    VideoIndex = VideoIndex
    execfile('klt_func.py')

def smoothStep(dataSource,VideoIndex):
    dataSource = 'DoT'
    VideoIndex = VideoIndex
    fit_extrapolate.fit_extrapolate_main(dataSource,VideoIndex)

def adjStep(dataSource,VideoIndex):
    dataSource = 'DoT'
    VideoIndex = VideoIndex
    trjcluster_func_SBS.trjcluster_func_SBS_main(dataSource,VideoIndex)

def sscStep(dataSource,VideoIndex):
    dataSource = 'DoT'
    VideoIndex = VideoIndex
    subspace_clustering_merge.ssc_main(dataSource, VideoIndex)


def unifyStep(dataSource,VideoIndex):
    dataSource = 'DoT'
    VideoIndex = VideoIndex
    unify_label_func.unify_main(dataSource,VideoIndex)

def dicStep(dataSource,VideoIndex):
    dataSource = 'DoT'
    VideoIndex = VideoIndex
    trj2dic.dic_main(dataSource, VideoIndex)


def pairStep(dataSource,VideoIndex,folderName):
    dataSource = 'DoT'
    VideoIndex = VideoIndex
    getVehiclesPairs.pair_main(dataSource, VideoIndex,folderName)

def stepsForOneVideo(VideoIndex,existingDirList):
    dataSource = 'DoT'
    filename = existingDirList[VideoIndex]  
    folderName = filename[-14:]
    print "*****************processing: ", folderName
    # if 2*len(glob.glob('../'+folderName+'/klt/*.mat'))<len(glob.glob('../'+folderName+'/incPCPmask/*.p'))-2:
    #     execfile('klt_func.py')
    # if len(glob.glob('../'+folderName+'/klt/smooth/*.mat'))<len(glob.glob('../'+folderName+'/klt/*.mat')):
    #     fit_extrapolate.fit_extrapolate_main(dataSource,VideoIndex)

    if len(glob.glob('../'+folderName+'/adj/*.mat'))<len(glob.glob('../'+folderName+'/klt/smooth/*.mat')):
        trjcluster_func_SBS.trjcluster_func_SBS_main(dataSource,VideoIndex)
    # if len(glob.glob('../'+folderName+'/adj/*.mat'))>0 and len(glob.glob('../'+folderName+'/ssc/*.mat'))==0:
    #     execfile('subspace_clustering_merge.py')
    # if len(glob.glob('../'+folderName+'/ssc/*.mat'))>0:
    #     execfile('unify_label_func.py')
    #     execfile('trj2dic.py')


if __name__ == '__main__':
    dataSource = 'DoT'
    existingDirList = sorted(glob.glob('/media/My Book/DOT Video/2015-06*'))
    notfinished = []
    error = []
    oldFile =[]
    Timewrong = []
    Timewrong_smooth_before_klt = [ 0,  1, 11, 12, 13, 14, 15, 31, 48, 49, 53, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 71]
    Timewrong_adj_before_smooth  = [48, 56, 57, 58, 59]

    smoothNotfinished =[] 
    adjNotfinished = []
    kltNotfinished = []
    sscNotfinished = []
    labelNotfinished = []
    dicNotfinished = []
    pairNotfinished = []
    error = []
    for VideoIndex in [68]:
        filename = existingDirList[VideoIndex]  
        folderName = filename[-14:]
        print folderName
        currentKLT = sorted(glob.glob('/media/My Book/DOT Video/'+folderName+'/klt/*.mat'))        
        currentSmooth = sorted(glob.glob('/media/My Book/DOT Video/'+folderName+'/klt/smooth/*.mat'))        
        currentAdj = sorted(glob.glob('/media/My Book/DOT Video/'+folderName+'/adj/*.mat'))
        currentExt = sorted(glob.glob('/media/My Book/DOT Video/'+folderName+'/adj/*extreme*'))
        currentDiff = sorted(glob.glob('/media/My Book/DOT Video/'+folderName+'/adj/*feature_diff*'))

        currentSSC = sorted(glob.glob('/media/My Book/DOT Video/'+folderName+'/ssc/*.mat'))

        """KLT finished on local"""
        # if 2*len(currentKLT)<len(glob.glob('/media/My Book/DOT Video/'+folderName+'/incPCPmask/*.p'))-2:
        #     # p = multiprocessing.Process(target=kltStep,args=(dataSource,VideoIndex,))
        #     # p.start()
        #     print 'klt not finished',folderName
        #     print 0.5*len(glob.glob('/media/My Book/DOT Video/'+folderName+'/incPCPmask/*.p'))-len(currentKLT)
        #     notfinished.append(VideoIndex)
        # elif 2*len(currentKLT)>len(glob.glob('/media/My Book/DOT Video/'+folderName+'/incPCPmask/*.p')):
        #     print "klt more!!!???", folderName
        #     # try:
        #     #     os.mkdir('/media/My Book/DOT Video/'+folderName+'/klt/old')
        #     # except:
        #     #     pass
        #     # for tempFile in glob.glob('/media/My Book/DOT Video/'+folderName+'/klt/*.mat'):
        #     #     shutil.move(tempFile, '/media/My Book/DOT Video/'+folderName+'/klt/old/')
        # else:
        #     print 'finished', folderName
        
        """smooth finished on local"""
        # if len(currentSmooth)<len(currentKLT) or (len(currentSmooth)==0):
        #     # p = multiprocessing.Process(target=smoothStep,args=(dataSource,VideoIndex,))
        #     # p.start()
        #     print 'smooth not finished',folderName
        #     notfinished.append(VideoIndex)
        # elif len(currentSmooth)==len(currentKLT):
        #     print 'finished',folderName
        # else:
        #     print 'error', folderName

        """adj finished on local"""
        # # if len(currentExt)>len(currentAdj): 
        # #     print 'need to delete the last 2 pickle files, not completly saved.', folderName
        # #     # os.remove(currentExt[-1])
        # #     # print  len(currentExt)-len(currentAdj),folderName
        # # if len(currentDiff)>len(currentAdj):
        # #     print 'need to delete the last 2 pickle files, not completly saved.', folderName
        # #     # os.remove(currentDiff[-1])
        # #     # print  len(currentDiff)-len(currentAdj),folderName

        # if len(currentAdj)<len(currentSmooth) or (len(currentAdj)==0):
        #     p = multiprocessing.Process(target=adjStep,args=(dataSource,VideoIndex,))
        #     p.start()
        #     # print 'adj not finished',folderName, len(currentSmooth)-len(currentAdj)
        #     # notfinished.append(VideoIndex)
        # elif len(currentAdj)==len(currentSmooth) and (len(currentAdj)!=0):
        #     print 'donnot assign process to: ', folderName ,'already processed ================'
        # else:
        #     print 'error', folderName
        #     # try:
        #     #     os.mkdir('/media/My Book/DOT Video/'+folderName+'/adj/old')
        #     # except:
        #     #     pass
        #     # for tempFile in glob.glob('/media/My Book/DOT Video/'+folderName+'/adj/*.mat'):
        #     #     shutil.move(tempFile, '/media/My Book/DOT Video/'+folderName+'/adj/old/')

        """ssc not finished on local"""
        # if len(currentSSC)<len(currentAdj):
        #     # p = multiprocessing.Process(target=sscStep,args=(dataSource,VideoIndex,))
        #     # p.start()
        #     print 'ssc not finished',folderName, len(currentAdj)-len(currentSSC)
        #     notfinished.append(VideoIndex)
        # elif len(currentAdj)==len(currentSSC):
        #     print 'donnot assign process to: ', folderName ,'already processed ================'
        # else:
        #     print 'error',folderName
        #     # error.append(folderName)
        #     # try:
        #     #     os.mkdir('/media/My Book/DOT Video/'+folderName+'/ssc/old')
        #     # except:
        #     #     pass
        #     # for tempFile in glob.glob('/media/My Book/DOT Video/'+folderName+'/ssc/*.mat'):
        #     #     shutil.move(tempFile, '/media/My Book/DOT Video/'+folderName+'/ssc/old/')


        """unify labels"""
        # if len(glob.glob('/media/My Book/DOT Video/'+folderName+'/unifiedLabel/*.mat'))==0 :
        #     # p = multiprocessing.Process(target=unifyStep,args=(dataSource,VideoIndex,))
        #     # p.start()
        #     print 'unifying label not finished',VideoIndex,folderName
        #     notfinished.append(VideoIndex)    
        # else:
        #     print 'finished',VideoIndex,folderName
        #     # try:
        #     #     os.mkdir('/media/My Book/DOT Video/'+folderName+'/unifiedLabel/old')
        #     # except:
        #     #     pass
        #     # for tempFile in glob.glob('/media/My Book/DOT Video/'+folderName+'/unifiedLabel/*.mat'):
        #     #     shutil.move(tempFile, '/media/My Book/DOT Video/'+folderName+'/unifiedLabel/old/')


        """trj2dic"""
        currentDic = glob.glob('/media/My Book/DOT Video/'+folderName+'/dic/*final_vcxtrj*')
        
        # # for ii in range(len(currentDic)):
        # #     os.remove(currentDic[ii])

        if len(currentDic)<len(currentSSC):
            p = multiprocessing.Process(target=dicStep,args=(dataSource,VideoIndex,))
            p.start()
            # print 'trj2dic not finished',VideoIndex,folderName
            # notfinished.append(VideoIndex)
            # print 'stsill need ',len(currentSSC)-len(currentDic), len(currentDic)
        elif len(currentDic)==len(currentSSC):
            print "finished===!"
        else:
            print 'error!',folderName
            error.append(VideoIndex)

        """pair generation"""
        # currentPairCSV = glob.glob('/media/My Book/DOT Video/'+folderName+'/pair/*.csv')
        # currentPairPickle = glob.glob('/media/My Book/DOT Video/'+folderName+'/pair/*.p')
        # if len(glob.glob('/media/My Book/DOT Video/'+folderName+'/dic/*final_vcxtrj*.p')) == len(glob.glob('/media/My Book/DOT Video/'+folderName+'/ssc/*.mat')):
        #     if len(glob.glob('/media/My Book/DOT Video/'+folderName+'/pair/*.csv'))==2:
        #         print 'finished paring===', folderName
        #     else:
        #         # p = multiprocessing.Process(target=pairStep,args=(dataSource,VideoIndex,folderName,))
        #         # p.start()
        #         print 'notfinished',folderName
        #         notfinished.append(VideoIndex)
        #     # print '--------------------'
        #     # currentPairCSV_old = glob.glob('/media/My Book/DOT Video/'+folderName+'/pair/old/*.csv')
        #     # currentPairPickle_old = glob.glob('/media/My Book/DOT Video/'+folderName+'/pair/old/*.p')  
        #     # print os.path.getsize(currentPairCSV[0])/1024.0/1024.0, os.path.getsize(currentPairCSV[1])/1024.0/1024.0
        #     # print os.path.getsize(currentPairPickle[0])/1024.0/1024.0, os.path.getsize(currentPairPickle[1])/1024.0/1024.0
        # else:
        #     dicNotfinished.append(VideoIndex)


        """check if old files"""
        # oldFile = check_if_old_generated(VideoIndex,folderName, oldFile)
        # Timewrong = check_whos_older(VideoIndex,folderName,Timewrong)

        """rename all pair file into a wholeFolder"""
        # print 'change',currentPairCSV[0] ,'to', currentPairCSV[0][:-4]+'_'+folderName+'.csv'
        # os.rename(currentPairCSV[0], currentPairCSV[0][:-4]+'_'+folderName+'.csv')
        # os.rename(currentPairCSV[1], currentPairCSV[1][:-4]+'_'+folderName+'.csv')
         
        # shutil.copy(currentPairCSV[0], '/media/My Book/DOT Video/allPairs/')
        # shutil.copy(currentPairCSV[1], '/media/My Book/DOT Video/allPairs/')


    # Timewrong = np.unique(Timewrong)