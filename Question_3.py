# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:31:12 2021

@author: adallora
"""

"""
Example for PC in right corner

Setup right corner:
    Camera 1: Regular camera
    Camera 2: SH sensor
"""

from camera.ueye_camera import uEyeCamera
from pyueye import ueye

import numpy as np
import matplotlib.pyplot as plt

SH_Sensor_Index = 2
Camera_Index = 1

# Pupil Function
P = np.zeros((1024,1280))
for i in range(1024):
    for k in range(1280):
        if ((i-512)**2+(k-640)**2)<(400**2):
            P[i,k]=1
            

def grabframes(nframes, cameraIndex=0):
    with uEyeCamera(device_id=cameraIndex) as cam:
        cam.set_colormode(ueye.IS_CM_MONO8)#IS_CM_MONO8)
        w=1280
        h=1024
        cam.set_aoi(0,0, w, h)
        
        cam.alloc(buffer_count=10)
        cam.set_exposure(50)
        cam.capture_video(True)
    
        imgs = np.zeros((nframes,h,w),dtype=np.uint8)
        acquired=0
        # For some reason, the IDS cameras seem to be overexposed on the first frames (ignoring exposure time?). 
        # So best to discard some frames and then use the last one
        while acquired<nframes:
            frame = cam.grab_frame()
            if frame is not None:
                imgs[acquired]=frame
                acquired+=1
            
    
        cam.stop_video()
    
    return imgs

if __name__ == "__main__":
    from dm.thorlabs.dm import ThorlabsDM
    
    with ThorlabsDM() as dm:
        
    
        #Sharpness
        
        max_sharp = 10**10
        opt_set = 0
        n_iter = 25
        sharpness = np.zeros(shape=(n_iter,1))
        for t in range(n_iter):
            DM_set = np.random.uniform(-1,1,size=len(dm))
            dm.setActuators(DM_set)
            img=grabframes(5, Camera_Index)
            for i in range(1024):
                for k in range(1280):
                    sharpness[t] = sharpness[t] + img[-1,i,k]**2
            if sharpness[t] < max_sharp:
                max_sharp = sharpness[t]
                opt_set = DM_set
        dm.setActuators(opt_set)
        img=grabframes(5, Camera_Index)
        plt.figure()    
        plt.imshow(img[-1])
        plt.figure()
        plt.plot(np.linspace(0,n_iter,n_iter),sharpness)
        plt.title('Sharpness Optimization Trajectory')
