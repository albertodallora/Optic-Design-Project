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
        
        print(f"Deformable m(irror with {len(dm)} actuators")
        #Actuator_matrix = np.zeros((43,43))
        #for k in range(len(dm)):
         #   Actuator_matrix[k,] = np.linspace(-1,1,num=len(dm))
            #Actuator_matrix = np.concatenate((Actuator_matrix,np.linspace(-1,1,num=len(dm))))
        act_vector = np.zeros(shape=(len(dm),1))
        act_vector[40:43] = np.array([[0, 0, 0]]).T
        dm.setActuators(act_vector)
        #dm.setActuators(np.linspace(1,-1,num=len(dm)))
        
        plt.figure()    
        img=grabframes(5, Camera_Index)
        plt.imshow(img[-1])
    
        plt.figure()
        img=grabframes(5, SH_Sensor_Index)
        plt.imshow(img[-1])
