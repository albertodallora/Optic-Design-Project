# -*- coding: utf-8 -*-

"""
Example for PC in right corner

Setup right corner:
    Camera 1: Regular camera
    Camera 2: SH sensor
"""

from camera.ueye_camera import uEyeCamera
from pyueye import ueye
from System import Array,Double,Boolean,UInt32

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

SH_Sensor_Index = 2
Camera_Index = 1


def grabframes(nframes, cameraIndex=0):
    with uEyeCamera(device_id=cameraIndex) as cam:
        cam.set_colormode(ueye.IS_CM_MONO8)  # IS_CM_MONO8)
        w = 1280
        h = 1024
        cam.set_aoi(0, 0, w, h)

        cam.alloc(buffer_count=10)
        cam.set_exposure(50)
        cam.capture_video(True)

        imgs = np.zeros((nframes, h, w), dtype=np.uint8)
        acquired = 0
        # For some reason, the IDS cameras seem to be overexposed on the first frames (ignoring exposure time?). 
        # So best to discard some frames and then use the last one
        while acquired < nframes:
            frame = cam.grab_frame()
            if frame is not None:
                imgs[acquired] = frame
                acquired += 1

        cam.stop_video()

    return imgs


def sharpness_opt(init_set):
    # Sharpness Metric
    threshold = 0.2
    gain = 1
    step = 0.05
    sharpness = 0
    opt_sharp = 0
    min_obj = 0
    C_reg = 1
    C_pos = 1
    C_norm = 1
    opt_set = init_set

    # Optimization process
    while gain > threshold:
        DM_set = opt_set + np.random.uniform(-step, step, size=len(dm))  # Random walk
        dm.setActuators(DM_set)
        img = grabframes(5, Camera_Index)

        # Compute centroid and sharpness
        I, N = np.meshgrid(np.arange(1024), np.arange(1280))

        # Compute centroid
        x_intens = (I * img[-1]).sum()
        y_intens = (N * img[-1]).sum()
        intens_sum = img[-1].sum()
        centroid_x = x_intens / intens_sum
        centroid_y = y_intens / intens_sum
        positioning = np.sqrt(centroid_x-640 ** 2 + centroid_y-512 ** 2)

        #Compute Sharpness
        sharpness = (img[-1]**2).sum()

        obj = -C_norm * sharpness + C_reg * np.linalg.norm(DM_set) + C_pos * positioning
        # If the result has improved, store it
        if obj < min_obj:
            gain = obj - min_obj
            min_obj = obj
            opt_sharp = sharpness
            opt_set = DM_set

    return opt_set, opt_sharp


def half_width_opt(init_set):
    # Sharpness Metric
    threshold = 0.2
    gain = 1
    step = 0.02
    min_obj = 0
    num_sum = 0
    C_reg = 0.001
    C_pos = 0.01
    C_norm = 100
    opt_set = init_set

    # Optimization process
    while (gain) > threshold:
        DM_set = opt_set + np.random.uniform(-step, step, size=len(dm))  # Random walk
        dm.setActuators(DM_set)
        img = grabframes(3, Camera_Index)

        # Compute centroid
        x_intens = (I * img[-1]).sum()
        y_intens = (N * img[-1]).sum()
        intens_sum = img[-1].sum()
        centroid_x = x_intens / intens_sum
        centroid_y = y_intens / intens_sum
        positioning = np.sqrt(centroid_x-640 ** 2 + centroid_y-512 ** 2)

        #Compute half-width r
        r = np.sqrt((img[-1]*((I-centroid_x)**2+(N-centroid_y)**2)).sum()/intens_sum)

        obj = C_norm * r + C_reg * np.linalg.norm(DM_set) + C_pos * positioning
        # If the result has improved, store it
        if obj < min_obj:
            gain = obj - min_obj
            min_obj = obj
            opt_r = r
            opt_set = DM_set

    return opt_set, opt_r


def sharpness_edge_opt(init_set):

    # Sharpness Metric
    threshold = 0.2
    gain = 1
    step = 0.02
    min_obj = 0
    num_sum = 0
    C_reg = 1
    C_pos = 1
    C_norm = 1
    opt_set = init_set

    # Optimization process
    while (gain) > threshold:
        DM_set = opt_set + np.random.uniform(-step, step, size=len(dm))  # Random walk
        dm.setActuators(DM_set)
        img = grabframes(5, Camera_Index)

        # Compute centroid
        x_intens = (I * img[-1]).sum()
        y_intens = (N * img[-1]).sum()
        intens_sum = img[-1].sum()
        centroid_x = x_intens / intens_sum
        centroid_y = y_intens / intens_sum
        positioning = np.sqrt(centroid_x-640 ** 2 + centroid_y-512 ** 2)

        #Compute half-width S_edge
        S_edge = np.sqrt(((img[-1]-np.roll(img[-1],1,axis=0))**2+(img[-1]-np.roll(img[-1],1,axis=1))**2).sum()/intens_sum)

        obj = -C_norm * S_edge + C_reg * np.linalg.norm(DM_set) + C_pos * positioning
        # If the result has improved, store it
        if obj < min_obj:
            gain = obj - min_obj
            min_obj = obj
            opt_Sedge = S_edge
            opt_set = DM_set

    return opt_set, opt_Sedge


if __name__ == "__main__":
    
    from dm.thorlabs.dm import ThorlabsDM

    with ThorlabsDM() as dm:
            
        #Sharpness Optimization
        #opt_set, max_sharp = sharpness_opt(np.zeros(43))     #Static aberrations
        #opt_set, max_sharp = sharpness_opt(np.random.uniform(-1, 1, size=len(dm)))     #Random aberration
        #defocus_DM = np.zeros(shape=(43))
        #defocus_DM[40:43]=-1
        #opt_set, max_sharp = sharpness_opt(defocus_DM)     #Defocus aberration
    
        # Half-Width Optimization
        #opt_set, min_r = half_width_opt(np.zeros(shape=(43)))  # Static aberrations
        #opt_set, min_r = half_width_opt(np.random.uniform(-1, 1, size=len(dm)))     #Random aberration
        #opt_set, min_r = half_width.opt(np.zeros(shape=(43,1))[40:43]=-1)     #Defocus aberration
    
        # Sharpness-Edge Optimization
        opt_set, max_sharp_edge = sharpness_edge_opt(np.zeros(shape=(43)))  # Static aberrations
        #opt_set, max_sharp_edge = sharpness_edge_opt(np.random.uniform(-1, 1, size=len(dm)))     #Random aberration
        #opt_set, max_sharp_edge = sharpness_edge_opt(np.zeros(shape=(43,1))[40:43]=-1)     #Defocus aberration
    
        # Plotting
        dm.setActuators(opt_set)
        img = grabframes(5, Camera_Index)
        plt.figure()
        plt.imshow(img[-1])
        plt.title('Half-Width Random Walk Optimization')
        plt.figure()
        plt.plot(np.linspace(0,len(dm),len(dm)), opt_set)
        plt.title('Control Voltage Set')
