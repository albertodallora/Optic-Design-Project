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

def cam_init(cam, w=1280,h=1024):
    cam.set_pixel_clock(10) # set the clock rate lower so we can have multiple cameras running without issue
    cam.set_colormode(ueye.IS_CM_MONO8)#IS_CM_MONO8)
    cam.set_aoi(0,0, w, h)
    
    cam.alloc(buffer_count=10)
    cam.set_exposure(0.1)
    cam.capture_video(True)




def grabframes(nframes, cam : uEyeCamera):

    imgs = np.zeros((nframes,cam.aoi_height(),cam.aoi_width()),dtype=np.uint8)
    acquired=0
    # For some reason, the IDS cameras seem to be overexposed on the first frames (ignoring exposure time?). 
    # So best to discard some frames and then use the last one
    while acquired<nframes:
        frame = cam.grab_frame()
        if frame is not None:
            imgs[acquired]=frame
            acquired+=1
        
    return imgs

if __name__ == "__main__":
    from dm.okotech.dm import OkoDM

    # Use "with" blocks so the hardware doesn't get locked when you press ctrl-c    
    with OkoDM(dmtype=0) as dm:
        print(f"Deformable mirror with {len(dm)} actuators")
        dm.setActuators(np.random.uniform(-1,1,size=len(dm)))

        with uEyeCamera(device_id=1) as cam, uEyeCamera(device_id=2) as cam2:
            cam_init(cam)
            cam_init(cam2)
            
            for f in range(10):
                a=grabframes(1,cam)[0]
                b=grabframes(1,cam2)[0]
                plt.figure()
                plt.imshow(np.concatenate([a,b],axis=-1))
                plt.title(f'frame {f}')
                
            cam.stop_video()
            cam2.stop_video()

            