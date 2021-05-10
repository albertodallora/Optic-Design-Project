from camera.ueye_camera import uEyeCamera
from pyueye import ueye

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

def SM_opt(init_set):
    from dm.thorlabs.dm import ThorlabsDM

    with ThorlabsDM() as dm:
        # Sharpness Metric
        threshold = 0.2
        gain = 1
        step = 0.02
        min_obj = 0
        SM = 0
        C_reg = 1
        C_pos = 1
        C_norm = 1
        opt_set = init_set

        # Optimization process
        while (gain) > threshold:
            DM_set = opt_set + np.random.uniform(-step, step, size=len(dm))  # Random walk
            dm.setActuators(DM_set)
            img = grabframes(5, Camera_Index)
            x_intens = 0
            y_intens = 0
            # Compute centroid
            intens_sum = np.matrix.sum(img[-1, i, k])
            for i in range(1024):
                for k in range(1280):
                    x_intens += k * img[-1, i, k]
                    y_intens += i * img[-1, i, k]
            centroid_x = x_intens / intens_sum
            centroid_y = y_intens / intens_sum
            positioning = np.sqrt(centroid_x ** 2 + centroid_y ** 2)
            # Compute second-order metric
            for i in range(1024):
                for k in range(1280):
                    SM += img[-1, i, k] * ((k**2 + i**2))

            obj = C_norm * SM + C_reg * np.linalg.norm(DM_set) + C_pos * positioning
            # If the result has improved, store it
            if obj < min_obj:
                gain = obj - min_obj
                min_obj = obj
                opt_SM = SM
                opt_set = DM_set

        return opt_set, opt_SM



if __name__ == "__main__":
    from dm.thorlabs.dm import ThorlabsDM

    with ThorlabsDM() as dm:
        #Sharpness Optimization
        opt_set, max_sharp = sharpness_opt(np.zeros(shape=(43, 1)))     #Static aberrations
        #opt_set, max_sharp = sharpness_opt(np.random.uniform(-1, 1, size=len(dm)))     #Random aberration
        #opt_set, max_sharp = sharpness_opt(np.zeros(shape=(43,1))[40:43]=-1)     #Defocus aberration

        # Half-Width Optimization
        #opt_set, min_r = half_width_opt(np.zeros(shape=(43, 1)))  # Static aberrations
        #opt_set, min_r = half_width_opt(np.random.uniform(-1, 1, size=len(dm)))     #Random aberration
        #opt_set, min_r = half_width.opt(np.zeros(shape=(43,1))[40:43]=-1)     #Defocus aberration

        # Sharpness-Edge Optimization
        #opt_set, max_sharp_edge = sharpness_edge_opt(np.zeros(shape=(43, 1)))  # Static aberrations
        #opt_set, max_sharp_edge = sharpness_edge_opt(np.random.uniform(-1, 1, size=len(dm)))     #Random aberration
        #opt_set, max_sharp_edge = sharpness_edge_opt(np.zeros(shape=(43,1))[40:43]=-1)     #Defocus aberration

        # Plotting
        dm.setActuators(opt_set)
        img = grabframes(5, Camera_Index)
        plt.figure()
        plt.imshow(img[-1])
        plt.figure()
        plt.hist(opt_set, bins=len(dm))
        plt.title('Control Voltage Set')
