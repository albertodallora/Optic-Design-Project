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
        step = 0.5
        min_obj = 0
        SM = 0
        C_reg = 1
        C_pos = 1
        C_norm = 1
        obj = np.zeros(shape=(len(dm)+1,1))

        # Optimization process
        # Compute Nelder-Mead Vertices
        for i in range(len(dm)):
            for k in range(len(dm) + 1):
                temp = np.zeros(len(dm))
                if i > 0:
                    temp[i] = step
                vertex_matrix[k,i] = init_set + temp
        # Compute SM for all the vertices
        for k in range(len(dm) + 1):
            dm.setActuators(vector_matrix[k,])
            img = grabframes(3, Camera_Index)
            x_intens = 0
            y_intens = 0
            # Compute centroid
            intens_sum = np.matrix.sum(img[-1, i, k])
            for i in range(1024):
                for m in range(1280):
                    x_intens += m * img[-1, i, m]
                    y_intens += m * img[-1, i, m]
            centroid_x = x_intens / intens_sum
            centroid_y = y_intens / intens_sum
            positioning = np.sqrt(centroid_x ** 2 + centroid_y ** 2)
            # Compute second-order metric
            for i in range(1024):
                for n in range(1280):
                    SM += img[-1, i, k] * ((n-640)**2 + (i-512)**2)

            obj[k] = C_norm * SM + C_reg * np.linalg.norm(DM_set) + C_pos * positioning
        # Opt. Iterations
        while gain > threshold:
            # Change the worst vertex
            worst_index = np.argmax(obj)
            vector_matrix[worst_index,] = vector_matrix[worst_index,] - np.ones((len(dm),1)*0.25
            #Compute SM for the new vertex
            dm.setActuators(vector_matrix[worst_index,])
            img = grabframes(3, Camera_Index)
            x_intens = 0
            y_intens = 0
            # Compute centroid
            intens_sum = np.matrix.sum(img[-1, i, k])
            for i in range(1024):
                for m in range(1280):
                    x_intens += m * img[-1, i, m]
                    y_intens += m * img[-1, i, m]
            centroid_x = x_intens / intens_sum
            centroid_y = y_intens / intens_sum
            positioning = np.sqrt(centroid_x ** 2 + centroid_y ** 2)
            # Compute second-order metric
            for i in range(1024):
                for n in range(1280):
                    SM += img[-1, i, k] * (((n - 640) ** 2 + (i - 512) ** 2))

            worst_obj = obj[worst_index]
            obj[worst_index] = C_norm * SM + C_reg * np.linalg.norm(DM_set) + C_pos * positioning

            gain = worst_obj - obj[worst_index]
            if gain < threshold:
                opt_SM = SM     #the last computed
                opt_set = vertex_matrix[worst_index,]

        return opt_set, opt_SM



if __name__ == "__main__":
    from dm.thorlabs.dm import ThorlabsDM

    with ThorlabsDM() as dm:
        #SM Optimization
        opt_set, opt_SM = SM_opt(np.zeros(shape=(43, 1)))     #Static aberrations
        #opt_set, opt_SM = SM_opt(np.random.uniform(-1, 1, size=len(dm)))     #Random aberration
        #opt_set, opt_SM = SM_opt(np.zeros(shape=(43,1))[40:43]=-1)     #Defocus aberration


        # Plotting
        dm.setActuators(opt_set)
        img = grabframes(5, Camera_Index)
        plt.figure()
        plt.imshow(img[-1])
        plt.figure()
        plt.hist(opt_set, bins=len(dm))
        plt.title('Control Voltage Set')
