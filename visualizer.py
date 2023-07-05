from typing import Dict, List, Tuple
from itertools import product, combinations
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Text3D, Line3DCollection, Line3D
from myMath import RotationMatrix3D
import numpy as np
from numpy import sin, cos


def visualise(t, states, torques: List = None, pids: Dict[str, List] = None, torqueScale:float = 0.2):
    ''' 
    ## Drone visualization

    ### Parameters
    ----------
    t : np.array
        n  -  number of timestamps.\n
        timestamps at which all other data is recorded

    states : np.array of shape (n, 12)
        n  -  number of timestamps.\n
        states, where each state is composed of\n
        [x,  y,  z,\n
        dx, dy, dz,\n
        yaw,  pitch,  roll,\n
        dyaw, dpitch, droll]

    torques : np.array of shape (n, 4)
        n  -  number of timestamps.\n
        Contains torques on each timestamp. Each entry composed of torques of [left, tail, right, head] motors
        
    pids : {pidName:np.array of shape (n)} 
        pidName - name of the PID regulator. Will be used as label for plot
        n - number of timestamps.\n
        Contains pid values on each timestamp
        
    torqueScale : float
        scale factor for the torque arrows. Smaller scale -> shorter arrows 
    '''



    def plotDrone3D(ax, x, y, z, yaw, pitch, roll, torque = None, l = 0.1, w = 0.1, torqueScale=0.2):
        core = np.array([x, y, z])
        R = RotationMatrix3D(yaw=yaw, pitch=pitch, roll=roll)
        m0 = core + R@np.array([0,  w, 0]) # left
        m1 = core + R@np.array([-l, 0, 0]) # tail
        m2 = core + R@np.array([0, -w, 0]) # right
        m3 = core + R@np.array([l,  0, 0]) # head
        line0 = ax.plot3D(*np.transpose([core, m0]), color="r") # left
        line1 = ax.plot3D(*np.transpose([core, m1]), color="gray") # tail
        line2 = ax.plot3D(*np.transpose([core, m2]), color="g") # right
        line3 = ax.plot3D(*np.transpose([core, m3]), color="k") # head

        if torque is not None:
            ax.quiver(*m0, *(R@np.array([0,0,torque[0]])), length=torqueScale)
            ax.quiver(*m1, *(R@np.array([0,0,torque[1]])), length=torqueScale)
            ax.quiver(*m2, *(R@np.array([0,0,torque[2]])), length=torqueScale)
            ax.quiver(*m3, *(R@np.array([0,0,torque[3]])), length=torqueScale)

        return (line0, line1, line2, line3)
        
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    states = np.array(states)
    # print(states.shape)

    fig = plt.figure(figsize=(16, 8))
    # Central frame
    dronePlot = plt.subplot2grid((2, 3), (0, 1), rowspan=2, projection='3d')
    dronePlot.set_xlim(-1, 1)
    dronePlot.set_ylim(-1, 1)
    dronePlot.set_zlim(-1, 1)
    dronePlot.set_xlabel("X")
    dronePlot.set_ylabel("Y")
    dronePlot.set_zlabel("Z")
    plotDrone3D(dronePlot, *states[0, :3], *states[0, 6:9], torqueScale=torqueScale)

    # Torques
    lTorque = plt.subplot2grid((3, 6), (0, 0)) # left
    bTorque = plt.subplot2grid((3, 6), (1, 0)) # back
    rTorque = plt.subplot2grid((3, 6), (1, 1)) # right
    fTorque = plt.subplot2grid((3, 6), (0, 1)) # front
    lTorque.set_title(r'Left motor torque [m_0]')
    bTorque.set_title(r'Back motor torque [m_1]', y=-0.0, pad=-30)
    rTorque.set_title(r'Right motor torque [m_2]', y=-0.0, pad=-30)
    fTorque.set_title(r'Front motor torque [m_3]')
    if torques is not None:
        lTorque.plot(t, torques[:, 0], 'red')
        bTorque.plot(t, torques[:, 1], 'gray')
        rTorque.plot(t, torques[:, 2], 'green')
        fTorque.plot(t, torques[:, 3], 'black')
    lRadio = bTorque.axvline(0, c='k', alpha = 0.3)
    bRadio = lTorque.axvline(0, c='k', alpha = 0.3)
    rRadio = fTorque.axvline(0, c='k', alpha = 0.3)
    fRadio = rTorque.axvline(0, c='k', alpha = 0.3)
        
    # PIDs
    pidRadios = []
    if pids is not None:
        m = len(pids.items())
        for i, (PIDname, PIDvalues) in enumerate(pids.items()):
            axpid = plt.subplot2grid((m, 3), (i, 2)) # left
            axpid.plot(t, PIDvalues)
            pidRadios.append(axpid.axvline(0, c='k', alpha = 0.3))
            axpid.set_title(PIDname, y=0.5, x=1.1)
            # print(PIDname, PIDvalues)

    # Slider
    axsl1 = fig.add_axes([0.3, 0.05, 0.4, 0.03])
    slider1 = Slider(
        ax=axsl1,
        label='time',
        valmin=t[0],
        valmax=t[-1],
        valinit=t[0],
    )
    def update(val):
        fig.canvas.draw_idle()
        for object in dronePlot.get_children():
            if isinstance(object, Line3D) or isinstance(object, Line3DCollection):
                object.remove()
            # else:
            #     print(type(object))
        i = find_nearest(val, t)
        if torques is None:
            plotDrone3D(dronePlot, *states[i, :3], *states[i, 6:9], torqueScale=torqueScale)
        else:
            plotDrone3D(dronePlot, *states[i, :3], *states[i, 6:9], torques[i], torqueScale=torqueScale)
        
        bRadio.set_xdata(val)
        lRadio.set_xdata(val)
        fRadio.set_xdata(val)
        rRadio.set_xdata(val)
        for radio in pidRadios:
            radio.set_xdata(val)
    slider1.on_changed(update)
    fig.tight_layout()
    plt.show()
        

if __name__ == '__main__':
    visualise(t=[0, 0.1, 0.2, 0.3], 
              states=np.array([[0,0,0,
                                0,0,0,
                                0,0,0,
                                0,0,0],

                                [0,0,0,
                                0,0,0,
                                .2,0,0,
                                0,0,0],

                                [0,0,0,
                                0,0,0,
                                0,.2,0,
                                0,0,0],

                                [0,0,0,
                                0,0,0,
                                0,0,.2,
                                0,0,0],
                                ]),

              torques=np.array([[1.0, 1.0, 1.0, 1.0],
                                [0.5, 1.5, 0.5, 1.5], # yaw+
                                [1.0, 0.5, 1.0, 1.5], # ptich+
                                [1.5, 1.0, 0.5, 1.0]]), # roll+

              pids={'pid1' : [1.0, 1.0, 1.0, 1.0],
                    'pid2' : [1.0, 1.5, 2.0, 1.5],
                    'pid3' : [0.0, 0.1, 0.3, 0.9],
                    'pid4' : [1.0, 2.0, 2.0, 1.0],
                    'pid5' : [1.0, 2.0, 2.0, 1.0],
                    'pid6' : [1.0, 2.0, 2.0, 1.0],
                    'pid7' : [1.0, 2.0, 2.0, 1.0]})