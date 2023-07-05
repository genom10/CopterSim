from dynamics import Dynamics
from controller import Controller
from visualizer import visualise
import numpy as np

########################################################
#
#   Made by @Genom10 as assignment work for SkillBox
#
########################################################


dynamics = Dynamics(paramsSim={'dt': 0.01,
                                't': 0,
                                'tf': 10})
controller = Controller(dt=dynamics.paramsSim['dt'])
t = [0]
torques = [[0, 0, 0, 0]]
states = [[0, 0, 0,
           0, 0, 0,
           0, 0, 0,
           0, 0, 0]]
PIDs = [[0. for i in controller.getPidKeys()]] 
try:
    while dynamics.paramsSim['t'] < dynamics.paramsSim['tf']:
        control = np.array(controller.update(state = states[-1], targetPos=[1, 1, 0], targetYaw=2))
        # control = np.clip(control, a_min=0, a_max=np.inf)
        torques.append(control)
        states.append(dynamics.updateStateVector(u=control))
        t.append(dynamics.paramsSim['t'])
        PIDs.append(controller.getPidValues())
except Exception as e:
    print("\033[93mSomething went wrong! Read the error below:\033[0m")
    print(e)

PIDs = dict(zip(controller.getPidKeys(), np.transpose(PIDs)))

torques = np.array(torques)
states = np.array(states)
visualise(states=states, t=t, torques=torques, pids=PIDs)
    