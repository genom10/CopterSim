from typing import Dict, List, Tuple
from enum import Enum
import numpy as np
from myMath import RotationMatrix3D, invRotationMatrix3D
from nptyping import NDArray, Float64

def enumPIDs(**enums):
    # print(enums)
    ret = type('Enum', (), enums)
    ret.keys = list(enums.keys())
    for key in ret.keys:
        # print(f'{key=}')
        # print(ret.__dict__[key])
        ret.__annotations__[key] = PID
    return ret

class PID:
    def __init__(self,
                 params: Tuple[float, float, float],
                 saturation: Tuple[float, float]) -> None:
        self.p, self.i, self.d = params
        self.saturation_min, self.saturation_max = saturation
        self.lastErr = None # Error on the previous step
        self.accuErr = 0    # Accumulated error
        self.val = 0

    def update(self, value: float, target: float, dt: float, verbose:bool = False) -> float:
        err = target - value
        self.accuErr += err*dt
        if self.lastErr is None:
            self.lastErr = err

        ret = self.p * err +\
              self.i * self.accuErr+\
              self.d * (err - self.lastErr)/dt
        if verbose:
            print(f'{err=}\t{self.lastErr=}\t{self.accuErr=}')
            print(f'\033[94m{self.p * err:.2f}(P) {self.i * self.accuErr:+.2f}(I) {self.d * (err - self.lastErr)/dt:+.2f}(D) = {ret:+.2f}\033[0m')
        self.lastErr = err
        
        self.val = np.clip(ret, self.saturation_min, self.saturation_max)
        return self.val
    
    def read(self):
        return self.val

class Controller:
    def __init__(self,
                 velLim: tuple[float, float] = (-2.0, 2.0),
                 accLim: tuple[float, float] = (-1.5, 1.5),
                 thrustLim: tuple[float, float] = (-5, 5),
                 angleLim: tuple[float, float] = (-1, 1),
                 dt: float = 0.1) -> None:
        self.dt = dt
        self.PIDs = enumPIDs(x = PID((6,0.03,0.8), velLim),
                            y = PID((6,0.03,0.8), velLim),
                            z = PID((5,0.1,1), velLim),
                            dx = PID((0.4,0,0.3), accLim),
                            dy = PID((0.4,0,0.3), accLim),
                            dz = PID((5,0.0,0.01), thrustLim),
                            pitch = PID((0.5,0,0.07), angleLim), # OY rotation
                            roll = PID((0.5,0,0.07), angleLim), # OX rotation
                            yaw = PID((1,0.01,1.2), angleLim), # OZ rotation
                            ) 
        # print(self.__getattribute__('PIDs').dx)
        
    def update(self, state, targetPos, targetYaw):
        targetX, targetY, targetZ = targetPos
        x, y, z, dx, dy, dz, yaw, pitch, roll, dyaw, dpitch, droll = state
        self.PIDs.x.update(value=x, target=targetX, dt=self.dt)
        self.PIDs.y.update(value=y, target=targetY, dt=self.dt)
        self.PIDs.z.update(value=z, target=targetZ, dt=self.dt)
        self.PIDs.dx.update(value=dx, target=self.PIDs.x.read(), dt=self.dt)
        self.PIDs.dy.update(value=dy, target=self.PIDs.y.read(), dt=self.dt)
        self.PIDs.dz.update(value=dz, target=self.PIDs.z.read(), dt=self.dt)

        targetDir = np.array([self.PIDs.dx.read(), self.PIDs.dy.read(), self.PIDs.dz.read()])
        targetPitch, targetRoll, _ = invRotationMatrix3D(yaw=yaw, pitch=pitch, roll=roll) @ targetDir # bring to drone coordinates
        
        self.PIDs.pitch.update(value=pitch, target=-targetPitch, dt=self.dt) # TODO maybe have to invert?
        self.PIDs.roll.update(value=roll, target=-targetRoll, dt=self.dt)    # TODO maybe have to invert?
        self.PIDs.yaw.update(value=yaw, target=targetYaw, dt=self.dt)
        
        return self.getTorques()


    def getTorques(self):
        thrust = self.PIDs.dz.read()
        OY = self.PIDs.pitch.read()
        OX = self.PIDs.roll.read()
        OZ = self.PIDs.yaw.read()
        u0 = thrust - OZ
        u2 = thrust - OZ
        u1 = thrust + OZ
        u3 = thrust + OZ

        OX = min(OX, u0, u2)
        u0 += OX 
        u2 -= OX
        OY = min(OY, u1, u3)
        u1 -= OY
        u3 += OY
        return u0, u1, u2, u3
    
    def getPidKeys(self):
        return self.PIDs.keys

    def getPidValues(self):
        return [self.PIDs.__dict__[key].read() for key in self.getPidKeys()]

if __name__ == '__main__':
    controller = Controller(velLim=(-4, 4),
                            thrustLim=(-3, 3),
                            angleLim=(-1, 1),
                            dt = 0.1)
    state = np.array([0, 0, 0, # x,  y,  z
                      0, 0, 0, # dx, dy, dz
                      0, 0, 0, # yaw  pitch  roll
                      0, 0, 0])# dyaw dpitch droll
    controller.update(state, [0, 0, 1], 0)