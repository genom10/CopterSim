from typing import TypedDict
from myMath import RotationMatrix3D as Rot
import numpy as np

class ParamsQuad(TypedDict):
    mass: float
    inertia: np.ndarray
    width: float
    length: float
    b: float
    d: float
    g: float

class ParamsSim(TypedDict):
    dt: float
    t:  float
    tf: float

class Dynamics:
    def __init__(self, 
                 paramsQuad : ParamsQuad = {'mass': 2,
                                            'inertia': np.diag([0.2, 0.2, 0.2]),
                                            'width': 0.3,
                                            'length': 0.3,
                                            'b': 1,
                                            'd': 1,
                                            'g': 9.81},
                 paramsSim : ParamsSim = {'dt': 0.01,
                                          't': 0,
                                          'tf': 10},
                 position = np.array([0,0,0]),
                 velocity = np.array([0,0,0]),
                 orientation = np.array([0,0,0]),
                 angularVelocity = np.array([0,0,0]),
                 ) -> None:
        self.paramsQuad = paramsQuad
        self.paramsQuad['inv_inertia'] = np.linalg.pinv(paramsQuad['inertia'])
        self.paramsSim = paramsSim
        self.state = np.concatenate((position, velocity, orientation, angularVelocity))

    def dxdt(self, state, u):
        _, _, _, dx, dy, dz, _, _, _, dyaw, dpitch, droll = state
        ddx, ddy, ddz, ddyaw, ddpitch, ddroll = self.calculateAccelerations(state, u)
        return np.array([dx, dy, dz, ddx, ddy, ddz, dyaw, dpitch, droll, ddyaw, ddpitch, ddroll])

    def integrateRK(self, state, u): #Runge-Kutta integration
        k1 = self.paramsSim['dt'] * self.dxdt(state, u)
        k2 = self.paramsSim['dt'] * self.dxdt(state+k1/2, u)
        k3 = self.paramsSim['dt'] * self.dxdt(state+k2/2, u)
        k4 = self.paramsSim['dt'] * self.dxdt(state+k3, u)
        return state + (k1 + k2*2 + k3*2 + k4) / 6
    
    def integrateEuler(self, state, u): #Euler integration
        return state + self.paramsSim['dt'] * self.dxdt(state, u)

    def updateStateVector(self, u=np.array([1,1,1,1]), integrator='RK'):
        self.state = self.calculateStateVector(self.state, u, integrator=integrator)
        self.paramsSim['t'] += self.paramsSim['dt']
        return self.state

    def calculateStateVector(self, state, u, integrator='RK'):
        '''
        ## calculating state vector in next timestamp

        ### Parameters
        ----------
        state : np.array of shape (12)
            each state is composed of
            [x, y, z, dx, dy, dz,
            yaw, pitch, roll, dyaw, dpitch, droll]
        u : np.array of shape (4)
            Motor control values. Each entry composed of angular speeds of [left, tail, right, head] motors
        integrator : str. 'RK' or 'Euler'
            Choose integration method. RK is more precise, Euler is faster but wackier on high speeds 
            '''
        if integrator == 'RK':
            return self.integrateRK(state, u)
        elif integrator == 'Euler':
            return self.integrateEuler(state, u)
        else:
            raise Exception("Unexpected integrator choice")
        

    def calculateAccelerations(self, state, u):
        x, y, z, dx, dy, dz, yaw, pitch, roll, dyaw, dpitch, droll = state
        dattitude = state[9:12]
        b = self.paramsQuad['b']
        d = self.paramsQuad['d']
        w = self.paramsQuad['width']
        l = self.paramsQuad['length']

        thrust = np.array([0, 0, self.paramsQuad['b'] * np.sum(np.power(u, 2, dtype=float))])
        ddx, ddy, ddz = Rot(yaw=yaw, pitch=pitch, roll=roll) @ thrust / self.paramsQuad['mass'] +\
                        np.array([0, 0, -self.paramsQuad['g']])
        
        torques = np.array([w*b*(u[0]**2 - u[2]**2), 
                            l*b*(u[3]**2 - u[1]**2), 
                              d*(u[3]**2 + u[1]**2 - u[0]**2 - u[2]**2)])
        ddroll, ddpitch, ddyaw = self.paramsQuad['inv_inertia'] @ (torques - np.cross(dattitude, np.dot(self.paramsQuad['inertia'], dattitude)))
        
        return ddx, ddy, ddz, ddyaw, ddpitch, ddroll



if __name__ == '__main__':
    # u0, u1, u2, u3 = 0.5, 1.0, 1.5, 1.0
    # ddyaw = u3**2 + u1**2 - u0**2 - u2**2 #OZ
    # ddpitch = u3**2 - u1**2 #OY
    # ddroll = u0**2 - u2**2 #OX
    # print(f'{ddyaw=}\n{ddpitch=}\n{ddroll=}')
    print(Dynamics().calculateStateVector([0,0,0,
                                           0,0,0,
                                           0,0,0,
                                           0,0,0], [2, 0, 2, 2*np.sqrt(2)]))