from numpy import sin, cos, array
from numpy.linalg import pinv


def RotationMatrix3D(yaw: float, pitch: float, roll: float):
    Ry = array([[cos(yaw),   sin(yaw),  0],            # OZ
                [-sin(yaw),  cos(yaw),  0],
                [0,          0,         1]])
    Rp = array([[cos(pitch), 0,         -sin(pitch)],   # OY 
                [0,          1,         0],
                [sin(pitch), 0,         cos(pitch)]])
    Rr = array([[1,          0,         0],            # OX
                [0,          cos(roll), -sin(roll)],
                [0,          sin(roll), cos(roll)]])
    return Ry@Rp@Rr

def invRotationMatrix3D(yaw: float, pitch: float, roll: float):
    return pinv(RotationMatrix3D(yaw=yaw, pitch=pitch, roll=roll))

if __name__ == '__main__':
    print(RotationMatrix3D(yaw=1, pitch = 0.5, roll = 2) @ RotationMatrix3D(yaw=-1, pitch = -0.5, roll = -2)) #not ok
    print(RotationMatrix3D(yaw=1, pitch = 0.5, roll = 2) @ invRotationMatrix3D(yaw=1, pitch = 0.5, roll = 2)) #marginally ok