import numpy as np


def rodrigues_to_rotation_matrix(r):
    """
    Convert Rodrigues parameters (rotation vector) to a 3×3 rotation matrix.
    r: 3‑element array-like Rodrigues vector
    """
    r = np.asarray(r, dtype=float)
    s = np.linalg.norm(r)

    # Handle the zero-rotation case
    if s < 1e-12:
        return np.eye(3)

    k = r / s                          # rotation axis
    theta = 2 * np.arctan(s)           # rotation angle

    # Skew-symmetric matrix
    K = np.array([
        [0,      -k[2],   k[1]],
        [k[2],    0,     -k[0]],
        [-k[1],   k[0],    0   ]
    ])

    # Rodrigues' rotation formula
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def rotation_matrix_to_euler(R):
    """
    Convert rotation matrix to Euler angles (roll, pitch, yaw)
    using the XYZ intrinsic rotation convention.
    """
    # pitch = -asin(R31)
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))

    # Handle possible gimbal-lock
    if abs(R[2, 0]) < 0.999999:
        roll  = np.arctan2(R[2, 1], R[2, 2])
        yaw   = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Near gimbal lock: pitch ≈ ±90°
        roll = 0.0
        yaw  = np.arctan2(-R[0, 1], R[1, 1])

    return roll, pitch, yaw


def rodrigues_to_euler(r):
    """
    Full conversion: Rodrigues vector → Euler angles (roll, pitch, yaw).
    """
    R = rodrigues_to_rotation_matrix(r)
    return rotation_matrix_to_euler(R)


def rodrigues_to_q(r):

    q = np.array([1, r[0], r[1], r[2]])/np.sqrt(1+np.linalg.norm(r))

    return q

def quat_to_euler_rpy(q, degrees=False):
    """
    Convert a unit quaternion q=(w, x, y, z) to Euler angles (roll, pitch, yaw)
    using the intrinsic XYZ (roll–pitch–yaw) convention.

    Parameters
    ----------
    q : iterable of 4 floats
        Quaternion in (w, x, y, z) order. Will be normalized internally.
    degrees : bool
        If True, returns angles in degrees; otherwise radians.

    Returns
    -------
    (roll, pitch, yaw) : tuple of floats
    """
    q = np.asarray(q, dtype=float)
    # Normalize to guard against drift
    q = q / np.linalg.norm(q)

    w, x, y, z = q

    # Rotation matrix elements from quaternion
    R11 = 1 - 2*(y*y + z*z)
    R12 = 2*(x*y - z*w)
    R13 = 2*(x*z + y*w)
    R21 = 2*(x*y + z*w)
    R22 = 1 - 2*(x*x + z*z)
    R23 = 2*(y*z - x*w)
    R31 = 2*(x*z - y*w)
    R32 = 2*(y*z + x*w)
    R33 = 1 - 2*(x*x + y*y)

    # Intrinsic XYZ extraction:
    # pitch = -asin(R31)
    pitch = -np.arcsin(np.clip(R31, -1.0, 1.0))

    # Check for gimbal lock
    if abs(R31) < 0.999999:
        roll = np.arctan2(R32, R33)
        yaw  = np.arctan2(R21, R11)
    else:
        # Near ±90° pitch: set roll = 0 and compute yaw from R12/R22
        roll = 0.0
        yaw  = np.arctan2(-R12, R22)

    if degrees:
        return tuple(np.degrees([roll, pitch, yaw]))
    return roll, pitch, yaw


def rod2euler(r):
    q = rodrigues_to_q(r)
    roll, pitch, yaw = quat_to_euler_rpy(q)
    return roll, pitch, yaw

def q2rod(q):
    r = q[1:4]/q[0]
    return r


def euler_to_quat(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)
    using intrinsic XYZ convention.
    """
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy

    return np.array([w, x, y, z])


def euler_to_rodrigues(rpy):

    q = euler_to_quat(rpy[0], rpy[1], rpy[2])
    r = q2rod(q)
    return r