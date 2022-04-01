import os.path as osp
import numpy as np
import pybullet as p


class BulletPhysicsEngine(object):
    """Physics engine API wrapper for Bullet."""

    def __init__(self):
        self._gravity = None

    @staticmethod
    def euler_from_quat(quat):
        quat = list(quat)
        euler = p.getEulerFromQuaternion(quat)
        return np.array(euler, dtype=np.float32)

    @staticmethod
    def euler_from_mat33(mat33):
        mat33 = list(mat33.flatten())
        raise NotImplementedError

    @staticmethod
    def quat_from_euler(euler):
        euler = list(euler)
        quat = p.getQuaternionFromEuler(euler)
        return np.array(quat, dtype=np.float32)

    @staticmethod
    def quat_from_mat33(mat33):
        mat33 = list(mat33.flatten())
        raise NotImplementedError

    @staticmethod
    def mat33_from_quat(quat):
        quat = list(quat)
        mat33 = p.getMatrixFromQuaternion(quat)
        return np.reshape(mat33, [3, 3])

    @staticmethod
    def mat33_from_euler(euler):
        euler = list(euler)
        quat = p.getQuaternionFromEuler(euler)
        mat33 = p.getMatrixFromQuaternion(quat)
        return np.reshape(mat33, [3, 3])

    @staticmethod
    def pos_in_frame(pos, frame):
        frame_xyz = frame[0]
        frame_rpy = frame[1]
        quat = p.getQuaternionFromEuler(frame_rpy)
        mat33 = p.getMatrixFromQuaternion(quat)
        mat33 = np.reshape(mat33, [3, 3])
        pos_in_frame = frame_xyz + np.dot(pos, mat33.T)
        return pos_in_frame

    @staticmethod
    def euler_in_frame(euler, frame):
        frame_rpy = frame[1]
        euler_in_frame = np.array(euler) + frame_rpy
        return euler_in_frame

    @staticmethod
    def get_body_pos(body):
        pos, _ = p.getBasePositionAndOrientation(body)
        return np.array(pos, dtype=np.float32)

    @staticmethod
    def get_body_quat(body):
        _, quat = p.getBasePositionAndOrientation(body)
        return np.array(quat, dtype=np.float32)

    @staticmethod
    def get_body_euler(body):
        _, quat = p.getBasePositionAndOrientation(body)
        euler = p.getEulerFromQuaternion(quat)
        return np.array(euler, dtype=np.float32)

    @staticmethod
    def get_body_pose(body):
        return p.getBasePositionAndOrientation(body)

    @staticmethod
    def get_body_mat33(body):
        _, quat = p.getBasePositionAndOrientation(body)
        mat33 = p.getMatrixFromQuaternion(quat)
        return np.reshape(mat33, [3, 3])

    @staticmethod
    def get_body_linvel(body):
        linvel, _ = p.getBaseVelocity(body)
        return linvel

    @staticmethod
    def get_body_angvel(body):
        _, angvel = p.getBaseVelocity(body)
        return angvel

    @staticmethod
    def get_link_uids(body):
        # Links and Joints have the corresponding UIDs
        link_uids = range(p.getNumJoints(body))
        return link_uids

    @staticmethod
    def get_link_name(body, link):
        # Links and Joints have the corresponding UIDs
        _, joint_name, joint_type, _, _, _, damping, friction, \
                lower, upper, max_force, max_vel, _ = \
                p.getJointInfo(body, link)
        return joint_name # TODO

    @staticmethod
    def get_link_pos(body, link):
        pos, _, _, _, _, _, _, _  = p.getLinkState(body, link)
        return np.array(pos, dtype=np.float32)

    @staticmethod
    def get_link_quat(body, link):
        _, quat, _, _, _, _, _, _  = p.getLinkState(body, link)
        return np.array(quat, dtype=np.float32)

    @staticmethod
    def get_joint_uids(body):
        joint_uids = range(p.getNumJoints(body))
        return joint_uids

    @staticmethod
    def get_joint_name(body, joint):
        # Links and Joints have the corresponding UIDs
        _, joint_name, joint_type, _, _, _, damping, friction, \
                lower, upper, max_force, max_vel, _ = \
                p.getJointInfo(body, joint)
        return joint_name

    @staticmethod
    def get_joint_dynamics(body, joint):
        # Links and Joints have the corresponding UIDs
        _, joint_name, joint_type, _, _, _, damping, friction, \
                lower, upper, max_force, max_vel, _ = \
                p.getJointInfo(body, joint)
        dynamics = {
                'damping': damping,
                'friction': friction
                }
        return dynamics

    @staticmethod
    def get_joint_limit(body, joint):
        # Links and Joints have the corresponding UIDs
        _, joint_name, joint_type, _, _, _, damping, friction, \
                lower, upper, max_force, max_vel, _ = \
                p.getJointInfo(body, joint)
        limit = {
                'lower': lower,
                'upper': upper,
                'effort': max_force,
                'velocity': max_vel
                }
        return limit

    @staticmethod
    def get_joint_pos(body, joint):
        pos, _, react_force, torque = p.getJointState(body, joint)
        return np.array(pos, dtype=np.float32)

    @staticmethod
    def get_joint_vel(body, joint):
        _, vel, react_force, torque = p.getJointState(body, joint)
        return np.array(vel, dtype=np.float32)

    @staticmethod
    def get_joint_force(body, joint):
        # TODO: Definition of react_force?
        _, _, react_force, _ = p.getJointState(body, joint)
        return np.array(react_force, dtype=np.float32)

    @staticmethod
    def get_joint_torque(body, joint):
        _, _, _, torque = p.getJointState(body, joint)
        return np.array(torque, dtype=np.float32)

    @staticmethod
    def set_body_pos(body, pos):
        _, quat = p.getBasePositionAndOrientation(body)
        p.resetBasePositionAndOrientation(body, list(pos), quat)

    @staticmethod
    def set_body_pose(body, pose):
        p.resetBasePositionAndOrientation(body, pose[0], pose[1])

    @staticmethod
    def set_body_quat(body, quat):
        pos, _ = p.getBasePositionAndOrientation(body)
        p.resetBasePositionAndOrientation(body, pos, list(quat))

    @staticmethod
    def set_body_linvel(body, linvel):
        p.resetBaseVelocity(body, linearVelocity=list(linvel))

    @staticmethod
    def set_body_angvel(body, angvel):
        p.resetBaseVelocity(body, angularVelocity=list(angvel))

    @staticmethod
    def set_joint_state(body, joint, pos):
        p.resetJointState(body, joint, list(pos))

    @staticmethod
    def set_joint_vel(body, joint, vel):
        pos, _, _, _ = p.getJointState(body, joint)
        p.resetJointState(body, joint, pos, list(vel))

    @staticmethod
    def control_joint_pos(body, joint, pos, max_vel=None, max_force=None):
        if max_vel is None:
            p.setJointMotorControl2(
                    bodyIndex=body,
                    jointIndex=joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=pos,
                    force=max_force)
        else:
            p.setJointMotorControl2(
                    bodyIndex=body,
                    jointIndex=joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=pos,
                    targetVelocity=max_vel,
                    force=max_force)

    @staticmethod
    def control_joint_vel(body, joint, vel, max_force=None):
        p.setJointMotorControl2(
                bodyIndex=body,
                jointIndex=joint,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vel,
                force=max_force)

    @staticmethod
    def control_joint_torque(body, joint, torque):
        p.setJointMotorControl2(
                bodyIndex=body,
                jointIndex=joint,
                controlMode=p.TORQUE_CONTROL,
                force=torque)

    @staticmethod
    def create_cstr(descr):
        if descr['joint_type'] == 'revolute':
            joint_type = p.JOINT_REVOLUTE
        elif descr['joint_type'] == 'prismatic':
            joint_type = p.JOINT_PRISMATIC
        elif descr['joint_type'] == 'fixed':
            joint_type = p.JOINT_FIXED
        elif descr['joint_type'] == 'point2point':
            joint_type = p.JOINT_POINT2POINT
        else:
            raise ValueError('Unrecognized joint type {}'.format(descr['joint_type']))

        if (descr['parent_frame_quat'] is None) or \
                (descr['child_frame_quat'] is None):
            cstr = p.createConstraint(
                    parentBodyUniqueId=descr['parent_body'],
                    parentLinkIndex=descr['parent_link'],
                    childBodyUniqueId=descr['child_body'],
                    childLinkIndex=descr['child_link'],
                    jointType=joint_type,
                    jointAxis=descr['joint_axis'],
                    parentFramePosition=descr['parent_frame_pos'],
                    childFramePosition=descr['child_frame_pos'])
        else:
            cstr = p.createConstraint(
                    parentBodyUniqueId=descr['parent_body'],
                    parentLinkIndex=descr['parent_link'],
                    childBodyUniqueId=descr['child_body'],
                    childLinkIndex=descr['child_link'],
                    jointType=joint_type,
                    jointAxis=descr['joint_axis'],
                    parentFramePosition=descr['parent_frame_pos'],
                    childFramePosition=descr['child_frame_pos'],
                    parentFrameOrientation=descr['parent_frame_quat'],
                    childFrameOrientation=descr['child_frame_quat'])

        return cstr

    @staticmethod
    def get_cstr_dof(cstr):
        _, _, _, _, _, _, _, pos, _, quat, max_force = p.getConstraintInfo(cstr)
        pos = np.array(pos, dtype=np.float32)
        euler = p.getEulerFromQuaternion(quat)
        euler = np.array(euler, dtype=np.float32)
        return pos, euler

    @staticmethod
    def get_cstr_max_force(cstr):
        _, _, _, _, _, _, _, pos, _, quat, max_force = p.getConstraintInfo(cstr)
        return np.array(max_force, dtype=np.float32)

    @staticmethod
    def set_cstr_dof(cstr, pos, euler, max_force):
        pos = list(pos)
        euler = list(euler)
        quat = p.getQuaternionFromEuler(euler)
        p.changeConstraint(
                userConstraintUniqueId=cstr,
                jointChildPivot=pos,
                jointChildFrameOrientation=quat,
                maxForce=max_force)

    @staticmethod
    def remove_cstr(cstr):
        p.removeConstraint(cstr)

    @staticmethod
    def set_camera(focal_point, focal_dist, euler, width=500, height=540):
        """ """
        _euler = euler / np.pi * 180
        roll = _euler[0]
        pitch = _euler[1]
        yaw = _euler[2]
        aspect = float(height) / float(width)
        # TODO: Order of pitch and yaw for the camera?
        p.resetDebugVisualizerCamera(
                cameraDistance=focal_dist,
                cameraYaw=pitch,
                cameraPitch=yaw,
                cameraTargetPosition=focal_point)

    def set_gravity(self, gravity):
        self._gravity = gravity
        p.setGravity(gravity[0], gravity[1], gravity[2])

    @staticmethod
    def load(path, pos=[0, 0, 0], euler=[0, 0, 0], fixed=False):
        """Load an body into the simulation."""
        # Set data path
        assert osp.exists(path), \
                'Model path {} does not exist.'.format(path)
        # Load body model
        model_name, ext = osp.splitext(path)
        if ext == '.urdf':
            quat = BulletPhysicsEngine.quat_from_euler(euler)
            uid = p.loadURDF(path, pos, quat, useFixedBase=fixed)
        elif ext == '.sdf':
            uid = p.loadSDF(path)
        else:
            raise ValueError('Unrecognized extension {}.'.format(ext))
        return uid

    @property
    def apply_force(uid, lid, force, pos):
        p.applyExternalForce(uid, lid, force, pos, p.WORLD_FRAME)


class Entity(object):
    """Entity base class."""
    def __init__(self, pe, name=None):
        # Physics engine API wrapper
        self._pe = pe
        # Link name
        self._name = name

    @property
    def pe(self):
        return self._pe

    @property
    def name(self):
        return self._name


class Body(Entity):
    """Body."""

    def __init__(self, uid, name=None, boundary=None, scale=None, pe=BulletPhysicsEngine):
        super(Body, self).__init__(pe, name)
        # Physics engine API wrapper
        self._uid = uid
        # Name
        self._name = name
        self._boundary = boundary
        self._scale = scale

    @ classmethod
    def create_from_descr(cls, pe, descr, data_dir='./data', frame=None):
        """Create an instance from description."""
        filename = descr['model_path']
        boundary, scale = None, None
        if 'boundary' in descr:
            boundary = np.array(descr['boundary'])
        if 'scale' in descr:
            scale = np.array(descr['scale'])

        xyz = descr['pose']['xyz']
        rpy = descr['pose']['rpy']
        if frame is not None:
            xyz = pe.pos_in_frame(xyz, frame)
            rpy = pe.euler_in_frame(rpy, frame)
        fixed = descr['fixed']
        path = osp.join(data_dir, filename)
        uid = pe.load(path, xyz, rpy, fixed)
        name = descr['name']
        return cls(pe, uid, name, boundary, scale)

    @property
    def name(self):
        return self._name

    @property
    def boundary(self):
        return self._boundary

    @property
    def scale(self):
        return self._scale

    @property
    def uid(self):
        return self._uid

    @property
    def pose(self):
        return self.pe.get_body_pose(self._uid)

    @property
    def links(self):
        return self._links

    @property
    def joints(self):
        return self._joints

    @property
    def pos(self):
        return self.pe.get_body_pos(self.uid)

    @pos.setter
    def pos(self, value):
        self.pe.set_body_pos(self.uid, value)

    @property
    def quat(self):
        return self.pe.get_body_quat(self.uid)

    @quat.setter
    def quat(self, value):
        self.pe.set_body_quat(self.uid, value)

    @property
    def euler(self):
        return self.pe.get_body_euler(self.uid)

    @euler.setter
    def euler(self, value):
        self.pe.set_body_euler(self.uid, value)

    def apply_force(self, vec):
        self.pe.apply_force(self.uid, -1, vec, self.pos)
