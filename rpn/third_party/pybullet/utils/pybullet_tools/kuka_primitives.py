import time
import numpy as np
import copy

from .pr2_utils import get_top_grasps
from .utils import Euler, draw_pose, get_pose, set_pose, get_movable_joints, get_configuration, \
    set_joint_positions, add_fixed_constraint, enable_real_time, disable_real_time, joint_controller, \
    enable_gravity, get_refine_fn, user_input, wait_for_duration, link_from_name, get_body_name, sample_placement, \
    end_effector_from_body, approach_from_grasp, plan_joint_motion, GraspInfo, Pose, INF, Point, \
    inverse_kinematics, pairwise_collision, remove_fixed_constraint, Attachment, get_sample_fn, \
    step_simulation, refine_path, plan_direct_joint_motion, center_placement, sample_center_placement, \
    draw_pose, get_name, grasp_pose_offset
from .ur5_primitives import GRIPPER_GRASPED_LIFT_HEIGHT, APPROACH_POSE_OFFSET, GRIPPER_MAX_WIDTH, close_gripper, gripper_contact, \
    gripper_contact, control_joints, open_gripper
from ..models.ur5.utilities import setup_sisbot

PHYSICS = False

GRASP_INFO = {
    'top': GraspInfo(lambda body: get_top_grasps(body, under=True, tool_pose = Pose(), #([0.18, 0., 0.], [0., 0.70710678, 0., 0.70710678]),   # Original: tool_pose=Pose(),
                                                max_width = INF, grasp_length=0), # Original: max_width=INF,  grasp_length=0),
                                                Pose(APPROACH_POSE_OFFSET*Point(z=1))), # Original: Pose(0.1*Point(z=1))),
}
TOOL_FRAMES = {
    'iiwa14': 'iiwa_link_ee_kuka', # iiwa_link_ee
    'cube_gripper': 'ee_link',
    'wsg50_with_gripper': 'base_link',
    'ur5_robotiq_85': 'ee_link'
}

DEBUG_FAILURE = False

RESET_JOINTS_CONF = [0,0,0,0,0,0]

def save_min_max_position(configuration, min_joints_conf, max_joints_conf):
    for i, conf in enumerate(configuration):
        if conf < min_joints_conf[i]:
            min_joints_conf[i] = conf
            
        elif conf > max_joints_conf[i]:
            max_joints_conf[i] = conf
            
    
    return min_joints_conf, max_joints_conf
    
def get_min_max_position(min_joints_conf, max_joints_conf):
    return min_joints_conf, max_joints_conf

def reset_min_max_position(min_joints_conf, max_joints_conf):
    min_joints_conf = RESET_JOINTS_CONF
    max_joints_conf = RESET_JOINTS_CONF

    return min_joints_conf, max_joints_conf

class BodyPose(object):
    def __init__(self, body, pose=None):
        if pose is None:
            pose = get_pose(body)
        self.body = body
        self.pose = pose
    def assign(self):
        set_pose(self.body, self.pose)
        return self.pose
    def __repr__(self):
        return 'p{}'.format(id(self) % 1000)


class BodyGrasp(object):
    def __init__(self, body, grasp_pose, approach_pose, robot, link):
        self.body = body
        self.grasp_pose = grasp_pose
        self.approach_pose = approach_pose
        self.robot = robot
        self.link = link
    #def constraint(self):
    #    grasp_constraint()
    def attachment(self):
        return Attachment(self.robot, self.link, self.grasp_pose, self.body)
    def assign(self):
        return self.attachment().assign()
    def __repr__(self):
        return 'g{}'.format(id(self) % 1000)


class BodyConf(object):
    def __init__(self, body, configuration=None, joints=None):
        if joints is None:
            joints = get_movable_joints(body)
        if configuration is None:
            configuration = get_configuration(body)
        self.body = body
        self.joints = joints
        self.configuration = configuration
    def assign(self):
        set_joint_positions(self.body, self.joints, self.configuration)
        return self.configuration
    def __repr__(self):
        return 'q{}'.format(id(self) % 1000)


class BodyPath(object):
    def __init__(self, body, path, digits, joints=None, attachments=[]):
        if joints is None:
            joints = get_movable_joints(body)
        self.body = body
        self.path = path
        self.joints = joints
        self.attachments = attachments
        
        #Modification for ur5
        if 'ur5' in get_body_name(body):
            self.gripper_type = get_body_name(body).split("_")[-1]
            self.jointsDict, self.controlGripper, self.controlJoints, self.mimicParentName =\
                setup_sisbot(body, self.gripper_type)
            self.still_open_flag_ = True  # Hot fix
            self.digits = digits          # Tactile sensor

    def bodies(self):
        return set([self.body] + [attachment.body for attachment in self.attachments])
    def iterator(self):
        # TODO: compute and cache these
        # TODO: compute bounding boxes as well
        
        for i, configuration in enumerate(self.path):
            enable_gravity()
            if 'ur5' in get_body_name(self.body):
                if self.attachments == []:
                    set_joint_positions(self.body, self.joints, configuration)
                    #open_gripper(self.controlGripper) # it does not work
                for grasp in self.attachments:
                    while self.still_open_flag_ and not gripper_contact(self.body, self.jointsDict):
                        self.still_open_flag_ = close_gripper(self.body, self.jointsDict, self.controlGripper, \
                            self.controlJoints, self.mimicParentName, self.digits, check_contact = True)
                    #print("Grab:   ", self.still_open_flag_)
                    control_joints(self.body, self.jointsDict, self.controlJoints, configuration)
            else:
                set_joint_positions(self.body, self.joints, configuration)
                for grasp in self.attachments:
                    #print("Grab sim")
                    grasp.assign()
            if PHYSICS:
                step_simulation()
            yield i, configuration
    def control(self, real_time=False, dt=0):
        # TODO: just waypoints
        if real_time:
            enable_real_time()
        else:
            disable_real_time()
        current_conf = None
        for values in self.path:
            for conf in joint_controller(self.body, self.joints, values):
                enable_gravity()
                if not real_time:
                    step_simulation()
                time.sleep(dt)
                current_conf = conf
        return current_conf

    # def full_path(self, q0=None):
    #     # TODO: could produce sequence of savers
    def refine(self, num_steps=0):
        return self.__class__(self.body, refine_path(self.body, self.joints, self.path, num_steps), self.digits, self.joints, self.attachments)
    def reverse(self):
        return self.__class__(self.body, self.path[::-1], self.digits, self.joints, self.attachments)
    def __repr__(self):
        return '{}({},{},{},{})'.format(self.__class__.__name__, self.body, len(self.joints), len(self.path), len(self.attachments))

class ApplyForce(object):
    def __init__(self, body, robot, link):
        self.body = body
        self.robot = robot
        self.link = link
    def bodies(self):
        return {self.body, self.robot}
    def iterator(self, **kwargs):
        return []
    def refine(self, **kwargs):
        return self
    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.robot, self.body)

class Attach(ApplyForce):
    def control(self, **kwargs):
        # TODO: store the constraint_id?
        add_fixed_constraint(self.body, self.robot, self.link)
    def reverse(self):
        return Detach(self.body, self.robot, self.link)

class Detach(ApplyForce):
    def control(self, **kwargs):
        remove_fixed_constraint(self.body, self.robot, self.link)
    def reverse(self):
        return Attach(self.body, self.robot, self.link)
    # def iterator(self): #New: open gripper when place pose is reached
    #     if 'ur5' in get_body_name(self.robot):
    #         open_gripper(self.robot)

class Command(object):
    def __init__(self, body_paths, args=[]):
        self.body_paths = body_paths
        self._args = args

        #NEW
        self.max_joints_conf = copy.deepcopy(RESET_JOINTS_CONF)
        self.min_joints_conf = copy.deepcopy(RESET_JOINTS_CONF)

    # def full_path(self, q0=None):
    #     if q0 is None:
    #         q0 = Conf(self.tree)
    #     new_path = [q0]
    #     for partial_path in self.body_paths:
    #         new_path += partial_path.full_path(new_path[-1])[1:]
    #     return new_path

    @property
    def args(self):
        return self._args

    def step_iter(self):
        for i, body_path in enumerate(self.body_paths):
            for step_i, conf in body_path.iterator():
                yield conf

    def step(self):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                msg = '{},{}) step?'.format(i, j)
                user_input(msg)
                #print(msg)
                #wait_for_interrupt()

    def execute(self, time_step=0.05, callback=None):
        current_conf = None
        for i, body_path in enumerate(self.body_paths):
            for j, conf in body_path.iterator():
                #time.sleep(time_step)
                if callback is not None:
                    callback()
                wait_for_duration(time_step)
                current_conf = conf


                #Modification for ur5 - saving min_max_position
                # if 'ur5' in get_body_name(body_path.body):    
                #     self.min_joints_conf, self.max_joints_conf = save_min_max_position(conf, self.min_joints_conf, self.max_joints_conf)

        # if 'ur5' in get_body_name(body_path.body):
        #     return current_conf, self.min_joints_conf, self.max_joints_conf
        # else:            


                #Modification for ur5
                if 'ur5' in get_body_name(body_path.body):    
                    self.min_joints_conf, self.max_joints_conf = save_min_max_position(conf, self.min_joints_conf, self.max_joints_conf)

        if 'ur5' in get_body_name(body_path.body):
            return current_conf, self.min_joints_conf, self.max_joints_conf
        else:            
            return current_conf


    def control(self, real_time=False, dt=0): # TODO: real_time
        for body_path in self.body_paths:
            body_path.control(real_time=real_time, dt=dt)

    def refine(self, **kwargs):
        return self.__class__([body_path.refine(**kwargs) for body_path in self.body_paths])

    def reverse(self):
        return self.__class__([body_path.reverse() for body_path in reversed(self.body_paths)])

    def __repr__(self):
        return 'c{}'.format(id(self) % 1000)


def get_grasp_gen(robot, grasp_name):
    grasp_info = GRASP_INFO[grasp_name]
    end_effector_link = link_from_name(robot, TOOL_FRAMES[get_body_name(robot)])
    
    def gen(body):
        grasp_poses = grasp_info.get_grasps(body)
        for grasp_pose in grasp_poses:
            body_grasp = BodyGrasp(body, grasp_pose, grasp_info.approach_pose,
                                   robot, end_effector_link)
            yield (body_grasp,)
    return gen


def get_stable_gen(fixed=[], bottom_percent=0.0, max_attempt=100): # TODO: continuous set of grasps
    def gen(body, surface):
        for _ in range(max_attempt):
            obstacles = [f for f in fixed if f != body]
            # pose = center_placement(body, surface, bottom_percent=bottom_percent)
            pose = sample_center_placement(body, surface, obstacles)
            if pose is None or any(pairwise_collision(body, b) for b in obstacles):
                continue
            body_pose = BodyPose(body, pose)
            yield (body_pose,)
            # TODO: check collisions
    return gen


def get_ik_fn(robot, digits, fixed=[], teleport=False, num_attempts=10, resolutions=None):
    movable_joints = get_movable_joints(robot)
    sample_fn = get_sample_fn(robot, movable_joints)
    def fn(body, pose, grasp):
        """
        Solves IK for a grasping-like motion
        :param body: body to be grasped
        :param pose: pose of the target body
        :param grasp: grasping pose in the target's frame
        :return: joint trajectory for a grasping motion
        """
        obstacles = fixed #[body] + 
        # get gripper pose in world frame
        gripper_pose = end_effector_from_body(pose.pose, grasp.grasp_pose)
        #draw_pose(gripper_pose)
        # get approach pose in world frame
        approach_pose = approach_from_grasp(grasp.approach_pose, gripper_pose)
        
        for _ in range(num_attempts):
            set_joint_positions(robot, movable_joints, sample_fn()) # Random seed
            # TODO: multiple attempts?

            #draw_pose(approach_pose)

            q_approach = inverse_kinematics(robot, grasp.link, approach_pose)
            if (q_approach is None) or any(pairwise_collision(robot, b) for b in obstacles):
                # if (q_approach is None):
                #     print("Error because q_approach is none")
                # else:
                #     print("Error for collision")
                continue
            # print("q_approach is not NONE!!!!!")
            conf = BodyConf(robot, q_approach)
            new_gripper_pose = grasp_pose_offset(robot, gripper_pose, GRIPPER_GRASPED_LIFT_HEIGHT)

            # draw_pose(gripper_pose)

            # draw_pose(gripper_pose)
            #draw_pose(new_gripper_pose)

            q_grasp = inverse_kinematics(robot, grasp.link, new_gripper_pose)
            if (q_grasp is None) or any(pairwise_collision(robot, b) for b in obstacles):
                if (q_grasp is None):
                    print("     Error q_grasp is None")
                if any(pairwise_collision(robot, b) for b in obstacles):
                    for b in obstacles:
                        pairwise_collision(robot, b)
                        print("     Error pairwise_collision with ", b)
                continue
            if teleport:
                path = [q_approach, q_grasp]
            else:
                conf.assign()
                #direction, _ = grasp.approach_pose
                #path = workspace_trajectory(robot, grasp.link, point_from_pose(approach_pose), -direction,
                #                                   quat_from_pose(approach_pose))
                path = plan_direct_joint_motion(
                    robot, conf.joints, q_grasp, obstacles=obstacles, resolutions=resolutions)
                if path is None:
                    if DEBUG_FAILURE: user_input('Approach motion failed')
                    continue
            command = Command([BodyPath(robot, path,  digits),
                               Attach(body, robot, grasp.link),
                               BodyPath(robot, path[::-1], digits, attachments=[grasp])])
            return (conf, command)
            # TODO: holding collisions
        return None
    return fn

def assign_fluent_state(fluents):
    obstacles = []
    for fluent in fluents:
        name, args = fluent[0], fluent[1:]
        if name == 'atpose':
            o, p = args
            obstacles.append(o)
            p.assign()
        else:
            raise ValueError(name)
    return obstacles


def get_free_motion_gen(robot, digits, fixed=[], teleport=False, resolutions=None):
    def fn(conf1, conf2, fluents=[]):
        assert ((conf1.body == conf2.body) and (conf1.joints == conf2.joints))
        if teleport:
            path = [conf1.configuration, conf2.configuration]
        else:
            conf1.assign()
            obstacles = fixed + assign_fluent_state(fluents)
            path = plan_joint_motion(
                robot, conf2.joints, conf2.configuration, obstacles=obstacles, resolutions=resolutions)
            if path is None:
                if DEBUG_FAILURE: user_input('Free motion failed')
                return None
        command = Command([BodyPath(robot, path, digits)])
        return (command,)
    return fn


def get_holding_motion_gen(robot, digits, fixed=[], teleport=False, resolutions=None):
    def fn(conf1, conf2, body, grasp, fluents=[]):
        assert ((conf1.body == conf2.body) and (conf1.joints == conf2.joints))
        if teleport:
            path = [conf1.configuration, conf2.configuration]
        else:
            conf1.assign()
            obstacles = [f for f in fixed if f != body]
            obstacles += assign_fluent_state(fluents)
            path = plan_joint_motion(robot, conf2.joints, conf2.configuration,
                                     obstacles=obstacles, attachments=[grasp.attachment()],
                                     resolutions=resolutions)
            if path is None:
                if DEBUG_FAILURE: user_input('Holding motion failed')
                return None
        command = Command([BodyPath(robot, path, digits, attachments=[grasp])])
        return (command,)
    return fn


def get_movable_collision_test():
    def test(command, body, pose):
        pose.assign()
        for path in command.body_paths:
            moving = path.bodies()
            if body in moving:
                # TODO: cannot collide with itself
                continue
            for _ in path.iterator():
                # TODO: could shuffle this
                if any(pairwise_collision(mov, body) for mov in moving):
                    if DEBUG_FAILURE: user_input('Movable collision')
                    return True
        return False
    return test
