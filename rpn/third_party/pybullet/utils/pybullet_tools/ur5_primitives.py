import time
import math

import pybullet as p
import numpy as np

from .pr2_utils import get_top_grasps
from .utils import get_pose, set_pose, get_movable_joints, get_configuration, \
    set_joint_positions, add_fixed_constraint, enable_real_time, disable_real_time, joint_controller, \
    enable_gravity, get_refine_fn, user_input, wait_for_duration, link_from_name, get_body_name, sample_placement, \
    end_effector_from_body, approach_from_grasp, plan_joint_motion, GraspInfo, Pose, INF, Point, \
    inverse_kinematics, pairwise_collision, remove_fixed_constraint, Attachment, get_sample_fn, \
    step_simulation, refine_path, plan_direct_joint_motion, center_placement, sample_center_placement

# PHYSICS = False

# GRASP_INFO = {
#     'top': GraspInfo(lambda body: get_top_grasps(body, under=True, tool_pose=Pose(),
#                                                  max_width=INF,  grasp_length=0),
#                      Pose(0.1*Point(z=1))),
# }
# TOOL_FRAMES = {
#     'iiwa14': 'iiwa_link_ee_kuka', # iiwa_link_ee
#     'cube_gripper': 'ee_link',
#     'wsg50_with_gripper': 'base_link',
# }

GRIPPER_GRASPED_LIFT_HEIGHT = 1.4 / 10
APPROACH_POSE_OFFSET = 0.2
GRIPPER_MAX_WIDTH = 0.14
GRIPPER_OPEN_LIMIT = (0.0, 0.085)
EE_POSITION_LIMIT = ((-0.224, 0.224),(-0.724, -0.276),(1.0, 1.3))

def reset_robot(world, robot):
    user_parameters = (-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                        -1.5707970583733368, 0.0009377758247187636, 0.085)
    
    for _ in range(100):
        for i, name in enumerate(world.controlJoints):
            if i == 6:
                world.controlGripper(controlMode=p.POSITION_CONTROL, targetPosition=user_parameters[i])
                break
            joint = world.joints[name]
            # control robot joints
            p.setJointMotorControl2(world.robot, joint.id, p.POSITION_CONTROL,
                                    targetPosition=user_parameters[i], force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)

            step_simulation()

def gripper_contact(robot, joints, bool_operator='and', force=100):
    left_index = joints['left_inner_finger_pad_joint'].id
    right_index = joints['right_inner_finger_pad_joint'].id

    contact_left = p.getContactPoints(bodyA=robot, linkIndexA=left_index)
    contact_right = p.getContactPoints(bodyA=robot, linkIndexA=right_index)
    
    if bool_operator == 'and' and not (contact_right and contact_left):
        return False

    # Check the force
    left_force = p.getJointState(robot, left_index)[2][:3]  # 6DOF, Torque is ignored
    right_force = p.getJointState(robot, right_index)[2][:3]
    left_norm, right_norm = np.linalg.norm(left_force), np.linalg.norm(right_force)
    #print("norms: ", left_norm, "  ", right_norm)
   
    if bool_operator == 'and':
        return left_norm > force and right_norm > force
    else:
        return left_norm > force or right_norm > force

def move_gripper(controlGripper, digits, gripper_opening_length: float, step: int = 120):
    gripper_opening_length = np.clip(gripper_opening_length, *GRIPPER_OPEN_LIMIT)
    gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)  # angle calculation
    # print("angle: ", gripper_opening_angle)
    for _ in range(step):
        controlGripper(controlMode=p.POSITION_CONTROL, targetPosition=gripper_opening_angle)
        # Tactile sensor
        color, depth = digits.render()
        digits.updateGUI(color, depth)
        time.sleep(0.1)
        # --------------
        step_simulation()

def open_gripper(controlGripper, step: int = 120):
    move_gripper(controlGripper, 0.085, step)

def close_gripper(robot, joints, controlGripper, controlJoints, mimicParentName, digits, step: int = 120, check_contact: bool = False) -> bool:
    # Get initial gripper open position
    initial_position = p.getJointState(robot, joints[mimicParentName].id)[0]
    initial_position = math.sin(0.715 - initial_position) * 0.1143 + 0.010

    still_open_flag_ = True
    for step_idx in range(1, step):
        current_target_open_length = initial_position - step_idx / step * initial_position

        if current_target_open_length < 1e-5:
            return False

        # time.sleep(1 / 120)
        if check_contact and gripper_contact(robot, joints):
            # print(p.getJointState(self.robotID, self.joints['left_inner_finger_pad_joint'].id))
            # self.move_gripper(current_target_open_length - 0.005)
            # print(p.getJointState(self.robotID, self.joints['left_inner_finger_pad_joint'].id))
            # self.controlGripper(stop=True)
            return True

        move_gripper(controlGripper, digits, current_target_open_length, 1)
    return False

def control_joints(robot, joints, controlJoints, joint_poses, custom_velocity=None):
    for i, name in enumerate(controlJoints[:-1]):  # Filter out the gripper
        joint = joints[name]
        pose = joint_poses[i]
        # control robot end-effector
        p.setJointMotorControl2(robot, joint.id, p.POSITION_CONTROL,
                                        targetPosition=pose, force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity if custom_velocity is None else custom_velocity * (i+1))
        step_simulation()