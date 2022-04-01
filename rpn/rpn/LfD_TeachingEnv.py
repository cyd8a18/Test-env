from cv2 import add
import init_path
import pybullet as p
import numpy as np
import time
import math
import argparse
import tacto

from rpn.env_utils import World, pb_session, URDFS
from rpn.problems_pb import kitchen_scene

from third_party.pybullet.utils.models.ur5.utilities import setup_sisbot

from third_party.pybullet.utils.pybullet_tools.utils import set_pose, Point, set_camera, step_simulation, \
    get_lower_upper, draw_pose 

from third_party.pybullet.utils.pybullet_tools.utils import get_model_path

# Create a child class from World Class to include 
# specific methods for UR5 robot

DIR_NAME = "/home/rl-gpis/RPN_UR5/rpn/third_party/pybullet/utils/"
OTHERS_URDF = {"sphere": "models/objects/sphere_small.urdf",
               "cube_small": "models/objects/cube_small.urdf"}

# Var to know mak and min grade for each control joint
control_joints_max = [0, 0, 0, 0, 0, 0]
control_joints_min = [0, 0, 0, 0, 0, 0]

class Teaching_World(World):
    SIMULATION_STEP_DELAY = 1 / 450.
    DIFFERENCE_CONTROLLER = 0.01

    def __init__(self, object_types, objects=(), robot=None, scale=1, control_type="keyboard"):
        World.__init__(self, object_types, objects=(), robot=None)
        self.scale = scale
        p.setGravity(0, 0, -9.8)
        # Setup some Limit
        self.gripper_open_limit = (0.0, 0.085)
        self.ee_position_limit = ((-0.224, 0.224),
                                  (-0.724, -0.276),
                                  (1.0, 1.3))       


    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        import pybulletX as px
        # run p.stepSimulation in another thread
        # t = px.utils.SimulationThread(real_time_factor=1.0)
        # t.start()
        p.stepSimulation()
        time.sleep(self.SIMULATION_STEP_DELAY)

    def load_robot(self, urdf_path, **kwargs):
        
        abs_path = get_model_path(urdf_path)
        self.robot = p.loadURDF(abs_path, useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)

        self.gripper_type='85'
        self.joints, self.controlGripper, self.controlJoints, self.mimicParentName =\
            setup_sisbot(self._robot, self.gripper_type) 
        self.eefID = 7  # ee_link -> End effector ID
        # Add force sensors
        p.enableJointForceTorqueSensor(self.robot, self.joints['left_inner_finger_pad_joint'].id)
        p.enableJointForceTorqueSensor(self.robot, self.joints['right_inner_finger_pad_joint'].id)
        # Change the friction of the gripper 
        p.changeDynamics(self.robot, self.joints['left_inner_finger_pad_joint'].id, lateralFriction=1)
        p.changeDynamics(self.robot, self.joints['right_inner_finger_pad_joint'].id, lateralFriction=1)

    def reset_robot(self):
        user_parameters = (-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                           -1.5707970583733368, 0.0009377758247187636, 0.085)
        
        for _ in range(100):
            for i, name in enumerate(self.controlJoints):
                if i == 6:
                    self.controlGripper(controlMode=p.POSITION_CONTROL, targetPosition=user_parameters[i])
                    break
                joint = self.joints[name]
                # control robot joints
                p.setJointMotorControl2(self.robot, joint.id, p.POSITION_CONTROL,
                                        targetPosition=user_parameters[i], force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity)
                step_simulation()

    # Loading tactile sensor
    def load_tacto(self):   
        links_ids = [self.joints['right_inner_finger_pad_joint'].id, self.joints['left_inner_finger_pad_joint'].id]

        CONF = "../third_party/config_digit_ur5.yml"
        #/home/rl-gpis/RPN_UR5/rpn/third_party/config_digit_ur5.yml

        self.digits = tacto.Sensor(width=120, height=160, visualize_gui=True, config_path=CONF)
        self.digits.add_camera(self.robot,  links_ids) 

    def keyboard_reading(self):
        x, y, z = (0, 0, 0)
        keys = p.getKeyboardEvents() 
        
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_WAS_TRIGGERED: 
            x = -0.01 * self.scale
        if p.B3G_LEFT_ARROW  in keys and keys[p.B3G_LEFT_ARROW]  & p.KEY_WAS_TRIGGERED: 
            x = 0.01 * self.scale
        if p.B3G_UP_ARROW    in keys and keys[p.B3G_UP_ARROW]    & p.KEY_WAS_TRIGGERED:
            y = -0.01 * self.scale
        if p.B3G_DOWN_ARROW  in keys and keys[p.B3G_DOWN_ARROW]  & p.KEY_WAS_TRIGGERED: 
            y = 0.01 * self.scale
        if ord('j') in keys and keys[ord('j')]&p.KEY_WAS_TRIGGERED: 
            z = -0.01 * self.scale
        if ord('u') in keys and keys[ord('u')]&p.KEY_WAS_TRIGGERED:  
            z = 0.01 * self.scale
        if ord('y') in keys and keys[ord('y')]&p.KEY_WAS_TRIGGERED:
            closed_gripper = self.close_gripper(check_contact=True)
            if closed_gripper:
                print('Grasped!')
        if ord('h') in keys and keys[ord('h')]&p.KEY_WAS_TRIGGERED:
            self.open_gripper()
        if ord('r') in keys and keys[ord('r')]&p.KEY_WAS_TRIGGERED:
            print("-------------------------")
            print("Max: ", control_joints_max)
            print("Min: ", control_joints_min)
            print("-------------------------")
        return x, y, z

    def move_cartesian_space(self, x, y, z, yaw, pitch, max_step=100, custom_velocity=None):
        real_xyz, real_xyzw = p.getLinkState(self.robot, self.eefID)[0:2] #Cartesian position of center of mass 
        desired_x, desired_y, desired_z = (real_xyz[0] + x, real_xyz[1] + y, real_xyz[2] + z)

        # angles
        roll = 0
        real_roll, real_pitch, real_yaw = p.getEulerFromQuaternion(real_xyzw)
        desired_roll, desired_pitch, desired_yaw = (real_roll, real_pitch, real_yaw + yaw)
        desired_yaw = np.clip(desired_yaw, -np.pi/2, np.pi/2)
        
        orn = p.getQuaternionFromEuler([desired_roll, desired_pitch, desired_yaw])
        
        for _ in range(max_step):
            # apply IK
            joint_poses = p.calculateInverseKinematics(self.robot, self.eefID, [desired_x, desired_y, desired_z],
                                                       orn, maxNumIterations=100)
            for i, name in enumerate(self.controlJoints[:-1]):  # Filter out the gripper
                joint = self.joints[name]
                pose = joint_poses[i]

                # Store max and min per joint
                if pose < control_joints_min[i]:
                    control_joints_min[i] = pose
                elif pose > control_joints_max[i]:
                    control_joints_max[i] = pose

                # control robot end-effector
                p.setJointMotorControl2(self.robot, joint.id, p.POSITION_CONTROL,
                                        targetPosition=pose, force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity if custom_velocity is None else custom_velocity * (i+1))

            step_simulation()

    def gripper_contact(self, bool_operator='and', force=100):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id

        contact_left = p.getContactPoints(bodyA=self.robot, linkIndexA=left_index)
        contact_right = p.getContactPoints(bodyA=self.robot, linkIndexA=right_index)
        
        if bool_operator == 'and' and not (contact_right and contact_left):
            return False

        # Check the force
        left_force = p.getJointState(self.robot, left_index)[2][:3]  # 6DOF, Torque is ignored
        right_force = p.getJointState(self.robot, right_index)[2][:3]
        left_norm, right_norm = np.linalg.norm(left_force), np.linalg.norm(right_force)
        print("norms: ", left_norm, "  ", right_norm)
        # print(left_norm, right_norm)
        if bool_operator == 'and':
            return left_norm > force and right_norm > force
        else:
            return left_norm > force or right_norm > force

    def move_gripper(self, gripper_opening_length: float, step: int = 120):
        gripper_opening_length = np.clip(gripper_opening_length, *self.gripper_open_limit)
        gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)  # angle calculation
        
        for _ in range(step):
            self.controlGripper(controlMode=p.POSITION_CONTROL, targetPosition=gripper_opening_angle)
            step_simulation()

    def open_gripper(self, step: int = 120):
        self.move_gripper(0.085, step)

    def close_gripper(self, step: int = 120, check_contact: bool = False) -> bool:
        # Get initial gripper open position
        initial_position = p.getJointState(self.robot, self.joints[self.mimicParentName].id)[0]
        initial_position = math.sin(0.715 - initial_position) * 0.1143 + 0.010
        for step_idx in range(1, step):
            current_target_open_length = initial_position - step_idx / step * initial_position

            self.move_gripper(current_target_open_length, 1)
            if current_target_open_length < 1e-5:
                return False

            # time.sleep(1 / 120)
            if check_contact and self.gripper_contact(force=2.5):
                # print(p.getJointState(self.robotID, self.joints['left_inner_finger_pad_joint'].id))
                # self.move_gripper(current_target_open_length - 0.005)
                # print(p.getJointState(self.robotID, self.joints['left_inner_finger_pad_joint'].id))
                # self.controlGripper(stop=True)
                return True
        return False



def teaching_env(args):
     
    scene = kitchen_scene()
    ingredients = scene['ingredients'] + ['mug']
    static_obj = ['table', 'stove', 'sink']
    additional_obj = ['sphere', 'cube_small']
    #with HideOutput():
    #print(scene['ingredients'])
    world = Teaching_World(static_obj + ingredients + additional_obj, scale=args.scale, control_type=args.device)
    #world = Teaching_World(['table', 'stove', 'sink', 'plate', 'pot', 'banana'], scale=3, control_type="Joy")

    world.load_robot(URDFS['ur5'])#, globalScaling=0.6)
    world.load_tacto()
    
    world.load_object(URDFS['short_floor'], 'table', fixed=True, globalScaling=0.6)
    world.load_object(URDFS['stove'], 'stove', fixed=True, globalScaling=0.8)
    world.load_object(URDFS['sink'], 'sink', fixed=True)
    world.place(world.id('stove/0'), world.id('table/0'), [[0.4, 0, 0], [0, 0, 0, 1]])
    world.place(world.id('sink/0'), world.id('table/0'), [[-0.2, -0.4, 0], [0, 0, 0, 1]])

    robot_z = get_lower_upper(world.id('table/0'))[1][2] - get_lower_upper(world.robot,0)[0][2]
    set_pose(world.robot, [[-0.5, 0.5, robot_z], p.getQuaternionFromEuler([0, 0, 0])])
    
    for ingredient in ingredients:
        world.load_object(URDFS[ingredient], ingredient, fixed=False)
        world.random_place(world.id(ingredient+'/0'), world.id('table/0'), np.array([[-0.2, -0.1], [0.05, 0.4]]))

        # Loading objects in sensor tactile sensor
        world.digits.add_object(DIR_NAME + URDFS[ingredient], world.id(ingredient+'/0')) 

    #Loading additional objects
    for obj in additional_obj:
        world.load_object(OTHERS_URDF[obj], obj, fixed=False)
        if obj == 'cube_small':
            set_pose(world.id(obj+'/0'), [[-0.4, 0, 0.5], p.getQuaternionFromEuler([0, 0, -0.7854])])
        else:
            world.random_place(world.id(obj+'/0'), world.id('table/0'), np.array([[-0.2, -0.1], [0.05, 0.4]]))

        # Loading objects in sensor tactile sensor
        world.digits.add_object(DIR_NAME + OTHERS_URDF[obj], world.id(obj+'/0'))
    
    
    set_camera(180, -60, 1.2, Point()) #Original 180, -60, 1, Point()
    world.reset_robot()
    return world


def main():
    parser = argparse.ArgumentParser(description='Teaching environment configuration')
    parser.add_argument('--device', default='keyboard', type=str)
    parser.add_argument('--scale', default=1, type=float)
    args = parser.parse_args()

    with pb_session(use_gui=True):
        # World initialization
        world = teaching_env(args)
        
        # Tactile sensor settings
        np.set_printoptions(suppress=True)

        # # Estimating sensor pose
        # adjust_angle = math.pi / 2
        # pos, ori = world.digits.cameras["cam0"].get_pose()
        # ori = (ori[0], ori[1], ori[2] + adjust_angle)
        # ori = p.getQuaternionFromEuler(ori)
        # draw_pose((pos, ori))
        pos, ori = world.digits.cameras["cam0"].get_pose()
        ori = p.getQuaternionFromEuler(ori)
        draw_pose((pos, ori))
        pos, ori = world.digits.cameras["cam1"].get_pose()
        ori = p.getQuaternionFromEuler(ori)
        draw_pose((pos, ori))

        # Control
        
        while True:
            x, y, z = world.keyboard_reading()
            #x, y, z = world.control_reading()
            world.move_cartesian_space(x,y,z, yaw = 0, pitch = 0)
            
            # Tactile sensor settings
            color, depth = world.digits.render()
            world.digits.updateGUI(color, depth)
            time.sleep(0.1)
        


if __name__ == '__main__':
    main()