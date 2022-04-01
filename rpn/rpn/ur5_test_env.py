from cv2 import add
import init_path
import pybullet as p
import numpy as np
import time
import math
import argparse
import tacto

from rpn.env_utils import URDFS
from rpn.env_utils import World, pb_session

#from third_party.pybullet.utils.models.ur5.env import ClutteredPushGrasp
#from third_party.pybullet.utils.models.ur5.utilities import setup_sisbot, setup_sisbot_force

from third_party.pybullet.utils.models.pybullet_ur5_robotiq.robot import UR5Robotiq85

from third_party.pybullet.utils.pybullet_tools.utils import WorldSaver, connect, control_joint, dump_world, get_center_extent, get_pose, set_pose, Pose, \
    Point, set_camera, stable_z, create_box, create_cylinder, create_plane, HideOutput, load_model, \
    BLOCK_URDF, BLOCK1_URDF, SMALL_BLOCK_URDF, get_configuration, SINK_URDF, STOVE_URDF, load_model, get_body_name, \
    disconnect, DRAKE_IIWA_URDF, PLATE_URDF, get_bodies, user_input, HideOutput, SHROOM_URDF, sample_placement, \
    get_movable_joints, pairwise_collision, stable_z, sample_placement_region, step_simulation, \
    get_lower_upper, draw_pose 

from third_party.pybullet.utils.pybullet_tools.utils import get_model_path

from rpn.problems_pb import kitchen_scene

from third_party.control_joy import JoyController
import pygame
from pygame.locals import *

DIR_NAME = "/home/rl-gpis/RPN_UR5/rpn/third_party/pybullet/utils/"
OTHERS_URDF = {"sphere": "models/objects/sphere_small.urdf",
               "cube_small": "models/objects/cube_small.urdf"}

class Teaching_World(World):
    SIMULATION_STEP_DELAY = 1 / 240.
    
    def __init__(self, object_types, objects=(), robot=None, scale=1, control_type="keyboard", robot_class=None):
        World.__init__(self, object_types, objects=(), robot=None)
        self.scale = scale
        self.robot_class = robot_class
        p.setGravity(0, 0, -9.8)

        # Controller initialization
        if control_type == "controller":
            pygame.init()
            self.controller = JoyController(0)
    
    def load_robot(self, urdf_path, **kwargs):
        self.robot_class.load()
        self.robot = self.robot_class.id
        self.robot_class.add_force_sensors()

    def keyboard_reading(self, gripper_length):
        x, y, z = (0, 0, 0)
        roll, pitch, yaw = (0, 0, 0)
        keys = p.getKeyboardEvents() 
        #gripper_length = self.robot_class.gripper_range[1]
        
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
            gripper_length = self.robot_class.gripper_range[0]
            #self.robot_class.close_gripper(force=25)
            # if closed_gripper:
            #     print('Grasped!')
        if ord('h') in keys and keys[ord('h')]&p.KEY_WAS_TRIGGERED:
            # self.open_gripper()
            gripper_length = self.robot_class.gripper_range[1]
            

        return x, y, z, roll, pitch, yaw, gripper_length

    def control_reading(self):
        eventlist = pygame.event.get()#[pygame.event.poll()]
        l_x, l_y, r_x, r_y, a_b, y_b, x_b, b_b, cross_key = self.controller.get_controller_value(eventlist)
        x, y, z = (0, 0, 0)
        roll, pitch, yaw = (0, 0, 0)
        
        if l_x < -0.5: 
            x = 0.01 * self.scale
        if l_x > 0.5: 
            x = -0.01 * self.scale
        if l_y < -0.5: 
            y = -0.01 * self.scale
        if l_y > 0.5: 
            y = 0.01 * self.scale
        if a_b == 1: 
            z = -0.01 * self.scale
        if y_b == 1:  
            z = 0.01 * self.scale
        if b_b == 1: 
            gripper_length = self.robot_class.gripper_range[1]
            # if closed_gripper:
            #     print('Grasped!')
        if x_b == 1: 
            # self.open_gripper()
            gripper_length = self.robot_class.gripper_range[0]
        if cross_key == (-1, 0):
            yaw = 0.1 * self.scale
        if cross_key == (1, 0):
            yaw = -0.1 * self.scale
        if cross_key == (0, 1):
            pitch = 0.1 * self.scale
        if cross_key == (0, -1):
            pitch = -0.1 * self.scale    
            
        return x, y, z, roll, yaw, pitch, gripper_length

    def move_cartesian_space(self, x, y, z, roll, pitch, yaw, gripper_length): #max_step=100, custom_velocity=None):
        real_xyz, real_xyzw = p.getLinkState(self.robot, self.robot_class.eef_id)[0:2] #Cartesian position of center of mass 
        
        desired_x, desired_y, desired_z = (real_xyz[0] + x, real_xyz[1] + y, real_xyz[2] + z)
        
        # angles
        roll = 0
        real_roll, real_pitch, real_yaw = p.getEulerFromQuaternion(real_xyzw)
        desired_roll, desired_pitch, desired_yaw = (real_roll, real_pitch, real_yaw)
        #desired_yaw = np.clip(desired_yaw, -np.pi/2, np.pi/2)

        return desired_x, desired_y, desired_z, desired_roll, desired_pitch, desired_yaw, gripper_length

    def step(self, action, control_method='joint'):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        assert control_method in ('joint', 'end')
        self.robot_class.move_ee(action[:-1], control_method)
        if action[-1] == self.robot_class.gripper_range[0]:
            close_result = self.robot_class.close_gripper(force=2)
            if close_result:
                print('Grasped!')
        else:
            self.robot_class.move_gripper(action[-1], 2, True)
        

    def reset_robot(self):
        #self.robot_class.reset()
        self.robot_class.reset_arm()
        self.robot_class.reset_gripper()
        # Wait for a few steps
        for _ in range(250):
            p.stepSimulation()
            #time.sleep(0.05)
            

def teaching_env(args):
     
    scene = kitchen_scene()
    ingredients = scene['ingredients'] + ['mug']
    static_obj = ['table', 'stove', 'sink']
    additional_obj = ['sphere', 'cube_small']
    #with HideOutput():
    #print(scene['ingredients'])
    world = Teaching_World(static_obj + ingredients + additional_obj, scale=args.scale, \
        control_type=args.device, robot_class = UR5Robotiq85((0, 0.5, 0), (0, 0, 0)))
    #world = Teaching_World(['table', 'stove', 'sink', 'plate', 'pot', 'banana'], scale=3, control_type="Joy")

    world.load_robot(URDFS['ur5'])#, globalScaling=0.6)
    #world.robot.load()
    # world.load_tacto()
    
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
        # world.digits.add_object(DIR_NAME + URDFS[ingredient], world.id(ingredient+'/0')) 

    #Loading additional objects
    for obj in additional_obj:
        world.load_object(OTHERS_URDF[obj], obj, fixed=False)
        if obj == 'cube_small':
            set_pose(world.id(obj+'/0'), [[-0.4, 0, robot_z], p.getQuaternionFromEuler([0, 0, -0.7854])])
        else:
            world.random_place(world.id(obj+'/0'), world.id('table/0'), np.array([[-0.2, -0.1], [0.05, 0.4]]))

        # Loading objects in sensor tactile sensor
        # world.digits.add_object(DIR_NAME + OTHERS_URDF[obj], world.id(obj+'/0'))
    
    
    set_camera(180, -60, 1.2, Point()) #Original 180, -60, 1, Point()
    #world.robot_class.reset()
    
    world.reset_robot()
    #world.robot_class.reset()

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
        # np.set_printoptions(suppress=True)

        # # Estimating sensor pose
        # adjust_angle = math.pi / 2
        # pos, ori = world.digits.cameras["cam0"].get_pose()
        # ori = (ori[0], ori[1], ori[2] + adjust_angle)
        # ori = p.getQuaternionFromEuler(ori)
        # draw_pose((pos, ori))
        # pos, ori = world.digits.cameras["cam0"].get_pose()
        # ori = p.getQuaternionFromEuler(ori)
        # draw_pose((pos, ori))
        # pos, ori = world.digits.cameras["cam1"].get_pose()
        # ori = p.getQuaternionFromEuler(ori)
        # draw_pose((pos, ori))

        gripper_length = world.robot_class.gripper_range[1] #initial state of gripper => open
        # Control
        if args.device == "keyboard":
            while True:
                # x, y, z, roll, yaw, pitch, gripper_length = world.keyboard_reading()
                x, y, z, roll, yaw, pitch, gripper_length = world.keyboard_reading(gripper_length)
                action = world.move_cartesian_space(x, y, z, roll, yaw, pitch, gripper_length)
                world.step(action, 'end')
                
                #x, y, z = world.control_reading()
                #world.move_cartesian_space(x,y,z, yaw = 0, pitch = 0)
                #print(world.control_reading())
                #time.sleep(0.1)
                # if world.keyboard_reading() != (0,0,0):
                #     print(world.keyboard_reading())
                
                # Tactile sensor settings
                # color, depth = world.digits.render()
                # world.digits.updateGUI(color, depth)
                # time.sleep(0.1)

        elif args.device == "controller":
            while True:
                x, y, z, yaw, pitch = world.control_reading()
                world.move_cartesian_space(x,y,z, yaw, pitch)
                #print(world.control_reading())
                #time.sleep(0.1)
                # if world.keyboard_reading() != (0,0,0):
                #     print(world.keyboard_reading())

                # Tactile sensor settings
                # color, depth = world.digits.render()
                # world.digits.updateGUI(color, depth)
                # time.sleep(0.1)

        #input()
        
        # while True:
        # for _ in range(1):
        #     step_simulation()
        # input()
        


if __name__ == '__main__':
    main()
    