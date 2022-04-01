# ENVIRONMENT TO TEST CONFIGURATION OF JOINTS FOR PICK AND PLACE TASK
# It also enquires the user to score the performance 1 => safe 0 1 => unsafe
# for further analysis of safe angles

from sre_constants import SUCCESS
from unittest import result
import init_path
import pybullet as p
import numpy as np
import time
import math
import argparse

from rpn.env_utils import URDFS
from rpn.env_utils import World, pb_session

from third_party.pybullet.utils.models.ur5.env import ClutteredPushGrasp
from third_party.pybullet.utils.models.ur5.utilities import setup_sisbot

from third_party.pybullet.utils.pybullet_tools.utils import WorldSaver, connect, dump_world, get_bodies_in_region, get_center_extent, get_pose, set_pose, Pose, \
    Point, set_camera, stable_z, create_box, create_cylinder, create_plane, HideOutput, load_model, \
    BLOCK_URDF, BLOCK1_URDF, get_configuration, SINK_URDF, STOVE_URDF, load_model, get_body_name, \
    disconnect, DRAKE_IIWA_URDF, PLATE_URDF, get_bodies, user_input, HideOutput, SHROOM_URDF, sample_placement, \
    get_movable_joints, pairwise_collision, stable_z, sample_placement_region, step_simulation, \
    get_lower_upper, get_custom_limits 

from third_party.pybullet.utils.pybullet_tools.utils import get_model_path

from rpn.problems_pb import kitchen_scene_ur5

from rpn.plan_utils import goal_to_motion_plan

from rpn.bullet_envs import TaskEnvCook

from rpn.bullet_task_utils import PBGoal

from third_party.pybullet.utils.pybullet_tools.kuka_primitives import save_min_max_position

# Create a child class from World Class to include 
# specific methods for UR5 robot

class Test_World(World):
    SIMULATION_STEP_DELAY = 1 / 450.
    DIFFERENCE_CONTROLLER = 0.01

    def __init__(self, object_types, objects=(), robot=None, scale=1):
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
        p.stepSimulation()
        time.sleep(self.SIMULATION_STEP_DELAY)

    def load_robot(self, urdf_path, robot_name):
        if robot_name == "drake":
            abs_path = get_model_path(urdf_path)
            self.robot = p.loadURDF(abs_path, useFixedBase=True)
        elif robot_name == "ur5":
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
            p.changeDynamics(self.robot, self.joints['left_inner_finger_pad_joint'].id, lateralFriction=0.5)
            p.changeDynamics(self.robot, self.joints['right_inner_finger_pad_joint'].id, lateralFriction=0.5)


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

        # time.sleep(0.1)
        # return l_x, l_y, r_x, r_y, y, a

    def move_cartesian_space(self, x, y, z, yaw, pitch, max_step=100, custom_velocity=None):
        real_xyz, real_xyzw = p.getLinkState(self.robot, self.eefID)[0:2] #Cartesian position of center of mass 
        desired_x, desired_y, desired_z = (real_xyz[0] + x, real_xyz[1] + y, real_xyz[2] + z)

        # set damping for robot arm and gripper
        jd = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        jd = jd * 0

        # angles
        roll = 0
        real_roll, real_pitch, real_yaw = p.getEulerFromQuaternion(real_xyzw)
        desired_roll, desired_pitch, desired_yaw = (real_roll, real_pitch, real_yaw + yaw)
        desired_yaw = np.clip(desired_yaw, -np.pi/2, np.pi/2)
        
        orn = p.getQuaternionFromEuler([desired_roll, desired_pitch, desired_yaw])
        
        for _ in range(max_step):
            # apply IK
            joint_poses = p.calculateInverseKinematics(self.robot, self.eefID, [desired_x, desired_y, desired_z],
                                                       orn, maxNumIterations=100, jointDamping=jd
                                                       )
            for i, name in enumerate(self.controlJoints[:-1]):  # Filter out the gripper
                joint = self.joints[name]
                pose = joint_poses[i]
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
            if check_contact and self.gripper_contact():
                # print(p.getJointState(self.robotID, self.joints['left_inner_finger_pad_joint'].id))
                # self.move_gripper(current_target_open_length - 0.005)
                # print(p.getJointState(self.robotID, self.joints['left_inner_finger_pad_joint'].id))
                # self.controlGripper(stop=True)
                return True
        return False



def test_env(args):
     
    scene = kitchen_scene_ur5()
    print(scene['ingredients'])
    ingredients = scene['ingredients']
    static_obj = ['table', 'stove', 'sink']
    #with HideOutput():
    print(scene['ingredients'])
    world = Test_World(static_obj + ingredients, scale=args.scale)
    #world = Teaching_World(['table', 'stove', 'sink', 'plate', 'pot', 'banana'], scale=3, control_type="Joy")

    
    
    world.load_object(URDFS['short_floor'], 'table', fixed=True, globalScaling=0.6)
    world.load_object(URDFS['stove'], 'stove', fixed=True, globalScaling=0.8)
    world.load_object(URDFS['sink'], 'sink', fixed=True)
    world.place(world.id('stove/0'), world.id('table/0'), [[0.4, 0, 0], [0, 0, 0, 1]])
    world.place(world.id('sink/0'), world.id('table/0'), [[-0.2, -0.4, 0], [0, 0, 0, 1]])
    
    world.load_robot(URDFS[args.robot], args.robot)#, globalScaling=0.6)
    if args.robot == "drake":
        set_pose(world.robot, [[-0.5, 0.2, 0.0], p.getQuaternionFromEuler([0, 0, 0])])
    elif args.robot == "ur5":
        robot_z = get_lower_upper(world.id('table/0'))[1][2] - get_lower_upper(world.robot,0)[0][2]
        set_pose(world.robot, [[-0.5, 0.2, robot_z], p.getQuaternionFromEuler([0, 0, 0])])
        world.reset_robot()
        world.open_gripper()
    
    
    poses = {
        "tomato": 
        [[-0.11704127490520477, 0.027094099670648575, 0.03935642167925835],
        [0.0, 0.0, 0.9892399823548468, -0.1463019388476535]],
        "pear": 
        [[-0.1451730728149414, -0.09236432611942291, 0.05021490156650543],
        [0.0, 0.0, -0.7849518496582935, 0.6195567719894793]],
        "orange": 
        [[-0.035095252096652985, -0.07544876635074615, 0.05651099979877472],
        [0.0, 0.0, -0.4088935248864854, 0.9125820978443009]],
        "banana": 
        [[-0.13826769590377808, 0.1415969431400299, 0.03197449818253517],
        [0.0, 0.0, -0.5637661935483798, 0.8259344277919227]],
    }

    for ingredient in ingredients[2:]:
        world.load_object(URDFS[ingredient], ingredient, fixed=False)
        world.random_place(world.id(ingredient+'/0'), world.id('table/0'), np.array([[-0.3, -0.2], [0.3, 0.2]]))
        # world.place(world.id(ingredient+'/0'), world.id('table/0'),np.array(poses[ingredient]))

    
    set_camera(180, -60, 1.2, Point()) #Original 180, -60, 1, Point()
    

    return world




def main():
    parser = argparse.ArgumentParser(description='Teaching environment configuration')
    parser.add_argument('--robot', default='ur5', type=str)
    parser.add_argument('--scale', default=1, type=float)
    parser.add_argument('--trials', default=1, type=int)
    parser.add_argument('--ing_num', default=0, type=int)
    args = parser.parse_args()

    results = {}
    conf_max_min = {}
    t = 0
    
    for trial in range(args.trials):
        # results[trial] = {}

        with pb_session(use_gui=True):
            world = test_env(args)
            env = TaskEnvCook(world)  #TaskEnvCook 

            sink_region = get_lower_upper(world.id('sink/0'))


            for obj_id in world.ids:
                print(obj_id, " ---> ", world.name(obj_id))
            
            ingredients = ["pear", "orange", "banana","tomato"]
            ingredients = [ingredients[args.ing_num]]  #For new world per ingredient

            if trial == 0:
                results[trial] = {}
                results[trial][ingredients[0]] = {"pick":0, "place":0, "error":0, "success":0}

            trial = 0
            for ingredient in ingredients:
                goal2Test = [PBGoal('on', True, ingredient+'/0', ingredient+'/0', 'sink/0', 'sink/0')]
                print("---------------- ", ingredient, " ----------------")
                # p.addUserDebugText(ingredient, [0,0,0.5], [1,0,0])
                conf_max_min[t] = {}
                if ingredient not in results[trial].keys():
                    results[trial][ingredient] = {"pick":0, "place":0, "error":0, "success":0}
                # try:
                for action_name, action_args, action_plan, ret in goal_to_motion_plan(goal2Test, env, False):
                    if action_name is None:
                        print("No plan for ", ingredient)
                        results[trial][ingredient]['error'] += 1
                        break

                    if action_name == 'pick':
                        print(action_name, env.objects.name(action_args[0]))
                        results[trial][ingredient][action_name] +=1#{action_name: 1}

                        conf_max_min[t]["pick"] = {"max": [0,0,0,0,0,0], \
                            "min": [0,0,0,0,0,0], \
                            "performance": 0}

                    elif action_name == 'place':
                        print(action_name, env.objects.name(action_args[1]))
                        results[trial][ingredient][action_name] += 1
                        conf_max_min[t]["place"] = {"max": [0,0,0,0,0,0], \
                            "min": [0,0,0,0,0,0], \
                            "performance": 0}
                    else:
                        print(action_name, action_args)
                    _, conf_min, conf_max = env.execute_command((action_name, action_args, action_plan), time_step=0.02)

                    if action_name == 'place' and 'ur5' in get_body_name(world.robot):
                        world.open_gripper()
                        
                    
                    conf_max_min[t][action_name]["min"], conf_max_min[t][action_name]["max"] = \
                        save_min_max_position(conf_min, conf_max_min[t][action_name]["min"], conf_max_min[t][action_name]["max"])
                    conf_max_min[t][action_name]["min"], conf_max_min[t][action_name]["max"] = \
                        save_min_max_position(conf_max, conf_max_min[t][action_name]["min"], conf_max_min[t][action_name]["max"])

                    performance = input("Performance: ")
                    while int(performance) not in [0, 1]:
                        performance = input("Performance: ")
                    conf_max_min[t][action_name]["performance"] = int(performance)
                    print(conf_max_min)
                    
                t += 1
                
                for item,_ in get_bodies_in_region(sink_region):
                    if item == world.id(ingredient+'/0'):
                        results[trial][ingredient]["success"] +=1
                        break

            p.stepSimulation()
            time.sleep(2)            

            print("------------------ ", trial)               
            print("Results: ", results)
            print("------------------") 

            print("------------------ ", trial)               
            print("Configuration: ", conf_max_min)
            print("------------------") 
            # except:
            #     continue


        
       
        


if __name__ == '__main__':
    main()