import numpy as np
import pdb
import cv2
import time
import socketio
import argparse
import pyvirtualcam
import csv
import json_numpy
import yaml
from scipy.spatial.transform import Rotation
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.ui import UI
from tdw.quaternion_utils import QuaternionUtils
from tdw.output_data import OutputData, Images, ScreenPosition, Transforms, Raycast, Keyboard as KBoard, SegmentationColors, Framerate
from tdw.add_ons.keyboard import Keyboard
from magnebot import Magnebot, Arm, ActionStatus, ImageFrequency
from magnebot.util import get_default_post_processing_commands

from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.add_ons.logger import Logger
from base64 import b64encode

from PIL import Image, ImageOps

import datetime
import json
import os
import random
import sys
from subprocess import Popen
from enum import Enum
import math
import string
from collections import deque



#Dimension of our camera view
width = 640 
height = 480 

num_users = 2
num_ais = 1

cams = []
video = []
video_meta_f = -1
global_refresh_sensor = 0

address = ''
dateTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

extra_commands = []
duration = []

game_finished = 0

# Given three collinear points p, q, r, the function checks if 
# point q lies on line segment 'pr' 
def onSegment(p, q, r):
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and 
           (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False
  
def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
      
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ 
    # for details of below formula. 
      
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
    if (val > 0):
          
        # Clockwise orientation
        return 1
    elif (val < 0):
          
        # Counterclockwise orientation
        return 2
    else:
          
        # Collinear orientation
        return 0
  
# The main function that returns true if 
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1,q1,p2,q2):
      
    # Find the 4 orientations required for 
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
  
    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True
  
    # Special Cases
  
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True
  
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True
  
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True
  
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True
  
    # If none of the cases
    return False
    

class Stats():
    
    def __init__(self):
        self.distance_traveled = 0
        self.grabbed_objects = 0
        self.grab_attempts = 0
        self.dropped_outside_goal = []
        self.objects_sensed = [] #Maybe []
        self.sensor_activation = 0
        self.objects_in_goal = []
        self.dangerous_objects_in_goal = []
        self.num_messages_sent = 0
        self.average_message_length = 0
        self.failed = 0
        self.time_with_teammates = {}
        self.end_time = 0
        self.team_dangerous_objects_in_goal = 0
        self.total_dangerous_objects = 0
        self.quality_work = 0
        self.effort = 0
        self.human_team_effort = 0
        self.team_end_time = 0
        self.team_failure_reasons = {}
        self.team_quality_work = 0
        self.team_speed_work = 0
        self.team_achievement = 0
        self.team_payment = 0
        self.individual_payment = 0
        self.token = ""
        
agents_good_sensors = []
#This class inherits the magnebot class, we just add a number of attributes over it
class Enhanced_Magnebot(Magnebot):

    def sensor_paramenter_lower_threshold(self):
    
        low_threshold = 0.5
        high_threshold = 0.9
    
        good_sensor = agents_good_sensors.pop(0)
    
        if self.difficulty_level:
            if self.difficulty_level == 1:
                if good_sensor == 1:
                    low_threshold = [0.8,0.8] #0.9 #0.7
                    high_threshold = [0.85,0.85] #0.95
                elif good_sensor == 2:
                    low_threshold = [0.8,0.75] 
                    high_threshold = [0.85,0.8]
                elif good_sensor == 3:
                    low_threshold = [0.75,0.8] 
                    high_threshold = [0.7,0.85]
                elif good_sensor == 4:
                    low_threshold = [0.8,0.8] 
                    high_threshold = [0.85,0.85]
                else:
                    low_threshold = 0.7 #0.7
                    high_threshold = 0.75
                
                low_threshold = [0.8,0.8] #0.9 #0.7
                high_threshold = [0.85,0.85] 
            elif self.difficulty_level == 2:
                low_threshold = 0.7
            elif self.difficulty_level == 3:
                low_threshold = 0.6
                
        return low_threshold,high_threshold

    def __init__(self,robot_id, position, controlled_by, difficulty_level, key_set=None,image_frequency=ImageFrequency.never,pass_masks=['_img'],strength=1, check_version=False):
        super().__init__(robot_id=robot_id, position=position,image_frequency=image_frequency,check_version=check_version)
        self.key_set = key_set
        self.ui = []
        self.ui_elements = {}
        self.strength = strength
        self.danger_estimates = []
        self.company = {}
        self.controlled_by = controlled_by
        self.focus_object = ""
        self.item_info = {}
        self.screen_positions = {"position_ids":[],"positions":[],"duration":[]}
        self.refresh_sensor = global_refresh_sensor
        self.messages = []
        self.grasping = False
        self.grasping_time = 0
        self.past_status = ActionStatus.ongoing
        self.view_radius = 0
        self.visibility_matrix = []
        self.centered_view = 0
        self.resetting_arm = False
        self.key_pressed = ''
        self.disabled = False
        self.last_output = False
        self.last_position = np.array([])
        self.stats = Stats()
        self.skip_frames = 0
        self.difficulty_level = difficulty_level
        low_thresh,high_thresh = self.sensor_paramenter_lower_threshold()
        self.p11 = float(random.uniform(low_thresh[0], high_thresh[0])) #Binary channel
        self.p22 = float(random.uniform(low_thresh[1], high_thresh[1]))
        print("Sensors", self.p11,self.p22)
        self.current_teammates = {}
        self.reported_objects = []
        
        
    def reset(self,position):
        super().reset(position=position)
        self.resetting_arm = False
        self.past_status = ActionStatus.ongoing
        self.messages = []
        self.grasping = False
        self.grasping_time = 0
        self.screen_positions = {"position_ids":[],"positions":[],"duration":[]}
        self.focus_object = ""
        self.item_info = {}
        self.company = {}
        self.disabled = False
        self.last_output = False
        self.stats = Stats()
        self.last_position = np.array([])
        low_thresh,high_thresh = self.sensor_paramenter_lower_threshold()
        self.p11 = float(random.uniform(low_thresh[0], high_thresh[0])) #Binary channel
        self.p22 = float(random.uniform(low_thresh[1], high_thresh[1]))
        print("Sensors",self.p11,self.p22)
        self.current_teammates = {}
        self.reported_objects = []
        self.danger_estimates = []
        
    def partial_reset(self,position):
        super().reset(position=position)
        self.resetting_arm = False
        self.past_status = ActionStatus.ongoing
        self.messages = []
        self.grasping = False
        self.grasping_time = 0
        self.screen_positions = {"position_ids":[],"positions":[],"duration":[]}
        self.focus_object = ""
        self.danger_estimates = []

#Main class
class Simulation(Controller):
  

    def __init__(self, args, cfg, port: int = 1071, check_version: bool = False, launch_build: bool = False, restart=False):
    
        super().__init__(port=port, check_version=check_version, launch_build=launch_build, restart=restart)


         
        self.keys_set = []
        self.local = args.local
        self.options = args
        self.cfg = cfg
        self.no_debug_camera = args.no_debug_camera
        
        self.reset = False
        self.reset_partial = False
        self.reset_message = False
        
        self.timer = 0 #time.time()
        self.real_timer = time.time()
        self.timer_start = self.timer
        self.reset_number = 0
        self.payment = 7
        self.max_time_unheld = 1
        self.enable_logs = False
        self.timer_limit = 0#float(self.cfg['timer'])
        self.waiting = False
        self.rooms = {}
        self.visibility_matrix = {}

        """
        if float(self.cfg['timer']) > 0:
            self.timer_limit = self.timer_start + float(self.cfg['timer'])
        else:
            self.timer_limit = 0
        """
        
        self.ai_skip_frames = int(self.cfg['ai_skip_frames'])
        
        
        self.segmentation_colors = {}
        self.scenario_size = 20
        self.wall_length = 6

        
        self.scenario = self.options.scenario
        
        if self.options.seed > -1:
            self.seed_value = self.options.seed
        else:
            self.seed_value = random.randrange(sys.maxsize)
            
        random.seed(self.seed_value)
        print("SEED:", self.seed_value)
        

        #Functionality of keys according to order of appearance: [Advance, Back, Right, Left, Grab with left arm, Grab with right arm, Camera down, Camera up, Activate sensor, Focus on object]
        self.proposed_key_sets = []
        with open('keysets.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for r_idx, row in enumerate(csv_reader):
                if r_idx > 0:      
                    self.proposed_key_sets.append(row)

        #proposed_key_sets = [["UpArrow","DownArrow","RightArrow","LeftArrow","Z","X","C","V","B","N"],["W","S","D","A","H","J","K","L","G","F"],["Alpha5","R","E","Y","U","I","O","P","Alpha0","Alpha9"]]

        

        #Creating occupancy map
        self.static_occupancy_map = OccupancyMap()       

        
        logger = Logger(path="tdw_log.txt", overwrite=True)

        #Add-ons
        self.add_ons.extend([self.static_occupancy_map, logger])


        # Create the scene.
        

        commands = self.create_scene()
        

        
        print("Creating scene")
        self.communicate(commands)
        print("Created scene")
        
        self.static_occupancy_map.generate(cell_size=self.cfg['cell_size']) #Get occupancy map only with walls
        
        self.communicate([])
        
        #self.visibility_matrix[6] = self.label_components(self.static_occupancy_map.occupancy_map, 6)

        out_of_bounds = np.where(self.static_occupancy_map.occupancy_map == 2)
        self.static_occupancy_map.occupancy_map[out_of_bounds[0],out_of_bounds[1]] = -1
        
        out_of_bounds = np.where(self.static_occupancy_map.occupancy_map == 3)
        self.static_occupancy_map.occupancy_map[out_of_bounds[0],out_of_bounds[1]] = 0
        
        self.manual_occupancy_map()
        
        print(self.static_occupancy_map.occupancy_map)
        #print(self.static_occupancy_map.occupancy_map[:20,:20])
        
        
        self.ai_original_spawn_positions = []#[{"x": -2, "y": 0, "z": 1.1},{"x": -2, "y": 0, "z": 2.1}, {"x": -2, "y": 0, "z": 3.1}, {"x": -3, "y": 0, "z": 0.1}, {"x": -2, "y": 0, "z": 0.1},{"x": -2, "y": 0, "z": -1.1}, {"x": -2, "y": 0, "z": -2.1},{"x": -2, "y": 0, "z": -3.1},{"x": -3, "y": 0, "z": -1.1},{"x": -3, "y": 0, "z": -2.1}, {"x": -3, "y": 0, "z": 1.1}, {"x": -3, "y": 0, "z": 2.1}, {"x": -3.5, "y": 0, "z": 0.5}, {"x": -3.5, "y": 0, "z": 1.5}, {"x": -3.5, "y": 0, "z": 2.5}, {"x": -3.5, "y": 0, "z": 3.5}, {"x": -3.5, "y": 0, "z": -2.5}, {"x": -3.5, "y": 0, "z": -3.5}]
        self.user_original_spawn_positions = []#[{"x": 0, "y": 0, "z": 1.1},{"x": 0, "y": 0, "z": 2.1}, {"x": 0, "y": 0, "z": 3.1}, {"x": 1, "y": 0, "z": 0.1}, {"x": 0, "y": 0, "z": 0.1},{"x": 0, "y": 0, "z": -1.1}, {"x": 0, "y": 0, "z": -2.1},{"x": 0, "y": 0, "z": -3.1},{"x": 1, "y": 0, "z": -3.1},{"x": 1, "y": 0, "z": -2.1}]
        
        locations = []
        if self.scenario != 3:
            for a in range(-5,5):
                for b in range(-5,5):
                    locations.append({"x": self.cfg['cell_size']/2 + self.cfg['cell_size']*a,  "y": 0, "z": self.cfg['cell_size']/2 + self.cfg['cell_size']*b})
                    
        else:
            for a in range(int(self.goal_area[0][1][0] - self.goal_area[0][0]), int(self.goal_area[0][1][0] + self.goal_area[0][0])):
                for b in range(int(self.goal_area[0][1][1] - self.goal_area[0][0]), int(self.goal_area[0][1][1] + self.goal_area[0][0])):
                    locations.append({"x": self.cfg['cell_size']/2 + self.cfg['cell_size']*a,  "y": 0, "z": self.cfg['cell_size']/2 + self.cfg['cell_size']*b})
        
        #print('LOCATIONS', locations)
        for l_idx,spawn_loc in enumerate(locations):
            
            if l_idx > len(locations)/2:
                self.ai_original_spawn_positions.append(spawn_loc)
            else:
                self.user_original_spawn_positions.append(spawn_loc)
                
        self.create_agents()
        print("Created agents")
        self.object_manager: ObjectManager = ObjectManager()
        self.add_ons.append(self.object_manager)
        
        commands = self.populate_world()
        
        
        self.communicate(commands)
        print("Populated world")

        if self.options.log_state:
            self.log_init_data()
        

        
        self.segmentation_colors = self.get_segmentation_colors()


        

        self.communicate({"$type": "set_post_process", "value": False})
        #self.communicate({"$type": "set_render_quality", "render_quality": 0})
        #self.communicate({"$type": "enable_reflection_probes", "enable": False})


        
        #print(self.static_occupancy_map.occupancy_map)


        #Initializing communication with server
        
        self.just_started = False

        if not restart:

            self.sio = None

            #Socket io event functions
            if not self.local:
                self.sio = socketio.Client(ssl_verify=False)
                
                @self.sio.event
                def connect():
                    print("I'm connected!")
                    
                    self.send_init_data()

                @self.sio.event
                def connect_error(data):
                    print("The connection failed!")

                @self.sio.event
                def disconnect():
                    print("I'm disconnected!")
                    
                @self.sio.event
                def set_goal(agent_id,obj_id):
                    print("Received new goal")
                    self.target[agent_id] = obj_id
                    
                """
                @self.sio.event
                def ai_message(message, source_agent_id, agent_id):
                    ai_magnebot = self.ai_magnebots[self.ai_magnebots_ids.index(agent_id)]
                    ai_magnebot.messages.append((source_agent_id,message))
                    print("message", message, source_agent_id, agent_id)
                """
                
                
                
                #Receive action for ai controlled robot
                @self.sio.event
                def ai_action(action_message, agent_id_translated): #No arbitrary eval should be allowed, check here
                
                    if not self.reset and not self.reset_partial:
                        print('New command:', action_message, agent_id_translated)
                        agent_id = list(self.robot_names_translate.keys())[list(self.robot_names_translate.values()).index(agent_id_translated)]
                        ai_agent_idx = self.ai_magnebots_ids.index(agent_id)
                        ai_agent = self.ai_magnebots[ai_agent_idx]
                        

                        
                        
                        for actions in action_message:
                        
                            if actions[0] == 'send_occupancy_map':
                                function = self.send_occupancy_map
                            elif actions[0] == 'send_objects_held_status':
                                function = self.send_objects_held_status
                            elif actions[0] == 'send_danger_sensor_reading':
                                function = self.send_danger_sensor_reading
                            elif actions[0] == 'turn_by':
                                
                                ai_agent.turn_by(float(actions[1]), aligned_at=float(actions[2]))
                            elif actions[0] == 'turn_to':
                                object_id = list(self.object_names_translate.keys())[list(self.object_names_translate.values()).index(actions[1])]
                                ai_agent.turn_to(object_id, aligned_at=float(actions[2]))
                            elif actions[0] == 'move_by':
                                ai_agent.move_by(float(actions[1]), arrived_at=float(actions[2]))
                            elif actions[0] == 'move_to':

                                ai_agent.move_to(json.loads(actions[1]), arrived_at=float(actions[2]), aligned_at=float(actions[3]), arrived_offset=float(actions[4]))
                            elif actions[0] == 'reach_for':
                                object_id = list(self.object_names_translate.keys())[list(self.object_names_translate.values()).index(actions[1])]
                                ai_agent.reach_for(object_id, Arm(int(actions[2])))
                            elif actions[0] == 'grasp':
                                object_id = list(self.object_names_translate.keys())[list(self.object_names_translate.values()).index(actions[1])]

                                #print("Grasping",object_id,ai_agent.strength,self.required_strength[object_id],all(object_id not in um.dynamic.held[arm] for um in [*self.user_magnebots,*self.ai_magnebots] for arm in [Arm.left,Arm.right])) #Grasping 10936571 2 1 True

                                if ai_agent.strength < self.required_strength[object_id]:
                                
                                    pass
                                    
                                    """
                                
                                    if object_id in self.dangerous_objects:

                                            
                                        #self.sio.emit("disable", (self.robot_names_translate[str(self.user_magnebots[idx].robot_id)]))
                                        ai_agent.disabled = True
                                        ai_agent.stats.end_time = self.timer
                                        ai_agent.stats.failed = 1
                                        
                                    else:
                                        pass
                                    """
                                
                                else:
                                    if all(object_id not in um.dynamic.held[arm] for um in [*self.user_magnebots,*self.ai_magnebots] for arm in [Arm.left,Arm.right]):
                                        print("grasping object 3", object_id)
                                        extra_commands.append({"$type": "set_mass", "mass": 1, "id": object_id})
                                        duration.append(1)
                                        ai_agent.grasp(object_id, Arm(int(actions[2])))
                                        ai_agent.stats.grab_attempts += 1
                                        ai_agent.resetting_arm = True
                            elif actions[0] == 'drop':
                                object_id = list(self.object_names_translate.keys())[list(self.object_names_translate.values()).index(actions[1])]
                                arm = Arm(int(actions[2]))
                                ai_agent.drop(object_id, arm)
                                self.object_dropping.append([int(ai_agent.dynamic.held[arm][0]),time.time(),ai_agent,arm])
                                
                                """
                                if self.danger_level[object_id] == 2 and np.linalg.norm(self.object_manager.transforms[object_id].position[[0,2]]) >= float(self.cfg["goal_radius"]):
                                
                                    robot_ids,sort_indices = self.get_involved_teammates(ai_agent.current_teammates)
                                
                                    for sidx in range(int(self.required_strength[object_id])-1):
                                        all_magnebots[robot_ids[sort_indices[sidx]]].stats.dropped_outside_goal.append(self.object_names_translate[object_id])
                                
                                    ai_agent.stats.dropped_outside_goal.append(self.object_names_translate[object_id])
                                """
                                    
                            elif actions[0] == 'reset_arm':
                                ai_agent.reset_arm(Arm(int(actions[1])))
                            elif actions[0] == 'rotate_camera':
                                ai_agent.rotate_camera(float(actions[1]), float(actions[2]), float(actions[3]))
                            elif actions[0] == 'look_at':
                                ai_agent.look_at(json.loads(actions[1]))
                            elif actions[0] == 'move_camera':
                                function = ai_agent.move_camera(json.loads(actions[1]))
                            elif actions[0] == 'reset_camera':
                                ai_agent.reset_camera()
                            elif actions[0] == 'slide_torso':
                                ai_agent.slide_torso(float(actions[1]))
                            elif actions[0] == 'reset_position':
                                ai_agent.reset_position()
                            else:
                                continue
                            
                            if 'send_' in actions[0]:
                                self.queue_perception_action.append([function,[agent_id],self.ai_skip_frames])

                            
                            '''
                            if 'send_' in actions[0]: # or 'send_occupancy_map' in actions[0] or 'send_objects_held_status' in actions[0]:
                                eval_string = "self."
                            else:
                                eval_string = "ai_agent."
                            
                            
                            eval_string += actions[0]+"("

                            for a_idx, argument in enumerate(actions[1:]):
                                if a_idx:
                                    eval_string += ','
                                eval_string += argument

                            if 'send_' in actions[0]:
                                if len(actions) > 1:
                                    eval_string += ','
                                eval_string +=  '"' + agent_id + '")'
                                self.queue_perception_action.append([eval_string,self.ai_skip_frames])
                            else:
                                #print("Eval string", eval_string)

                                eval(eval_string + ")")
                            '''
                            
                        
                        

                #Indicate use of occupancy maps
                @self.sio.event
                def watcher_ai(agent_id_translated, view_radius, centered, skip_frames):
                
                    agent_id = list(self.robot_names_translate.keys())[list(self.robot_names_translate.values()).index(agent_id_translated)]
                    ai_agent_idx = self.ai_magnebots_ids.index(agent_id)

                    self.ai_magnebots[ai_agent_idx].view_radius = int(view_radius)
                    self.ai_magnebots[ai_agent_idx].centered_view = int(centered)
                    self.ai_magnebots[ai_agent_idx].skip_frames = int(skip_frames)

                    for aim_idx,aim in enumerate(self.ai_magnebots):
                        if aim_idx != ai_agent_idx and aim.view_radius == int(view_radius) and aim.visibility_matrix:
                            self.ai_magnebots[ai_agent_idx].visibility_matrix = aim.visibility_matrix
                            break

                    if not self.ai_magnebots[ai_agent_idx].visibility_matrix:
                        if int(view_radius) not in self.visibility_matrix.keys():
                            self.visibility_matrix[int(view_radius)] = self.label_components(self.static_occupancy_map.occupancy_map, int(view_radius))
                    
                        self.ai_magnebots[ai_agent_idx].visibility_matrix = self.visibility_matrix[int(view_radius)]
                   
                #Reset environment
                @self.sio.event 
                def reset():
                    self.reset = True
                    self.previous_scenario = self.scenario
                    #self.scenario = 1
                    
                @self.sio.event 
                def reset_partial():
                    self.reset_partial = True
                    
                @self.sio.event
                def reset_tutorial():
                    
                    self.reset = True
                    self.previous_scenario = self.scenario
                    if not self.options.no_human_test:
                        self.scenario = 2    
                    else:
                        self.scenario = self.options.scenario
                        self.waiting = False
                #Key
                @self.sio.event
                def key(key, agent_id_translated):

                    try:
                        agent_id = list(self.robot_names_translate.keys())[list(self.robot_names_translate.values()).index(agent_id_translated)]
                    
                        user_agent_idx = self.user_magnebots_ids.index(agent_id)
                        
                        if key in self.user_magnebots[0].key_set: #Check whether key is in magnebot key set
                            k_idx = self.user_magnebots[0].key_set.index(key)
                            self.extra_keys_pressed.append(self.user_magnebots[user_agent_idx].key_set[k_idx]) #Key is converted to required one
                            #print(key, self.user_magnebots[user_agent_idx].key_set[k_idx])
                    except:
                        print("Key error", key, agent_id_translated)
                    
                #Disable robot
                @self.sio.event 
                def disable(agent_id_translated):

                    print("Disabling robot", agent_id_translated)

                    agent_id = list(self.robot_names_translate.keys())[list(self.robot_names_translate.values()).index(agent_id_translated)]
                    
                    if agent_id in self.user_magnebots_ids:
                        agent_idx = self.user_magnebots_ids.index(agent_id)
                        self.user_magnebots[agent_idx].disabled = True
                        self.user_magnebots[agent_idx].stats.end_time = self.timer
                    elif agent_id in self.ai_magnebots_ids:
                        agent_idx = self.ai_magnebots_ids.index(agent_id)
                        self.ai_magnebots[agent_idx].disabled = True
                        self.ai_magnebots[agent_idx].stats.end_time = self.timer
                        
                #Heavy dangerous objects reported in the end
                @self.sio.event
                def report(object_list, agent_id_translated):
                
                    all_ids = [*self.user_magnebots_ids,*self.ai_magnebots_ids]
                    all_magnebots = [*self.user_magnebots,*self.ai_magnebots]
                
                    agent_id = list(self.robot_names_translate.keys())[list(self.robot_names_translate.values()).index(agent_id_translated)]
                    agent_idx = all_ids.index(agent_id)
                
                    all_magnebots[agent_idx].reported_objects = object_list
                
                #Enable time limit    
                @self.sio.event
                def enable_timer():
                    self.timer_limit = float(self.cfg['timer'])
                    self.enable_logs = True
                    
                print(address)
                self.sio.connect(address)

    def log_init_data(self):
        log_dir = './log/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        env_objects_data = [[round(self.object_manager.transforms[eo].position[0],2),round(self.object_manager.transforms[eo].position[2],2)] for eo in self.env_objects]
            

        robots_type = []
        for magn in [*self.user_magnebots,*self.ai_magnebots]:
            robot_id = self.robot_names_translate[str(magn.robot_id)]
            robots_type.append([robot_id,magn.controlled_by,magn.p11,magn.p22])
            
        scenario_dict = {"scenario_size": self.scenario_size, "wall_length": self.wall_length, "walls": self.walls, "env_objects": env_objects_data, "robots_type": robots_type, "objects": self.objects_spawned}


        self.log_state_f = open(log_dir + dateTime + '_state.txt', "a")
        self.log_state_f.write(str(self.cfg)+'\n')
        self.log_state_f.write(str(self.options)+'\n')
        self.log_state_f.write(str(scenario_dict)+'\n')
        self.log_state_f.close()


    def extra_config_population(self):
    
        extra_config = {}


        extra_config['edge_coordinate'] = [float(self.static_occupancy_map.positions[0,0,0]),float(self.static_occupancy_map.positions[0,0,1])] #self.static_occupancy_map.get_occupancy_position(0,0)
        extra_config['cell_size'] = self.cfg['cell_size']
        #print(self.static_occupancy_map.occupancy_map.shape)
        extra_config['num_cells'] = self.static_occupancy_map.occupancy_map.shape
        extra_config['num_objects'] = len(self.graspable_objects)
        extra_config['all_robots'] = [(self.robot_names_translate[str(um.robot_id)],um.controlled_by) for um in [*self.user_magnebots,*self.ai_magnebots]]
        extra_config['timer_limit'] = float(self.cfg['timer'])
        extra_config['strength_distance_limit'] = self.cfg['strength_distance_limit']
        extra_config['communication_distance_limit'] = self.cfg['communication_distance_limit']
        extra_config['sensing_distance_limit'] = self.cfg['sensing_radius']
        extra_config["goal_radius"] = self.goal_area
        extra_config["scenario"] = self.scenario
        extra_config["sensor_parameters"] = [(self.robot_names_translate[str(um.robot_id)], um.p11, um.p22) for um in [*self.user_magnebots,*self.ai_magnebots]]
        extra_config["rooms"] = self.rooms #self.rooms
        
        
        return extra_config
        
        
    def label_components(self, occupancy_map, view_radius):
        """
        Label each free cell (0) in the occupancy_map with a unique component ID.
        Returns a 2D array of the same dimensions, where each cell contains:
          - -1 if it is an obstacle (1 in the occupancy map)
          - a non-negative integer indicating its connected component ID otherwise.
        """
        rows = occupancy_map.shape[0]
        cols = occupancy_map.shape[1]
        
        current_label = 0

        component_maps = []
        
        for i in range(rows):
            component_maps.append([])
            for j in range(cols):
                # If this cell is free (0) and unlabeled, run a BFS from here.
                component_id = []
                x_min = max(0,i-view_radius)
                y_min = max(0,j-view_radius)
                x_max = min(occupancy_map.shape[0]-1,i+view_radius)
                y_max = min(occupancy_map.shape[1]-1,j+view_radius)
                
                if occupancy_map[i][j] == 0:
                
                    sub_occupancy_map = occupancy_map[x_min:x_max+1,y_min:y_max+1]
                    
                    sub_rows = sub_occupancy_map.shape[0]
                    sub_cols = sub_occupancy_map.shape[1]
                
                    sub_i = i - x_min
                    sub_j = j - y_min
                    #component_id = [[-2] * sub_cols for _ in range(sub_rows)]
                    component_id = np.ones_like(sub_occupancy_map)*(-2)
                    m_ids = np.where(sub_occupancy_map == 1)
                    if m_ids[0].size:
                        component_id[m_ids] = 1
                    component_id = component_id.tolist()
                    queue = deque([(sub_i, sub_j)])
                    component_id[sub_i][sub_j] = current_label
                    #print(sub_occupancy_map)
                    while queue:
                        x, y = queue.popleft()
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < sub_rows and 0 <= ny < sub_cols:
                                # If neighbor is free and still unlabeled, label it
                                
                                within_range = False
                                
                                #print([sub_i,sub_j],[nx,ny])
                                #print(sub_occupancy_map)
                                
                                ''''
                                if [sub_i,sub_j] == [nx,ny]:
                                    within_range = True
                                elif len(self.findPath([sub_i,sub_j],[nx,ny],sub_occupancy_map)) <= view_radius:
                                    within_range = True
                                '''
                                within_range = True
                                
                                if sub_occupancy_map[nx][ny] == 0 and component_id[nx][ny] == -2 and within_range:
                                    component_id[nx][ny] = current_label
                                    queue.append((nx, ny))
                                elif sub_occupancy_map[nx][ny] == 1 and component_id[nx][ny] == -2 and within_range:
                                    component_id[nx][ny] = current_label
            

                #print(np.array(component_id))
                component_maps[-1].append(component_id)
        return component_maps


    def send_init_data(self):
        #Occupancy map info
        extra_config = self.extra_config_population()

        
        translated_user_magnebots_ids = [self.robot_names_translate[robot_id] for robot_id in self.user_magnebots_ids]
        translated_ai_magnebots_ids = [self.robot_names_translate[robot_id] for robot_id in self.ai_magnebots_ids]
        self.sio.emit("simulator", (translated_user_magnebots_ids,translated_ai_magnebots_ids, self.options.video_index, extra_config, dateTime, self.timer, self.real_timer))#[*self.user_magnebots_ids, *self.ai_magnebots_ids])

    def create_agents(self):
    
        self.robot_names_translate = {}
        self.user_magnebots = []
        self.ai_magnebots = []
        self.uis = []
        
        self.ai_spawn_positions = self.ai_original_spawn_positions.copy()
        self.user_spawn_positions = self.user_original_spawn_positions.copy()
        
        global agents_good_sensors
        agents_good_sensors = [1,2,3,4]
        random.shuffle(agents_good_sensors)
        
        extra_ai_agents = 0
        
        if self.scenario != 2:
            random.shuffle(self.ai_spawn_positions)
            random.shuffle(self.user_spawn_positions)
            ai_spawn_positions = self. ai_spawn_positions
            user_spawn_positions = self.user_spawn_positions
        else:
            cell_size = self.cfg['cell_size']
            user_spawn_positions = []
            ai_spawn_positions = []
            extra_ai_agents = num_users
            for um in range(num_users):
                #c1 = [self.wall_length[0]*um + cell_size*1.5 - self.scenario_size[0]/2, cell_size*1.5 - self.scenario_size[1]/2]
                user_spawn_positions.append({"x": self.wall_length[0]*um + cell_size*3.5 - self.scenario_size[0]/2, "y": 0, "z": -(self.wall_length[1] - cell_size*1.5 - self.scenario_size[1]/2)})
                ai_spawn_positions.append({"x": self.wall_length[0]*um + cell_size*4.5 - self.scenario_size[0]/2, "y": 0, "z": -(self.wall_length[1] - cell_size*5.5 - self.scenario_size[1]/2)})



        #Create user magnebots
        for us_idx in range(num_users):
            robot_id = self.get_unique_id()
            self.user_magnebots.append(Enhanced_Magnebot(robot_id=robot_id, position=user_spawn_positions[us_idx], image_frequency=ImageFrequency.always, pass_masks=['_img'],key_set=self.proposed_key_sets[us_idx], controlled_by='human', difficulty_level=self.options.level))
            self.robot_names_translate[str(robot_id)] = chr(ord('A') + us_idx)
            print(self.robot_names_translate[str(robot_id)])
    
        #Create ai magnebots
        for ai_idx in range(num_ais+extra_ai_agents):  
            robot_id = self.get_unique_id()                                 
            if self.options.ai_vision:
                self.ai_magnebots.append(Enhanced_Magnebot(robot_id=robot_id, position=ai_spawn_positions[ai_idx],image_frequency=ImageFrequency.always, pass_masks=['_img'], controlled_by='ai', difficulty_level=self.options.level))
            else:
                self.ai_magnebots.append(Enhanced_Magnebot(robot_id=robot_id, position=ai_spawn_positions[ai_idx],image_frequency=ImageFrequency.never, controlled_by='ai', difficulty_level=self.options.level))
            
            self.robot_names_translate[str(robot_id)] = chr(ord('A') + ai_idx + num_users)
            print(self.robot_names_translate[str(robot_id)])
        
        
        reticule_size = 9
        # Create a reticule.
        arr = np.zeros(shape=(reticule_size, reticule_size), dtype=np.uint8)
        x = np.arange(0, arr.shape[0])
        y = np.arange(0, arr.shape[1])
        # Define a circle on the array.
        r = reticule_size // 2
        mask = ((x[np.newaxis, :] - r) ** 2 + (y[:, np.newaxis] - r) ** 2 < r ** 2)
        # Set the color of the reticule.
        arr[mask] = 200
        arr = np.stack((arr,) * 4, axis=-1)
        # Add pointer in the middle

        Image.fromarray(arr).save('pointer.png', "PNG")


        image = "white.png"
        # Set the dimensions of the progress bar.
        self.progress_bar_position = {"x": 16, "y": -16}
        self.progress_bar_size = {"x": 16, "y": 16}
        self.progress_bar_scale = {"x": 10, "y": 2}
        self.progress_bar_anchor = {"x": 0, "y": 1}
        self.progress_bar_pivot = {"x": 0, "y": 1}

            
        #Initializing user interface objects
        for um_idx,um in enumerate(self.user_magnebots):
            um.collision_detection.objects = True
            um.collision_detection.walls = True
            ui = UI(canvas_id=um_idx)
            ui.attach_canvas_to_avatar(avatar_id=str(um.robot_id))
            
            #Create a global key_set
            if um_idx == 0:
                self.keys_set = [[um.key_set[0]],[um.key_set[1]],[um.key_set[2]],[um.key_set[3]],[um.key_set[4]],[um.key_set[5]],[um.key_set[6]],[um.key_set[7]],[um.key_set[8]], [um.key_set[9]], [um.key_set[10]], [um.key_set[11]]]
            else:
                for kidx in range(len(self.keys_set)):
                    self.keys_set[kidx].append(um.key_set[kidx])

            # Add the background sprite.
            ui.add_image(image=image,
                                 position=self.progress_bar_position,
                                 size=self.progress_bar_size,
                                 anchor=self.progress_bar_anchor,
                                 pivot=self.progress_bar_pivot,
                                 color={"r": 0, "g": 0, "b": 0, "a": 1},
                                 scale_factor=self.progress_bar_scale,
                                 rgba=False)
            
            bar_id = ui.add_image(image=image,
                                  position=self.progress_bar_position,
                                  size=self.progress_bar_size,
                                  anchor=self.progress_bar_anchor,
                                  pivot=self.progress_bar_pivot,
                                  color={"r": 0, "g": 0, "b": 1, "a": 1},
                                  scale_factor={"x": 0, "y": self.progress_bar_scale["y"]},
                                  rgba=False)
            # Add some text.
            text_id = ui.add_text(text="My Strength: 1",
                                  position=self.progress_bar_position,
                                  anchor=self.progress_bar_anchor,
                                  pivot=self.progress_bar_pivot,
                                  font_size=18)

            
            status_text_id = ui.add_text(text="Action Status: ",
            				position={"x": 80, "y": 10},
                                  	anchor={"x": 0, "y": 0},
                                  	font_size=18,
                                  	color={"r": 1, "g": 0, "b": 0, "a": 1})
                                  	
            goal_status_text = ui.add_text(text="Targets in goal: ",
                                    position={"x": -70, "y": 10},
          	                        anchor={"x": 1, "y": 0},
                                    font_size=18,
                                    color={"r": 1, "g": 0, "b": 0, "a": 1})
                                    
            ui.add_text(text="My Location",
                                    position={"x": -50, "y": -100},
          	                        anchor={"x": 1, "y": 1},
                                    font_size=18,
                                    color={"r": 1, "g": 0, "b": 0, "a": 1})
                                    
            position_text = ui.add_text(text="",
                                    position={"x": -50, "y": -120},
          	                        anchor={"x": 1, "y": 1},
                                    font_size=18,
                                    color={"r": 1, "g": 0, "b": 0, "a": 1})
                      
            ui.add_text(text="Carried object",
                                    position={"x": -60, "y": -150},
          	                        anchor={"x": 1, "y": 1},
                                    font_size=18,
                                    color={"r": 1, "g": 0, "b": 0, "a": 1})   
                                               
            arm_text = ui.add_text(text="L: R: ",
                                    position={"x": -50, "y": -160},
          	                        anchor={"x": 1, "y": 1},
                                    font_size=18,
                                    color={"r": 1, "g": 0, "b": 0, "a": 1})
            """                        
            in_goal_text = ui.add_text(text="In goal area",
                                    position={"x": -70, "y": -80},
          	                        anchor={"x": 1, "y": 1},
                                    font_size=18,
                                    color={"r": 1, "g": 0, "b": 0, "a": 1})
            """
            
            ui.add_image(image='pointer.png',
                        size={"x": reticule_size, "y": reticule_size},
                        rgba=True,
                        position={"x": 0, "y": 0})

            """
            # Add some text.
            if self.timer_limit:
                #mins, remainder = divmod(self.timer_limit-self.timer, 60)
                mins, remainder = divmod(self.timer, 60)
                secs,millisecs = divmod(remainder,1)
            else:
            """
            mins = 0
            secs = 0

            #Add timer
            timer_text_id = ui.add_text(text='{:02d}:{:02d}'.format(int(mins), int(secs)),
                                  position= {"x": -60, "y": -30},
                                  anchor = {"x": 1, "y": 1},
                                  font_size=35,
                                  color={"r": 1, "g": 0, "b": 0, "a": 1})
            
            self.uis.append(ui)
            um.ui = ui
            um.ui_elements = ((bar_id,text_id,timer_text_id, status_text_id, goal_status_text, position_text, arm_text))
            
            
            if self.scenario != 2:
                um.stats.token = ''.join(random.choices(string.ascii_lowercase + string.digits + string.ascii_uppercase, k=8))



        self.add_ons.extend([*self.ai_magnebots,  *self.user_magnebots, *self.uis])

        self.user_magnebots_ids = [str(um.robot_id) for um in self.user_magnebots]
        self.ai_magnebots_ids = [str(um.robot_id) for um in self.ai_magnebots]
     
    def convert_to_grid_coordinates(self, location, min_pos, multiple):

        if not location:
            pos_new = [-1,-1]
        else:
            pos_new = [round((location[0] + abs(min_pos[0])) / multiple),
                       round((location[1] + abs(min_pos[1])) / multiple)]

        return pos_new
    
    def calculateHValue(self,current,dest):

        dx = abs(current[0] - dest[0])
        dy = abs(current[1] - dest[1])
     
  
        h = dx + dy #For only four movements

        return h   
        
    def tracePath(self,node_details,dest):
        path = []
        
        currentNode = dest

        while node_details[currentNode[0]][currentNode[1]]["parent"][0] != currentNode[0] or node_details[currentNode[0]][currentNode[1]]["parent"][1] != currentNode[1]:
            path.append(currentNode)
            currentNode = node_details[currentNode[0]][currentNode[1]]["parent"]
            
        path.reverse()
        
        return path

    def findPath(self,startNode,endNode,occMap):

        
        openSet = [startNode]
        closedSet = []
        
        highest_cost = float('inf') #2147483647
        
        node_details = {}
        
        for s0 in range(occMap.shape[0]):
            node_details[s0] = {}
            for s1 in range(occMap.shape[1]):
                if s0 == startNode[0] and s1 == startNode[1]:
                    node_details[s0][s1] = {"f":0, "g":0, "h":0, "parent":[startNode[0],startNode[1]]}
                else:
                    node_details[s0][s1] = {"f":highest_cost, "g":highest_cost, "h":highest_cost, "parent":[-1,-1]}
        


        
        next_nodes = np.array([[-1,0],[1,0],[0,1],[0,-1]])

        while openSet:
        
            currentNode = openSet.pop(0)
            closedSet.append(tuple(currentNode))
            
     
                
            for nx in next_nodes:
                neighborNode = currentNode + nx
                
                if neighborNode[0] == endNode[0] and neighborNode[1] == endNode[1]:
                    node_details[neighborNode[0]][neighborNode[1]]["parent"] = currentNode
                    return self.tracePath(node_details, endNode)
                
                if min(neighborNode) == -1 or any(neighborNode >= occMap.shape) or not (occMap[neighborNode[0],neighborNode[1]] == 0 or occMap[neighborNode[0],neighborNode[1]] == 3 or occMap[neighborNode[0],neighborNode[1]] == -2) or tuple(neighborNode) in closedSet: #modified to allow a robot to step into another robot's place
                    continue

            
                gNew = node_details[currentNode[0]][currentNode[1]]["g"] + 1
                hNew = self.calculateHValue(neighborNode,endNode)
                fNew = gNew + hNew
                
                if node_details[neighborNode[0]][neighborNode[1]]["f"] == highest_cost or node_details[neighborNode[0]][neighborNode[1]]["f"] > fNew:
                    openSet.append(neighborNode)
                    
                    node_details[neighborNode[0]][neighborNode[1]]["f"] = fNew
                    node_details[neighborNode[0]][neighborNode[1]]["g"] = gNew
                    node_details[neighborNode[0]][neighborNode[1]]["h"] = hNew
                    node_details[neighborNode[0]][neighborNode[1]]["parent"] = currentNode
                    

        return [] #No path
        
    def manual_occupancy_map(self):
    
        self.static_occupancy_map.occupancy_map[:,:] = 0
        self.static_occupancy_map.occupancy_map[0,:] = 1
        self.static_occupancy_map.occupancy_map[-1,:] = 1
        self.static_occupancy_map.occupancy_map[:,0] = 1
        self.static_occupancy_map.occupancy_map[:,-1] = 1
        
        min_pos = [float(self.static_occupancy_map.positions[0,0,0]),float(self.static_occupancy_map.positions[0,0,1])]
        multiple = self.cfg['cell_size']
        
        for wall in self.walls:
            first_point = [wall[0][0], wall[0][1]]
            last_point = [wall[-1][0], wall[-1][1]]
            
            first_cell = [round((first_point[0] + abs(min_pos[0])) / multiple),
                   round((first_point[1] + abs(min_pos[1])) / multiple)]
                   
            last_cell = [round((last_point[0] + abs(min_pos[0])) / multiple),
                   round((last_point[1] + abs(min_pos[1])) / multiple)]
                   
                   
            if first_cell[0] == last_cell[0]:
                if first_cell[1] > last_cell[1]:
                    for loc in range(last_cell[1],first_cell[1]+1):
                        self.static_occupancy_map.occupancy_map[first_cell[0],loc] = 1
                else:
                    for loc in range(first_cell[1],last_cell[1]+1):
                        self.static_occupancy_map.occupancy_map[first_cell[0],loc] = 1
                    
            elif first_cell[1] == last_cell[1]:
                if first_cell[0] > last_cell[0]:
                    for loc in range(last_cell[0],first_cell[0]+1):
                        self.static_occupancy_map.occupancy_map[loc,first_cell[1]] = 1
                else:
                    for loc in range(first_cell[0],last_cell[0]+1):
                        self.static_occupancy_map.occupancy_map[loc,first_cell[1]] = 1
        

    #Create the scene environment
    def create_scene(self):
    
        self.state_machine = [None]*(num_users+num_ais)
        self.goal_area = [(float(self.cfg["goal_radius"]), [0,0])]
        self.wall_edges = []
        
                     
        #commands.append({"$type": "simulate_physics", "value": False})
                     
        fps = int(self.cfg['fps'])     
        if fps:    
            commands.append({"$type": "set_target_framerate", "framerate": fps})
            
            
        if self.scenario == 0:
            self.scenario_size = 10
            self.walls = []
            self.wall_length = 0
            commands = [#{'$type': 'add_scene','name': 'building_site','url': 'https://tdw-public.s3.amazonaws.com/scenes/linux/2019.1/building_site'}, 
                        {"$type": "load_scene", "scene_name": "ProcGenScene"},
                        TDWUtils.create_empty_room(self.scenario_size, self.scenario_size),
                        self.get_add_material("parquet_long_horizontal_clean",
                                              library="materials_high.json"),
                        {"$type": "set_screen_size",
                         "width": width, #640,
                         "height": height}, #480},
                        {"$type": "rotate_directional_light_by",
                         "angle": 30,
                         "axis": "pitch"}]
                         

            
            

                                       

        elif self.scenario == 1:
            self.scenario_size = 20
            self.wall_length = 6
            cell_size = self.cfg['cell_size']
            wall_width = 0.5
            
            wall1_1 = [{"x": self.wall_length, "y": idx+1} for idx in range(self.wall_length-2)]
            wall1_2 = [{"x": idx+1, "y": self.wall_length} for idx in range(self.wall_length-2)]
            
            wall2_1 = [{"x": self.scenario_size-(self.wall_length), "y": idx+1} for idx in range(self.wall_length-2)]
            wall2_2 = [{"x": self.scenario_size-(idx+1), "y": self.wall_length} for idx in range(self.wall_length-2)]
            
            wall3_1 = [{"x": self.wall_length, "y": self.scenario_size-(idx+1)} for idx in range(self.wall_length-2)]
            wall3_2 = [{"x": idx+1, "y": self.scenario_size-(self.wall_length)} for idx in range(self.wall_length-2)]
            
            wall4_1 = [{"x": self.scenario_size-(self.wall_length), "y": self.scenario_size-(idx+1)} for idx in range(self.wall_length-2)]
            wall4_2 = [{"x": self.scenario_size-(idx+1), "y": self.scenario_size-(self.wall_length)} for idx in range(self.wall_length-2)]
            
            self.walls = [[[wall[0]['x']+wall_width-self.scenario_size/2,wall[0]['y']+wall_width-self.scenario_size/2],[wall[-1]['x']+wall_width-self.scenario_size/2,wall[-1]['y']+wall_width-self.scenario_size/2]] for wall in [wall1_1,wall1_2,wall2_1,wall2_2,wall3_1,wall3_2,wall4_1,wall4_2]]

            
            commands = [#{'$type': 'add_scene','name': 'building_site','url': 'https://tdw-public.s3.amazonaws.com/scenes/linux/2019.1/building_site'}, 
                        {"$type": "load_scene", "scene_name": "ProcGenScene"},
                        TDWUtils.create_empty_room(self.scenario_size, self.scenario_size),
                        self.get_add_material("parquet_long_horizontal_clean",
                                              library="materials_high.json"),
                        {"$type": "set_screen_size",
                         "width": width, #640,
                         "height": height}, #480},
                        {"$type": "rotate_directional_light_by",
                         "angle": 30,
                         "axis": "pitch"},
                         {"$type": "create_interior_walls", "walls": [*wall1_1,*wall1_2]},
                         {"$type": "create_interior_walls", "walls": [*wall2_1,*wall2_2]},
                         {"$type": "create_interior_walls", "walls": [*wall3_1,*wall3_2]},
                         {"$type": "create_interior_walls", "walls": [*wall4_1,*wall4_2]},
                        #{"$type": "create_interior_walls", "walls": [{"x": 6, "y": 1}, {"x": 6, "y": 2},{"x": 6, "y": 3},{"x": 6, "y": 4},{"x": 1, "y": 6},{"x": 2, "y": 6},{"x": 3, "y": 6},{"x": 4, "y": 6}]},
                        #{"$type": "create_interior_walls", "walls": [{"x": 14, "y": 1}, {"x": 14, "y": 2},{"x": 14, "y": 3},{"x": 14, "y": 4},{"x": 19, "y": 6},{"x": 18, "y": 6},{"x": 17, "y": 6},{"x": 16, "y": 6}]},   
                        #{"$type": "create_interior_walls", "walls": [{"x": 6, "y": 19}, {"x": 6, "y": 18},{"x": 6, "y": 17},{"x": 6, "y": 16},{"x": 1, "y": 14},{"x": 2, "y": 14},{"x": 3, "y": 14},{"x": 4, "y": 14}]},
                        #{"$type": "create_interior_walls", "walls": [{"x": 14, "y": 19}, {"x": 14, "y": 18},{"x": 14, "y": 17},{"x": 14, "y": 16},{"x": 19, "y": 14},{"x": 18, "y": 14},{"x": 17, "y": 14},{"x": 16, "y": 14}]},
                        {"$type": "set_floor_color", "color": {"r": 1, "g": 1, "b": 1, "a": 1}},
                        {"$type": "set_proc_gen_walls_color", "color": {"r": 1, "g": 1, "b": 0, "a": 1.0}}]
        
        
            
            #self.communicate(commands)
            
            #commands = [{"$type": "create_interior_walls", "walls": [{"x": 6, "y": 19}, {"x": 6, "y": 18},{"x": 6, "y": 17},{"x": 6, "y": 16},{"x": 6, "y": 15},{"x": 1, "y": 14},{"x": 2, "y": 14},{"x": 3, "y": 14},{"x": 4, "y": 14},{"x": 5, "y": 14}]}]
                        
            #self.communicate(commands)
            
            #commands = [{"$type": "create_interior_walls", "walls": [{"x": 14, "y": 19}, {"x": 14, "y": 18},{"x": 14, "y": 17},{"x": 14, "y": 16},{"x": 14, "y": 15},{"x": 19, "y": 14},{"x": 18, "y": 14},{"x": 17, "y": 14},{"x": 16, "y": 14},{"x": 15, "y": 14}]}]
            
            
            number_angles = int(float(self.cfg["goal_radius"])*2*np.pi)
            
            for n in range(number_angles):
                angle_side = 2*n*np.pi/number_angles
                xn = float(self.cfg["goal_radius"])*np.cos(angle_side)
                zn = float(self.cfg["goal_radius"])*np.sin(angle_side)
            
                commands.append({"$type": "add_position_marker",
                                         "position": {"x": xn, "y": 0.01, "z": zn},
                                         "scale": 0.2,
                                         "shape":"circle"})
            self.wall_edges = [[wall['x']+wall_width-self.scenario_size/2,wall['y']+wall_width-self.scenario_size/2] for wall in [wall1_1[-1],wall1_2[-1],wall2_1[-1],wall2_2[-1], wall3_1[-1],wall3_2[-1], wall4_1[-1],wall4_2[-1]]]
            
            
            number_angles = 100
            
            vertex = np.array([[self.wall_edges[w_idx][0],self.wall_edges[w_idx+1][1]] for w_idx in range(0,len(self.wall_edges),2)])

            distance = [[] for v in range(vertex.shape[0])]
            angle_number = [[] for v in range(vertex.shape[0])]
            for n in range(number_angles):
                angle_side = 2*n*np.pi/number_angles
                xn = float(self.cfg["goal_radius"]+4)*np.cos(angle_side)
                zn = float(self.cfg["goal_radius"]+4)*np.sin(angle_side)
            
                res = np.linalg.norm(vertex-np.array([xn,zn]),axis=1)
                chosen_room = np.argmin(res)
                distance[chosen_room].append(res[chosen_room])
                angle_number[chosen_room].append([xn,zn])
                
            
            for w_number in range(vertex.shape[0]):
                room_number = w_number+1
                #pdb.set_trace()
                ind = np.argpartition(distance[w_number], room_number)[:room_number]
                for r in range(room_number):
                    xn,zn = angle_number[w_number][ind[r]]
                    commands.append({"$type": "add_position_marker",
                                             "position": {"x": xn, "y": 2, "z": zn},
                                             "scale": 0.2,
                                             "shape":"sphere",
                                             "color": {"r": 0, "g": 0, "b": 1, "a": 1}})
            
                                         
        elif self.scenario == 2: #Tutorial
            
            #self.state_machine = [self.Tutorial_State.move_to_agent] * (num_users+num_ais)
            self.state_machine = [self.Tutorial_State.start] * (num_users+num_ais)
            self.wall_length = [5,10]
            cell_size = self.cfg['cell_size']
            wall_width = 0.5

            self.scenario_size = [self.wall_length[0]*num_users+1,self.wall_length[1]]
            
            
            commands = [#{'$type': 'add_scene','name': 'building_site','url': 'https://tdw-public.s3.amazonaws.com/scenes/linux/2019.1/building_site'}, 
                        {"$type": "load_scene", "scene_name": "ProcGenScene"},
                        TDWUtils.create_empty_room(self.scenario_size[0], self.scenario_size[1]),
                        self.get_add_material("parquet_long_horizontal_clean",
                                              library="materials_high.json"),
                        {"$type": "set_screen_size",
                         "width": width, #640,
                         "height": height}, #480},
                        {"$type": "rotate_directional_light_by",
                         "angle": 30,
                         "axis": "pitch"}]
            
            tutorial_walls = []
            
            for um in range(num_users-1):
                tutorial_walls.append([{"x": self.wall_length[0]*(um+1), "y": idx+1} for idx in range(self.wall_length[1]-2)])
                #tutorial_walls[-1][1] = [{"x": idx+1, "y": self.wall_length[1]} for idx in range(self.wall_length[0])]
                commands.append({"$type": "create_interior_walls", "walls": [*tutorial_walls[-1]]}) #,*tutorial_walls[-1][1]]})
                
            #self.walls = [[[wall[0]['x']+wall_width-self.scenario_size[0]/2,wall[0]['y']+wall_width-self.scenario_size[1]/2],[wall[-1]['x']+wall_width-self.scenario_size[0]/2,wall[-1]['y']+wall_width-self.scenario_size[1]/2]] for room in tutorial_walls for wall in room]
            self.walls = [[[wall[0]['x']+wall_width-self.scenario_size[0]/2,wall[0]['y']+wall_width-self.scenario_size[1]/2], [wall[-1]['x']+wall_width-self.scenario_size[0]/2,wall[-1]['y']+wall_width-self.scenario_size[1]/2]] for wall in tutorial_walls]

            
            commands.extend([
                        {"$type": "set_floor_color", "color": {"r": 1, "g": 1, "b": 1, "a": 1}},
                        {"$type": "set_proc_gen_walls_color", "color": {"r": 1, "g": 1, "b": 0, "a": 1.0}}])
                
            tutorial_goal_radius = 1
            number_angles = int(tutorial_goal_radius*2*np.pi)

            self.goal_area = []            
            
            for um in range(num_users):
                self.goal_area.append((tutorial_goal_radius,[self.wall_length[0]*um + self.wall_length[0]/2 -self.scenario_size[0]/2,self.wall_length[1]/2 -self.scenario_size[1]/2]))
                for n in range(number_angles):
                    angle_side = 2*n*np.pi/number_angles
                    xn = self.wall_length[0]*um + self.wall_length[0]/2 -self.scenario_size[0]/2 + tutorial_goal_radius*np.cos(angle_side)
                    zn = self.wall_length[1]/2 -self.scenario_size[1]/2 + tutorial_goal_radius*np.sin(angle_side)
                
                    commands.append({"$type": "add_position_marker",
                                             "position": {"x": xn, "y": 0.01, "z": zn},
                                             "scale": 0.2,
                                             "shape":"circle"})
                                             
            for x in range(self.scenario_size[0]):
                for y in range(self.scenario_size[1]):
                    xn = x -self.scenario_size[0]/2
                    zn = y -self.scenario_size[1]/2
                    commands.append({"$type": "add_position_marker",
                                             "position": {"x": xn, "y": 0.01, "z": zn},
                                             "scale": 0.05,
                                             "shape":"circle",
                                             "color": {"r": 0, "g": 0, "b": 1, "a": 1}})
                  
           
        
        elif self.scenario == 3:
            self.scenario_size = 20
            self.wall_length = 6
            cell_size = self.cfg['cell_size']
            wall_width = 0.5
            
            
            wall0_1 = [{"x": idx+4, "y": 10} for idx in range(self.scenario_size-4)]
            
            wall1_1 = [{"x": self.wall_length, "y": idx+3} for idx in range(3)]
            wall1_2 = [{"x": idx+3, "y": self.wall_length} for idx in range(12)]
            
            wall2_1 = [{"x": self.scenario_size-(self.wall_length), "y": idx+1} for idx in range(self.wall_length-3)]
            wall2_2 = [{"x": self.scenario_size-(idx+1), "y": self.wall_length} for idx in range(self.wall_length-3)]
            
            wall3_1 = [{"x": self.wall_length, "y": self.scenario_size-(idx+1)} for idx in range(self.wall_length-2)]
            wall3_2 = [{"x": idx+1, "y": self.scenario_size-(self.wall_length)} for idx in range(self.wall_length-2)]
            
            wall4_1 = [{"x": self.scenario_size-(self.wall_length), "y": self.scenario_size-(idx+1)} for idx in range(self.wall_length)]
            #wall4_2 = [{"x": self.scenario_size-(idx+1), "y": self.scenario_size-(self.wall_length)} for idx in range(self.wall_length-2)]
            
            self.walls = [[[wall[0]['x']+wall_width-self.scenario_size/2,wall[0]['y']+wall_width-self.scenario_size/2],[wall[-1]['x']+wall_width-self.scenario_size/2,wall[-1]['y']+wall_width-self.scenario_size/2]] for wall in [wall1_1,wall1_2,wall2_1,wall2_2,wall3_1,wall3_2,wall4_1,wall0_1]]

            print([wall1_1,wall1_2,wall2_1,wall2_2,wall3_1,wall3_2,wall4_1,wall0_1])
            commands = [#{'$type': 'add_scene','name': 'building_site','url': 'https://tdw-public.s3.amazonaws.com/scenes/linux/2019.1/building_site'}, 
                        {"$type": "load_scene", "scene_name": "ProcGenScene"},
                        TDWUtils.create_empty_room(self.scenario_size, self.scenario_size),
                        self.get_add_material("parquet_long_horizontal_clean",
                                              library="materials_high.json"),
                        {"$type": "set_screen_size",
                         "width": width, #640,
                         "height": height}, #480},
                        {"$type": "rotate_directional_light_by",
                         "angle": 30,
                         "axis": "pitch"},
                         {"$type": "create_interior_walls", "walls": [*wall1_1,*wall1_2]},
                         {"$type": "create_interior_walls", "walls": [*wall2_1,*wall2_2]},
                         {"$type": "create_interior_walls", "walls": [*wall3_1,*wall3_2]},
                         {"$type": "create_interior_walls", "walls": [*wall4_1]},
                         {"$type": "create_interior_walls", "walls": [*wall0_1]},
                        #{"$type": "create_interior_walls", "walls": [{"x": 6, "y": 1}, {"x": 6, "y": 2},{"x": 6, "y": 3},{"x": 6, "y": 4},{"x": 1, "y": 6},{"x": 2, "y": 6},{"x": 3, "y": 6},{"x": 4, "y": 6}]},
                        #{"$type": "create_interior_walls", "walls": [{"x": 14, "y": 1}, {"x": 14, "y": 2},{"x": 14, "y": 3},{"x": 14, "y": 4},{"x": 19, "y": 6},{"x": 18, "y": 6},{"x": 17, "y": 6},{"x": 16, "y": 6}]},   
                        #{"$type": "create_interior_walls", "walls": [{"x": 6, "y": 19}, {"x": 6, "y": 18},{"x": 6, "y": 17},{"x": 6, "y": 16},{"x": 1, "y": 14},{"x": 2, "y": 14},{"x": 3, "y": 14},{"x": 4, "y": 14}]},
                        #{"$type": "create_interior_walls", "walls": [{"x": 14, "y": 19}, {"x": 14, "y": 18},{"x": 14, "y": 17},{"x": 14, "y": 16},{"x": 19, "y": 14},{"x": 18, "y": 14},{"x": 17, "y": 14},{"x": 16, "y": 14}]},
                        {"$type": "set_floor_color", "color": {"r": 1, "g": 1, "b": 1, "a": 1}},
                        {"$type": "set_proc_gen_walls_color", "color": {"r": 1, "g": 1, "b": 0, "a": 1.0}}]
        
        
            
            #self.communicate(commands)
            
            #commands = [{"$type": "create_interior_walls", "walls": [{"x": 6, "y": 19}, {"x": 6, "y": 18},{"x": 6, "y": 17},{"x": 6, "y": 16},{"x": 6, "y": 15},{"x": 1, "y": 14},{"x": 2, "y": 14},{"x": 3, "y": 14},{"x": 4, "y": 14},{"x": 5, "y": 14}]}]
                        
            #self.communicate(commands)
            
            #commands = [{"$type": "create_interior_walls", "walls": [{"x": 14, "y": 19}, {"x": 14, "y": 18},{"x": 14, "y": 17},{"x": 14, "y": 16},{"x": 14, "y": 15},{"x": 19, "y": 14},{"x": 18, "y": 14},{"x": 17, "y": 14},{"x": 16, "y": 14},{"x": 15, "y": 14}]}]
            
            self.goal_area = [(float(self.cfg["goal_radius"]), [0,5])]
            
            number_angles = int(float(self.goal_area[0][0])*2*np.pi)
            
            for n in range(number_angles):
                angle_side = 2*n*np.pi/number_angles
                xn = float(self.goal_area[0][0])*np.cos(angle_side) + self.goal_area[0][1][0]
                zn = float(self.goal_area[0][0])*np.sin(angle_side) + self.goal_area[0][1][1]
            
                commands.append({"$type": "add_position_marker",
                                         "position": {"x": xn, "y": 0.01, "z": zn},
                                         "scale": 0.2,
                                         "shape":"circle"})
                                         
            
            limits = [[4,y] for y in np.arange(6,14,0.5)] + [[x,6] for x in np.arange(0,3,0.5)] + [[6,y] for y in np.arange(14,16,0.5)] + [[x,14] for x in np.arange(4,6,0.5)] + [[6,y] for y in np.arange(0,3,0.5)] + [[14,y] for y in np.arange(3,6,0.5)] + [[14,y] for y in np.arange(10,14,0.5)] + [[17,y] for y in np.arange(6,10,0.5)] + [[x,6] for x in np.arange(14,17,0.5)]
            
            for loc in limits:
                real_limits = [loc[0]-self.scenario_size/2+cell_size*0.5,loc[1]-self.scenario_size/2+cell_size*0.5]
                    
                commands.append({"$type": "add_position_marker",
                                         "position": {"x": real_limits[0], "y": 0.01, "z": real_limits[1]},
                                         "scale": 0.2,
                                         "shape":"circle",
                                         "color": {"r": 0, "g": 0, "b": 0, "a": 1}})  
            
            
            signs = [["images/zero.png",{"x": 14, "y": 2, "z": 12},{"x": 0, "y": 90, "z": 0}],
                                ["images/one.png",{"x": 5, "y": 2, "z": 15},{"x": 0, "y": 315, "z": 0}],
                                ["images/two.png",{"x": 1, "y": 2, "z": 10},{"x": 0, "y": 270, "z": 0}],
                                ["images/three.png",{"x": 2, "y": 2, "z": 2},{"x": 0, "y": 225, "z": 0}],
                                ["images/four.png",{"x": 10, "y": 2, "z": 1},{"x": 0, "y": 180, "z": 0}],
                                ["images/five.png",{"x": 16, "y": 2, "z": 4},{"x": 0, "y": 135, "z": 0}],
                                ["images/six.png",{"x": 9, "y": 2, "z": 9.8},{"x": 0, "y": 0, "z": 0}],
                                ["images/seven.png",{"x": 17, "y": 2, "z": 8},{"x": 0, "y": 90, "z": 0}],
                                ["images/main.png",{"x": 10, "y": 2, "z": 18},{"x": 0, "y": 0, "z": 0}]]
                                
            
            for s in signs:
            
                s[1]["x"] = s[1]["x"]-self.scenario_size/2+cell_size*0.5
                s[1]["z"] = s[1]["z"]-self.scenario_size/2+cell_size*0.5
            
                input_image_path = s[0]
                # Open the image and encode it to base 64.
                with open(input_image_path, "rb") as f:
                    image = b64encode(f.read()).decode("utf-8")
                # Get the image size.
                size = Image.open(input_image_path).size
                quad_id = self.get_unique_id()
                commands.extend([{"$type": "create_textured_quad",
                    "position": s[1],
                    "size": {"x": 2, "y": 1},
                    "euler_angles": s[2],
                    "id": quad_id},
                    {"$type": "set_textured_quad",
                    "id": quad_id,
                    "dimensions": {"x": size[0], "y": size[1]},
                    "image": image}])
            
            self.wall_edges = [[wall['x']+wall_width-self.scenario_size/2,wall['y']+wall_width-self.scenario_size/2] for wall in [wall0_1[0],wall1_1[0],wall1_2[0],wall1_2[-1],wall2_1[-1],wall2_2[-1], wall3_1[-1],wall3_2[-1], wall4_1[-1]]]
            
            '''
            room_centers = [[r[0]+wall_width-self.scenario_size/2,r[1]+wall_width-self.scenario_size/2] for r in [[16,12],[4,16],[3,3],[10,3],[15,4]]]
            room_angles = [90,45,90,45,45]
            
            #pdb.set_trace()
            for center_idx in range(len(room_centers)):
                for c in range(center_idx+1):
                
                    xn = room_centers[center_idx][0]
                    zn = room_centers[center_idx][1]
                   
                    add_space = 0.3
                    angle_side = room_angles[center_idx]
                    xn += add_space*c*np.cos(angle_side)
                    zn += add_space*c*np.sin(angle_side)   
                            
                    commands.append({"$type": "add_position_marker",
                                     "position": {"x": xn, "y": 2, "z": zn},
                                     "scale": 0.2,
                                     "shape":"sphere",
                                     "color": {"r": 0, "g": 0, "b": 1, "a": 1}})    
                            
           '''
                  
        commands.append({"$type": "send_framerate", "frequency": "always"})                             
                                     
        return commands




    #Used to create all objects
    def populate_world(self, partial=[]):
    
        self.graspable_objects = []
        self.object_dropping = []
        self.occupancy_map_request = []
        self.objects_held_status_request = []
        self.danger_sensor_request = []
        self.ai_status_request = []
        self.raycast_request = []
        self.queue_perception_action = []
        self.extra_keys_pressed = []
        self.object_names_translate = {}
        self.required_strength = {}
        self.danger_level = {} 
        self.dangerous_objects = []
        self.env_objects = []
        
        if not partial:
            
            self.already_collected = []
            
            self.timer = 0 #time.time()
            self.real_timer = time.time()
            self.timer_start = self.timer

            """
            if float(self.cfg['timer']) > 0:
                self.timer_limit = self.timer_start + float(self.cfg['timer'])
            else:
                self.timer_limit = 0
            """
            
            self.terminate = False
        
            self.objects_spawned = []

        
        #self.communicate([])

        commands = []
        

        #Instantiate and locate objects
        

        
        if self.scenario == 0:
        
            max_coord = int(self.scenario_size/2)-2#8
            object_models = ['iron_box'] #['iron_box','4ft_shelf_metal','trunck','lg_table_marble_green','b04_backpack','36_in_wall_cabinet_wood_beach_honey']
            coords = {}
            
            #coords[object_models[0]] = [[max_coord,max_coord],[max_coord-1,max_coord-0.1],[max_coord-0.5,max_coord-0.2],[max_coord-0.4,max_coord],[max_coord,max_coord-0.5]]
            coords[object_models[0]] = [[max_coord,max_coord]]
            #coords[object_models[1]] = [[max_coord-3,max_coord]]

            #coords[object_models[2]] = [[max_coord,max_coord-3]]
            #coords[object_models[3]] = [[max_coord-2,max_coord-2]]
            #coords[object_models[4]] = [[max_coord-1,max_coord-2]]
            #coords[object_models[5]] = [[max_coord-3,max_coord-3]]

            modifications = [[1.0,1.0],[-1.0,1.0],[1.0,-1.0],[-1.0,-1.0]]
            
            

            final_coords = {}

            for objm in object_models:
                final_coords[objm] = []
            

            for fc in final_coords.keys():
                for m in modifications:
                    final_coords[fc].extend(np.array(coords[fc])*m)

            object_index = 0
            for fc in final_coords.keys():
                for c in final_coords[fc]:

                    weight = int(random.choice([1,2,3],1)[0])
                    danger_level = np.random.choice([1,2],1,p=[0.9,0.1])[0]
                    #commands.extend(self.instantiate_object(fc,{"x": c[0], "y": 0, "z": c[1]},{"x": 0, "y": 0, "z": 0},10,danger_level,weight))
                    commands.extend(self.instantiate_object(fc,{"x": c[0], "y": 0, "z": c[1]},{"x": 0, "y": 0, "z": 0},10,2,1, object_index)) #Danger level 2 and weight 1
                    #print("Position:", {"x": c[0], "y": 0, "z": c[1]})
                    object_index += 1
                    
        elif self.scenario == 1:
        
            """
            wall1 = [{"x": self.wall_length, "y": idx+1} for idx in range(self.wall_length-2)]
            wall1.extend([{"x": idx+1, "y": self.wall_length} for idx in range(self.wall_length-2)])
            
            x = np.arange(cell_size
            
            y = np.arange(cell_size*1.5,self.wall_length-cell_size*1.5, cell_size)
            
            wall2 = [{"x": self.scenario_size-(self.wall_length), "y": idx+1} for idx in range(self.wall_length-2)]
            wall2.extend([{"x": self.scenario_size-(idx+1), "y": self.wall_length} for idx in range(self.wall_length-2)])
            
            wall3 = [{"x": self.wall_length, "y": self.scenario_size-(idx+1)} for idx in range(self.wall_length-2)]
            wall3.extend([{"x": idx+1, "y": self.scenario_size-(self.wall_length)} for idx in range(self.wall_length-2)])
            
            wall4 = [{"x": self.scenario_size-(self.wall_length), "y": self.scenario_size-(idx+1)} for idx in range(self.wall_length-2)]
            wall4.extend([{"x": self.scenario_size-(idx+1), "y": self.scenario_size-(self.wall_length)} for idx in range(self.wall_length-2)])
            """
            
            max_coord = int(self.scenario_size/2)-1
            cell_size = self.cfg['cell_size']
            min_pos = [float(self.static_occupancy_map.positions[0,0,0]),float(self.static_occupancy_map.positions[0,0,1])]
            
            
            if not partial:
        
                object_models = {'iron_box':5} #, 'duffle_bag':1} #,'4ft_shelf_metal':1,'trunck':1,'lg_table_marble_green':1,'b04_backpack':1,'36_in_wall_cabinet_wood_beach_honey':1}
                #object_models = {'iron_box':1}

                #possible_ranges = [np.arange(max_coord-3,max_coord+0.5,0.5),np.arange(max_coord-3,max_coord+0.5,0.5)]
                possible_ranges = [np.arange(self.scenario_size/2-self.wall_length+cell_size*1.5,self.scenario_size/2-cell_size*0.5,cell_size),np.arange(self.scenario_size/2-self.wall_length+cell_size*1.5,self.scenario_size/2-cell_size*0.5,cell_size)]
                
                possible_locations = [[i, j] for i in possible_ranges[0] for j in possible_ranges[1] if not (self.options.no_block and i == 5.5 and j == 5.5)]
                

                modifications = [[1.0,1.0],[-1.0,1.0],[1.0,-1.0],[-1.0,-1.0]]
                #modifications = [[1.0,1.0]]
                
                print([[np.array(pl)*np.array(m) for pl in possible_locations] for m in modifications])
                
                total_num_objects = sum(np.array(list(object_models.values()))*len(modifications))
                
                weight_assignment = {}
                if self.options.level:
                    if self.options.level == 1:
                        num_dangerous = round(float(random.uniform(0.2, 0.3))*total_num_objects)
                    elif self.options.level == 2:
                        num_dangerous = round(float(random.uniform(0.45, 0.55))*total_num_objects)
                    elif self.options.level == 3:
                        num_dangerous = round(float(random.uniform(0.7, 0.8))*total_num_objects)
                    
                    dangerous_candidates = random.sample(list(range(total_num_objects)),num_dangerous)
                
                    percentage_weight = 0.5
                    
                    objects_remaining = total_num_objects
                
                    
                    if self.options.level == 2: #approx. normal distribution
                        middle_weight = round((num_users+num_ais+1)/2)
                        
                        first_half = list(range(middle_weight-1,0,-1))
                        second_half = list(range(middle_weight+1, num_users+num_ais+2))

                        if len(first_half) > len(second_half):
                            weight_range = first_half
                            other_weight_range = second_half
                        else:
                            weight_range = second_half
                            other_weight_range = first_half
                            
                        num_objects_to_assign = round(total_num_objects*percentage_weight)
                        weight_assignment[middle_weight] = num_objects_to_assign
                        objects_remaining = total_num_objects-num_objects_to_assign
                        percentage_weight /= 2
                            
                        for wi in range(len(weight_range)):
                            
                            num_objects_to_assign = math.ceil(objects_remaining/2)+1 #math.ceil(total_num_objects*percentage_weight)
                        
                            if objects_remaining-num_objects_to_assign < 0 or (not num_objects_to_assign and objects_remaining):
                                num_objects_to_assign = objects_remaining
                            
                            if wi < len(weight_range)-1:
                                half_assign = math.ceil(num_objects_to_assign/2)
                            else:
                                half_assign = math.ceil(num_objects_to_assign)
                            
                            #if num_objects_to_assign/2 == 0.5:
                            #    half_assign = 1
                            
                            weight_assignment[weight_range[wi]] = half_assign
                            objects_remaining -= half_assign
                            
                            if wi < len(other_weight_range):
                                weight_assignment[other_weight_range[wi]] = half_assign
                                objects_remaining -= half_assign
                            elif wi-1 < len(other_weight_range):
                                weight_assignment[other_weight_range[wi-1]] += num_objects_to_assign-half_assign
                                objects_remaining -= num_objects_to_assign-half_assign
                            
                            #objects_remaining -= num_objects_to_assign
                            
                            if not objects_remaining:
                                break
                            elif wi == len(weight_range)-1:
                                weight_assignment[weight_range[0]] += objects_remaining
                            
                            percentage_weight /= 2
                            
                        
                    elif self.options.level == 1 or self.options.level == 3: #positively skewed or negatively skewed distribution
                    
                        if self.options.level == 1:
                            weight_range = list(range(1,num_users+num_ais+2))
                        elif self.options.level == 3:
                            weight_range = list(range(num_users+num_ais,0,-1))
                            weight_range.append(num_users+num_ais+1)
                        
                    
                    
                        objects_remaining = total_num_objects
                        for w in weight_range:
                            num_objects_to_assign = math.ceil(total_num_objects*percentage_weight)
                            
                            if objects_remaining-num_objects_to_assign < 0 or (not num_objects_to_assign and objects_remaining):
                                num_objects_to_assign = objects_remaining
                            
                            weight_assignment[w] = num_objects_to_assign
                            
                            objects_remaining -= num_objects_to_assign
                            
                            if not objects_remaining:
                                break
                            
                            percentage_weight /= 2
      
                        if objects_remaining and self.options.level == 1:
                            weight_assignment[weight_range[-2]] += objects_remaining
                            objects_remaining = 0
                        
                    object_index_list = list(range(total_num_objects))
                    random.shuffle(object_index_list)
                    
                    weight_object_assignment = {}
                    
                    for k in weight_assignment.keys():
                        for ob in object_index_list[:weight_assignment[k]]:
                            weight_object_assignment[ob] = k
                            
                        object_index_list = object_index_list[weight_assignment[k]:]
                        
                danger_prob = self.cfg['danger_prob']*100 #0.3 #1.0 #0.3

                final_coords = {objm: [] for objm in object_models.keys()}
                
                print(weight_assignment)
                
                if not self.options.single_object:
                
                
                    for m in modifications:
                        while True:
                            possible_locations_temp = possible_locations.copy()
                            occMap = self.static_occupancy_map.occupancy_map.copy()
                            chosen_locations = {objm: [] for objm in object_models.keys()}
                            for fc in final_coords.keys():
                                for n_obj in range(object_models[fc]):
                                
                                    location = random.choice(possible_locations_temp)
                        
                                    possible_locations_temp.remove(location)
                                
                                    chosen_locations[fc].append(np.array(location)*m)
                                    
                                    grid_location = self.convert_to_grid_coordinates(chosen_locations[fc][-1].tolist(), min_pos, cell_size)
                                    occMap[grid_location[0],grid_location[1]] = 1
                                    
                                    
                                    
                                    '''
                                    location = []
                                    while True:
                                        try:
                                            location = random.choice(possible_locations_temp)
                                        except:
                                            pdb.set_trace()
                                        grid_location = self.convert_to_grid_coordinates(location, min_pos, cell_size)
                                        
                                        possible_locations_temp.remove(location)
                                        
                                        if self.options.no_block:
                                        
                                            occMap[grid_location[0],grid_location[1]] = 1
                                            
                                            if not self.findPath([10,10],grid_location,occMap):
                                                occMap[grid_location[0],grid_location[1]] = 0
                                                
                                                if not possible_locations_temp:
                                                    location = []
                                                    break
                                                else:
                                                    continue
                                            else:
                                                break
                                        else:
                                            break
                                        
                                    if location:
                                        final_coords[fc].append(np.array(location)*m)
                                    '''
                            feasible_room = True
                            
                            if self.options.no_block:
                                for fc in chosen_locations.keys():
                                    for c in chosen_locations[fc]:
                                        grid_location = self.convert_to_grid_coordinates(c.tolist(), min_pos, cell_size)
                                        if not self.findPath([10,10],grid_location,occMap):
                                            feasible_room = False
                                            break
                                    if not feasible_room:
                                        break
                            
                            if feasible_room:
                                for fc in chosen_locations.keys():
                                    final_coords[fc].extend(chosen_locations[fc])
                                
                                break
                
                else:
                    final_coords = {"iron_box": [possible_locations[0]]}

                object_index = 0
                for fc in final_coords.keys():
                    for c in final_coords[fc]:
                        
                        if self.options.level:
                        
                            if object_index in dangerous_candidates:
                                danger_level = 2
                            else:
                                danger_level = 1
                            try:
                                weight = weight_object_assignment[object_index]
                            except:
                                pdb.set_trace()
                            
                        else:
                            possible_weights = list(range(1,num_users+num_ais+2)) #Add 1 for objects too heavy to carry [1] #list(range(1,num_users+num_ais+1))
                            weights_probs = [100]*len(possible_weights)
                            
                            """
                            for p_idx in range(len(possible_weights)):
                                if not p_idx:
                                    weights_probs[p_idx] /= 2
                                elif p_idx == len(possible_weights)-1:
                                    weights_probs[p_idx] = weights_probs[p_idx-1]
                                else:
                                    weights_probs[p_idx] = weights_probs[p_idx-1]/2
                            """
                            
                            weights_probs = [int(100/len(possible_weights))]*len(possible_weights)
                            
                            if len(possible_weights) == 1:
                                weight = 1
                            else:
                                weight = int(random.choices(possible_weights,weights=weights_probs)[0])
                            danger_level = random.choices([1,2],weights=[100-danger_prob,danger_prob])[0]
                            
                            
                            #weight = 1
                            #danger_level = 2
                        
                        try:
                            commands.extend(self.instantiate_object(fc,{"x": c[0], "y": 0, "z": c[1]},{"x": 0, "y": 0, "z": 0},1000,danger_level,weight, object_index))
                        except:
                            pdb.set_trace()
                        object_index += 1
                        #commands.extend(self.instantiate_object(fc,{"x": c[0], "y": 0, "z": c[1]},{"x": 0, "y": 0, "z": 0},10,2,1)) #Danger level 2 and weight 1
                        #print("Position:", {"x": c[0], "y": 0, "z": c[1]})

                #commands.extend(self.instantiate_object('iron_box',{"x": 0, "y": 0, "z": 0},{"x": 0, "y": 0, "z": 0},10,1,1)) #Single box


            else:
            
                model_name = "iron_box"
                mass = 1000
                for obj in partial:
                    
                    required_strength = obj[1]
                    danger_level = obj[2]
                    object_name = obj[0]
                    position = {"x": float(obj[3][0]), "y": 0, "z": float(obj[3][2])} 
                    rotation = {"x": 0, "y": 0, "z": 0}
                    object_id = self.get_unique_id()
                    self.graspable_objects.append(object_id)
                    self.object_names_translate[object_id] = object_name
                    self.required_strength[object_id] = required_strength
                    self.danger_level[object_id] = danger_level
                    command = self.get_add_physics_object(model_name=model_name,
                                                     object_id=object_id,
                                                     position=position,
                                                     rotation=rotation,
                                                     default_physics_values=False,
                                                     mass=mass,
                                                     scale_mass=False)
                    if self.danger_level[object_id] == 2:
                        self.dangerous_objects.append(object_id)

                    commands.extend(command)
        
            #Create environment objects
            
            
            '''
            self.env_objects.append(self.get_unique_id())
            
            
            
            commands.extend(self.get_add_physics_object(model_name="satiro_sculpture",
                                             object_id=self.env_objects[-1],
                                             position={"x": 0, "y": 0, "z": 0},
                                             default_physics_values=False,
                                             mass=1000,
                                             scale_mass=False,
                                             rotation={"x": 0, "y": 0, "z": 0}))
            '''
            
                                             
            self.env_objects.append(self.get_unique_id())
            

            
            
            commands.extend(self.get_add_physics_object(model_name="zenblocks",
                                             object_id=self.env_objects[-1],
                                             position={"x": max_coord-self.wall_length+cell_size/2, "y": 0, "z": max_coord-cell_size*1.5},
                                             default_physics_values=False,
                                             mass=1000,
                                             scale_mass=False,
                                             rotation={"x": 0, "y": 0, "z": 0}))
         
                                             
            self.env_objects.append(self.get_unique_id())
            
            commands.extend(self.get_add_physics_object(model_name="amphora_jar_vase",
                                             object_id=self.env_objects[-1],
                                             position={"x": 2.5*cell_size-max_coord, "y": 0, "z": self.wall_length-max_coord+cell_size/2},
                                             default_physics_values=False,
                                             mass=1000,
                                             scale_mass=False,
                                             rotation={"x": 0, "y": 0, "z": 0}))
                                             
            self.env_objects.append(self.get_unique_id())
            
            commands.extend(self.get_add_physics_object(model_name="linen_dining_chair",
                                             object_id=self.env_objects[-1],
                                             position={"x": 2.5*cell_size-max_coord, "y": 0, "z": max_coord-self.wall_length+cell_size/2},
                                             default_physics_values=False,
                                             mass=1000,
                                             scale_mass=False,
                                             rotation={"x": 0, "y": 0, "z": 0}))
                                             
            self.env_objects.append(self.get_unique_id())
            
            commands.extend(self.get_add_physics_object(model_name="cgaxis_models_50_12_vray",
                                             object_id=self.env_objects[-1],
                                             position={"x": max_coord-self.wall_length+cell_size/2, "y": 0, "z": 2.5*cell_size-max_coord},
                                             default_physics_values=False,
                                             mass=1000,
                                             scale_mass=False,
                                             rotation={"x": 0, "y": 0, "z": 0}))
        
            
        elif self.scenario == 2: #Tutorial
        
        
            cell_size = self.cfg['cell_size']
            
            for um in range(num_users):
                c1 = [self.wall_length[0]*um + cell_size*1.5 - self.scenario_size[0]/2, cell_size*1.5 - self.scenario_size[1]/2]
                c2 = [self.wall_length[0]*um + cell_size*2.5 - self.scenario_size[0]/2, cell_size*7.5 - self.scenario_size[1]/2]
                c3 = [self.wall_length[0]*um + cell_size*3.5 - self.scenario_size[0]/2, cell_size*7.5 - self.scenario_size[1]/2]
                c4 = [self.wall_length[0]*um + cell_size*3 - self.scenario_size[0]/2, cell_size*7.8 - self.scenario_size[1]/2]
                
                commands.extend(self.instantiate_object('iron_box',{"x": c1[0], "y": 0, "z": c1[1]},{"x": 0, "y": 0, "z": 0},1000,2,2, um*4))
                commands.extend(self.instantiate_object('iron_box',{"x": c2[0], "y": 0, "z": c2[1]},{"x": 0, "y": 0, "z": 0},1000,2,1, 1 + um*4))
                commands.extend(self.instantiate_object('iron_box',{"x": c3[0], "y": 0, "z": c3[1]},{"x": 0, "y": 0, "z": 0},1000,1,1, 2 + um*4))
                commands.extend(self.instantiate_object('iron_box',{"x": c4[0], "y": 0, "z": c4[1]},{"x": 0, "y": 0, "z": 0},1000,2,3, 3 + um*4))
            
            
        elif self.scenario == 3:
        
            """
            wall1 = [{"x": self.wall_length, "y": idx+1} for idx in range(self.wall_length-2)]
            wall1.extend([{"x": idx+1, "y": self.wall_length} for idx in range(self.wall_length-2)])
            
            x = np.arange(cell_size
            
            y = np.arange(cell_size*1.5,self.wall_length-cell_size*1.5, cell_size)
            
            wall2 = [{"x": self.scenario_size-(self.wall_length), "y": idx+1} for idx in range(self.wall_length-2)]
            wall2.extend([{"x": self.scenario_size-(idx+1), "y": self.wall_length} for idx in range(self.wall_length-2)])
            
            wall3 = [{"x": self.wall_length, "y": self.scenario_size-(idx+1)} for idx in range(self.wall_length-2)]
            wall3.extend([{"x": idx+1, "y": self.scenario_size-(self.wall_length)} for idx in range(self.wall_length-2)])
            
            wall4 = [{"x": self.scenario_size-(self.wall_length), "y": self.scenario_size-(idx+1)} for idx in range(self.wall_length-2)]
            wall4.extend([{"x": self.scenario_size-(idx+1), "y": self.scenario_size-(self.wall_length)} for idx in range(self.wall_length-2)])
            """
            
            max_coord = int(self.scenario_size/2)-1
            cell_size = self.cfg['cell_size']
            min_pos = [float(self.static_occupancy_map.positions[0,0,0]),float(self.static_occupancy_map.positions[0,0,1])]
            
            
            if not partial:
        
                object_models = {'iron_box':20} #, 'duffle_bag':1} #,'4ft_shelf_metal':1,'trunck':1,'lg_table_marble_green':1,'b04_backpack':1,'36_in_wall_cabinet_wood_beach_honey':1}
                #object_models = {'iron_box':1}
                
                
                self.scenario_size = 20
                cell_size = self.cfg['cell_size']
                wall_width = 0.5
                self.rooms = {}
                self.rooms_limits = {}

                
                self.rooms[0] = [[x,y] for x in range(15,19) for y in range(11,19)]
                self.rooms[1] = [[x,y] for x in range(1,6) for y in range(15,19)]
                self.rooms[2] = [[x,y] for x in range(1,5) for y in range(7,14)]
                self.rooms[3] = [[x,y] for x in range(1,6) for y in range(1,6)]
                self.rooms[4] = [[x,y] for x in range(7,14) for y in range(1,6)]
                self.rooms[5] = [[x,y] for x in range(15,19) for y in range(1,6)]
                self.rooms[6] = [[x,y] for x in range(5,17) for y in range(7,10)]
                self.rooms[7] = [[x,y] for x in range(17,19) for y in range(7,10)]
                
                map_variations = [[[x,6] for x in range(1,3)],[[x,6] for x in range(15,17)]]
                
                room7_constraint = [[17,y] for y in range(7,10)] #we cannot have three objects in line
                room7_constraint = [[loc[0]-self.scenario_size/2+cell_size*0.5,loc[1]-self.scenario_size/2+cell_size*0.5] for loc in room7_constraint]
                
                
                for r_key in self.rooms.keys():
                    if r_key == 6:
                        multipliers = [0.5,0.5,1.2,1.5]
                    else:
                        multipliers = [0.5,0.5,1.5,1.5]
                        
                    self.rooms_limits[r_key] = [[loc[0]-self.scenario_size/2-cell_size*multipliers[0],loc[1]-self.scenario_size/2-cell_size*multipliers[1]] if loc_idx < len(self.rooms[r_key])-1 else [loc[0]-self.scenario_size/2+cell_size*multipliers[2],loc[1]-self.scenario_size/2+cell_size*multipliers[3]] for loc_idx,loc in enumerate(self.rooms[r_key])] #Slight modification from 0.5 to 1
                    self.rooms[r_key] = [[loc[0]-self.scenario_size/2+cell_size*0.5,loc[1]-self.scenario_size/2+cell_size*0.5] for loc in self.rooms[r_key]]
                
                original_room_capacity = {0:10,1:5,2:0,3:5,4:7,5:5,6:0,7:3}
                room_capacity = original_room_capacity.copy()
                room_empty = random.choice([r for r in room_capacity.keys() if room_capacity[r] > 0])
                room_capacity[room_empty] = 0
                actual_room_capacity = {r_key:0 if not room_capacity[r_key] else 2 for r_key in self.rooms.keys()}
                
                #possible_ranges = [np.arange(max_coord-3,max_coord+0.5,0.5),np.arange(max_coord-3,max_coord+0.5,0.5)]
                #possible_ranges = [np.arange(self.scenario_size/2-self.wall_length+cell_size*1.5,self.scenario_size/2-cell_size*0.5,cell_size),np.arange(self.scenario_size/2-self.wall_length+cell_size*1.5,self.scenario_size/2-cell_size*0.5,cell_size)]
                
                #possible_locations = [[i, j] for i in possible_ranges[0] for j in possible_ranges[1] if not (self.options.no_block and i == 5.5 and j == 5.5)]
                
                #modifications = [[1.0,1.0],[-1.0,1.0],[1.0,-1.0],[-1.0,-1.0]]
                #modifications = [[1.0,1.0]]
                
                #print([[np.array(pl)*np.array(m) for pl in possible_locations] for m in modifications])
                
                total_num_objects = random.choice([19,20,21]) #20
                
                to_assign_total_num_objects = total_num_objects - sum(actual_room_capacity[r_key] for r_key in actual_room_capacity.keys()) #sum(np.array(list(object_models.values()))*len(modifications))
                
                for t in range(to_assign_total_num_objects):
                    while True:
                        room_assignment = random.choice(range(len(self.rooms.keys())))
                        if actual_room_capacity[room_assignment] < room_capacity[room_assignment]:
                            actual_room_capacity[room_assignment] += 1
                            break
                        else:
                            continue
                
                
                weight_assignment = {}
                if self.options.level:
                    if self.options.level == 1:
                        num_dangerous = round(float(random.uniform(0.2, 0.3))*total_num_objects)
                    elif self.options.level == 2:
                        num_dangerous = round(float(random.uniform(0.45, 0.55))*total_num_objects)
                    elif self.options.level == 3:
                        num_dangerous = round(float(random.uniform(0.7, 0.8))*total_num_objects)
                    
                    dangerous_candidates = random.sample(list(range(total_num_objects)),num_dangerous)
                
                    percentage_weight = 0.5
                    
                    objects_remaining = total_num_objects
                
                    
                    if self.options.level == 2 or self.options.level == 1: #approx. normal distribution
                    
                        if self.options.level == 1:
                            middle_weight = 2
                        else:
                            middle_weight = round((num_users+num_ais+1)/2)
                        
                        first_half = list(range(middle_weight-1,0,-1))
                        second_half = list(range(middle_weight+1, num_users+num_ais+2))

                        if len(first_half) > len(second_half):
                            weight_range = first_half
                            other_weight_range = second_half
                        else:
                            weight_range = second_half
                            other_weight_range = first_half
                            
                        num_objects_to_assign = round(total_num_objects*percentage_weight)
                        weight_assignment[middle_weight] = num_objects_to_assign
                        objects_remaining = total_num_objects-num_objects_to_assign
                        percentage_weight /= 2
                            
                        for wi in range(len(weight_range)):
                            
                            num_objects_to_assign = math.ceil(objects_remaining/2)+1 #math.ceil(total_num_objects*percentage_weight)
                        
                            if objects_remaining-num_objects_to_assign < 0 or (not num_objects_to_assign and objects_remaining):
                                num_objects_to_assign = objects_remaining
                            
                            if wi < len(weight_range)-1:
                                half_assign = math.ceil(num_objects_to_assign/2)
                            else:
                                half_assign = math.ceil(num_objects_to_assign)
                            
                            #if num_objects_to_assign/2 == 0.5:
                            #    half_assign = 1
                            
                            weight_assignment[weight_range[wi]] = half_assign
                            objects_remaining -= half_assign
                            
                            if wi < len(other_weight_range):
                                weight_assignment[other_weight_range[wi]] = half_assign
                                objects_remaining -= half_assign
                            elif wi-1 < len(other_weight_range):
                                weight_assignment[other_weight_range[wi-1]] += num_objects_to_assign-half_assign
                                objects_remaining -= num_objects_to_assign-half_assign
                            
                            #objects_remaining -= num_objects_to_assign
                            
                            if not objects_remaining:
                                break
                            elif wi == len(weight_range)-1:
                                weight_assignment[weight_range[0]] += objects_remaining
                            
                            percentage_weight /= 2
                            
                        
                    elif self.options.level == 1 or self.options.level == 3: #positively skewed or negatively skewed distribution
                    
                        if self.options.level == 1:
                            weight_range = list(range(1,num_users+num_ais+2))
                        elif self.options.level == 3:
                            weight_range = list(range(num_users+num_ais,0,-1))
                            weight_range.append(num_users+num_ais+1)
                        
                    
                    
                        objects_remaining = total_num_objects
                        for w in weight_range:
                            num_objects_to_assign = math.ceil(total_num_objects*percentage_weight)
                            
                            if objects_remaining-num_objects_to_assign < 0 or (not num_objects_to_assign and objects_remaining):
                                num_objects_to_assign = objects_remaining
                            
                            weight_assignment[w] = num_objects_to_assign
                            
                            objects_remaining -= num_objects_to_assign
                            
                            if not objects_remaining:
                                break
                            
                            percentage_weight /= 2
      
                        if objects_remaining and self.options.level == 1:
                            weight_assignment[weight_range[-2]] += objects_remaining
                            objects_remaining = 0
                        
                    object_index_list = list(range(total_num_objects))
                    random.shuffle(object_index_list)
                    
                    weight_object_assignment = {}
                    
                    for k in weight_assignment.keys():
                        for ob in object_index_list[:weight_assignment[k]]:
                            weight_object_assignment[ob] = k
                            
                        object_index_list = object_index_list[weight_assignment[k]:]
                        
                danger_prob = self.cfg['danger_prob']*100 #0.3 #1.0 #0.3

                final_coords = {objm: [] for objm in object_models.keys()}
                
                print(weight_assignment)
                #pdb.set_trace()
                
                #Ensure connectivity between rooms!!!
                
                if not self.options.single_object:
                
                    goal_center = self.convert_to_grid_coordinates(self.goal_area[0][1], min_pos, cell_size)
                    for r_key in self.rooms.keys():
                        while True:
                            possible_locations_temp = self.rooms[r_key].copy()
                            occMap = self.static_occupancy_map.occupancy_map.copy()
                            chosen_locations = {objm: [] for objm in object_models.keys()}
                            for fc in final_coords.keys():
                                for n_obj in range(actual_room_capacity[r_key]):
                                
                                    location = random.choice(possible_locations_temp)
                        
                                    possible_locations_temp.remove(location)
                                
                                    chosen_locations[fc].append(np.array(location))
                                    
                                    grid_location = self.convert_to_grid_coordinates(chosen_locations[fc][-1].tolist(), min_pos, cell_size)
                                    occMap[grid_location[0],grid_location[1]] = 1
                                    
                                    if r_key == 7 and sum(locs in possible_locations_temp for locs in room7_constraint) == 1:
                                        for locs in room7_constraint:
                                            if locs in possible_locations_temp:
                                                possible_locations_temp.remove(locs)
                                    
                                    
                                    '''
                                    location = []
                                    while True:
                                        try:
                                            location = random.choice(possible_locations_temp)
                                        except:
                                            pdb.set_trace()
                                        grid_location = self.convert_to_grid_coordinates(location, min_pos, cell_size)
                                        
                                        possible_locations_temp.remove(location)
                                        
                                        if self.options.no_block:
                                        
                                            occMap[grid_location[0],grid_location[1]] = 1
                                            
                                            if not self.findPath([10,10],grid_location,occMap):
                                                occMap[grid_location[0],grid_location[1]] = 0
                                                
                                                if not possible_locations_temp:
                                                    location = []
                                                    break
                                                else:
                                                    continue
                                            else:
                                                break
                                        else:
                                            break
                                        
                                    if location:
                                        final_coords[fc].append(np.array(location)*m)
                                    '''
                            feasible_room = True
                            
                            if self.options.no_block:
                            
                                occMap_variations = []
                                for m1 in map_variations:
                                    occMap_variations.append(occMap.copy())
                                    for m2 in m1:
                                         occMap_variations[-1][m2[0],m2[1]] = 1
                            
                                for fc in chosen_locations.keys():
                                    for c in chosen_locations[fc]:
                                        grid_location = self.convert_to_grid_coordinates(c.tolist(), min_pos, cell_size)
                                        if not self.findPath(goal_center,grid_location,occMap) or any(not self.findPath(goal_center,grid_location,ov) for ov in occMap_variations):
                                            feasible_room = False
                                            break
                                    if not feasible_room:
                                        break
                            
                            if feasible_room:
                                for fc in chosen_locations.keys():
                                    final_coords[fc].extend(chosen_locations[fc])
                                    
                                break
                
                else:
                    final_coords = {"iron_box": [possible_locations[0]]}

                
                #weight to dangerous object
                
                dangerous_candidates = []
                while True:
                    
                    dangerous_candidates = random.sample(list(weight_object_assignment.keys()), num_dangerous)
                    
                    # per-room cap
                    counts = {}
                    max_per_room = 2
                    for oid in dangerous_candidates:
                        sum_room = 0
                        room_num = 0
                        for r in actual_room_capacity.keys():
                            sum_room += actual_room_capacity[r]
                            if oid < sum_room:
                                room_num = r
                                break
                        counts[room_num] = counts.get(room_num, 0) + 1
                    if any(cnt > max_per_room for cnt in counts.values()):
                        continue
                    
                    # weight constraints
                    half_weight_thresh = 2
                    high_weight_thresh = 3
                    wts = [weight_object_assignment[oid] for oid in dangerous_candidates]
                    if sum(1 for w in wts if w >= half_weight_thresh) < (num_dangerous / 2):
                        continue
                    if not any(w >= high_weight_thresh for w in wts):
                        continue
        
                    break
                
                object_index = 0
                for fc in final_coords.keys():
                    for c in final_coords[fc]:
                        
                        if self.options.level:
                        
                            if object_index in dangerous_candidates:
                                danger_level = 2
                            else:
                                danger_level = 1
                            try:
                                weight = weight_object_assignment[object_index]
                            except:
                                pdb.set_trace()
                            
                        else:
                            possible_weights = list(range(1,num_users+num_ais+2)) #Add 1 for objects too heavy to carry [1] #list(range(1,num_users+num_ais+1))
                            weights_probs = [100]*len(possible_weights)
                            
                            """
                            for p_idx in range(len(possible_weights)):
                                if not p_idx:
                                    weights_probs[p_idx] /= 2
                                elif p_idx == len(possible_weights)-1:
                                    weights_probs[p_idx] = weights_probs[p_idx-1]
                                else:
                                    weights_probs[p_idx] = weights_probs[p_idx-1]/2
                            """
                            
                            weights_probs = [int(100/len(possible_weights))]*len(possible_weights)
                            
                            if len(possible_weights) == 1:
                                weight = 1
                            else:
                                weight = int(random.choices(possible_weights,weights=weights_probs)[0])
                            danger_level = random.choices([1,2],weights=[100-danger_prob,danger_prob])[0]
                            
                            
                            #weight = 1
                            #danger_level = 2
                        
                        try:
                            #if object_index:
                            commands.extend(self.instantiate_object(fc,{"x": c[0], "y": 0, "z": c[1]},{"x": 0, "y": 0, "z": 0},1000,danger_level,weight, object_index))
                            #else:
                            #    commands.extend(self.instantiate_object(fc,{"x": -5, "y": 0, "z": 2},{"x": 0, "y": 0, "z": 0},1000,danger_level,weight, object_index))
                        except:
                            pdb.set_trace()
                        object_index += 1
                        #commands.extend(self.instantiate_object(fc,{"x": c[0], "y": 0, "z": c[1]},{"x": 0, "y": 0, "z": 0},10,2,1)) #Danger level 2 and weight 1
                        #print("Position:", {"x": c[0], "y": 0, "z": c[1]})

                #commands.extend(self.instantiate_object('iron_box',{"x": 0, "y": 0, "z": 0},{"x": 0, "y": 0, "z": 0},10,1,1)) #Single box


            else:
            
                model_name = "iron_box"
                mass = 1000
                for obj in partial:
                    
                    required_strength = obj[1]
                    danger_level = obj[2]
                    object_name = obj[0]
                    position = {"x": float(obj[3][0]), "y": 0, "z": float(obj[3][2])} 
                    rotation = {"x": 0, "y": 0, "z": 0}
                    object_id = self.get_unique_id()
                    self.graspable_objects.append(object_id)
                    self.object_names_translate[object_id] = object_name
                    self.required_strength[object_id] = required_strength
                    self.danger_level[object_id] = danger_level
                    command = self.get_add_physics_object(model_name=model_name,
                                                     object_id=object_id,
                                                     position=position,
                                                     rotation=rotation,
                                                     default_physics_values=False,
                                                     mass=mass,
                                                     scale_mass=False)
                    if self.danger_level[object_id] == 2:
                        self.dangerous_objects.append(object_id)

                    commands.extend(command)
        
            #Create environment objects
            
            
            '''
            self.env_objects.append(self.get_unique_id())
            
            
            
            commands.extend(self.get_add_physics_object(model_name="satiro_sculpture",
                                             object_id=self.env_objects[-1],
                                             position={"x": 0, "y": 0, "z": 0},
                                             default_physics_values=False,
                                             mass=1000,
                                             scale_mass=False,
                                             rotation={"x": 0, "y": 0, "z": 0}))
            '''
            
                                             
            #self.env_objects.append(self.get_unique_id())
            

            '''
            
            
            commands.extend(self.get_add_physics_object(model_name="zenblocks",
                                             object_id=self.env_objects[-1],
                                             position={"x": max_coord-self.wall_length+cell_size/2, "y": 0, "z": max_coord-cell_size*1.5},
                                             default_physics_values=False,
                                             mass=1000,
                                             scale_mass=False,
                                             rotation={"x": 0, "y": 0, "z": 0}))
         
                                             
            self.env_objects.append(self.get_unique_id())
            
            commands.extend(self.get_add_physics_object(model_name="amphora_jar_vase",
                                             object_id=self.env_objects[-1],
                                             position={"x": 2.5*cell_size-max_coord, "y": 0, "z": self.wall_length-max_coord+cell_size/2},
                                             default_physics_values=False,
                                             mass=1000,
                                             scale_mass=False,
                                             rotation={"x": 0, "y": 0, "z": 0}))
                                             
            self.env_objects.append(self.get_unique_id())
            
            commands.extend(self.get_add_physics_object(model_name="linen_dining_chair",
                                             object_id=self.env_objects[-1],
                                             position={"x": 2.5*cell_size-max_coord, "y": 0, "z": max_coord-self.wall_length+cell_size/2},
                                             default_physics_values=False,
                                             mass=1000,
                                             scale_mass=False,
                                             rotation={"x": 0, "y": 0, "z": 0}))
                                             
            self.env_objects.append(self.get_unique_id())
            
            commands.extend(self.get_add_physics_object(model_name="cgaxis_models_50_12_vray",
                                             object_id=self.env_objects[-1],
                                             position={"x": max_coord-self.wall_length+cell_size/2, "y": 0, "z": 2.5*cell_size-max_coord},
                                             default_physics_values=False,
                                             mass=1000,
                                             scale_mass=False,
                                             rotation={"x": 0, "y": 0, "z": 0}))
            '''
            
        # Add post-processing.
        commands.extend(get_default_post_processing_commands())     
        
        self.target = {}
        
        #Creating third person camera
        #commands.extend(TDWUtils.create_avatar(position={"x": 0, "y": 10, "z": 0},#{"x": 0, "y": 10, "z": -1},
        #                                                   look_at={"x": 0, "y": 0, "z": 0},
        #                                                   avatar_id="a"))
                                
        if not self.no_debug_camera:       
                        
            commands.extend([{"$type": "create_avatar", "type": "A_Img_Caps_Kinematic", "id": "a"}, 
            {"$type": "teleport_avatar_to", "avatar_id": "a", "position": {"x": 0, "y": 30, "z": 0}},
            {"$type": "look_at_position", "avatar_id": "a", "position": {"x": 0, "y": 0, "z": 0}},
            {"$type": "rotate_avatar_by", "angle": 90, "axis": "yaw", "is_world": True, "avatar_id": "a"}])
            commands.extend([{"$type": "set_pass_masks","pass_masks": ["_img"],"avatar_id": "a"},
                      {"$type": "send_images","frequency": "always","ids": ["a"]},
                      {"$type": "set_img_pass_encoding", "value": False},
                      {"$type": "set_render_order", "render_order": 1, "sensor_name": "SensorContainer", "avatar_id": "a"}])
        
            #commands.append({"$type": "send_keyboard", "frequency": "always"})
            
            commands.append({"$type": "set_render_quality", "render_quality": 0})
        
        
        if self.scenario != 2 and self.scenario != 3:
            commands.append({"$type": "add_compass_rose"})
        
        return commands


    
    def get_segmentation_colors(self):
    
        commands = []
    
        commands.append({"$type": "send_segmentation_colors",
           "frequency": "once"})
           
        resp = self.communicate(commands)
        
        segmentation_colors = dict()
        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            # Get segmentation color output data.
            if r_id == "segm":
                segm = SegmentationColors(resp[i])
                for j in range(segm.get_num()):
                    object_id = segm.get_object_id(j)
                    segmentation_color = segm.get_object_color(j)
                    segmentation_colors[tuple(segmentation_color.tolist())] = object_id
                    
                    
        return segmentation_colors
               
    #Function to instantiate objects
    def instantiate_object(self, model_name, position, rotation, mass, danger_level, required_strength, object_index):

        if self.options.single_weight:
            required_strength = self.options.single_weight
        if self.options.single_danger:
            danger_level = 2

        object_id = self.get_unique_id()
        self.graspable_objects.append(object_id)
        self.object_names_translate[object_id] = str(object_index)
        self.required_strength[object_id] = required_strength
        self.danger_level[object_id] = danger_level
        command = self.get_add_physics_object(model_name=model_name,
                                         object_id=object_id,
                                         position=position,
                                         rotation=rotation,
                                         default_physics_values=False,
                                         mass=mass,
                                         scale_mass=False)
        if self.danger_level[object_id] == 2:
            self.dangerous_objects.append(object_id)
            print("Dangerous object: ", object_index, ", weight: ", required_strength)
        else:
            print("Benign object: ", object_index, ", weight: ", required_strength)


        self.objects_spawned.append([str(object_index),required_strength,danger_level,position])

        return command

    #Function to add ui to camera frames
    def add_ui(self, original_image, screen_positions,topview):
        font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale
        fontScale = 0.5         
        # Blue color in BGR
        colorFont = screen_positions['color']
        # Line thickness of 2 px
        thickness = 2
        
        if topview:
            fontScale = 0.2
            thickness = 1
        
        for s_idx,s in enumerate(screen_positions['coords']):
            try:
                cv2.putText(original_image, screen_positions['ids'][s_idx], (int(s[0]),int(s[1])), font, fontScale, colorFont[s_idx], thickness, cv2.LINE_AA)
            except:
                pdb.set_trace()
                
    def info_message_ui(self, all_magnebots, idx, text, color):
    
        if color == "red":
            color_val = {"r": 1, "g": 0, "b": 0, "a": 1}
        elif color == "blue":
            color_val = {"r": 0, "g": 0, "b": 1, "a": 1}
    
        
        txt = all_magnebots[idx].ui.add_text(text=text,
         position={"x": 0, "y": 0},
         color=color_val,
         font_size=20
         )
        return [idx,txt,0]
    

    def get_involved_teammates(self, current_teammates, object_id): #Assign contributions to each teammate
    
        robot_ids = []
        total_time_spent = []
    
        try:
            for robot in current_teammates[object_id].keys():
                robot_ids.append(robot)
                total_time_spent.append(current_teammates[object_id][robot])
        except:
            pdb.set_trace()
            
        sort_indices = np.argsort(np.array(total_time_spent)).tolist()
        sort_indices.reverse()
        
        return robot_ids, sort_indices

    #Process raycasting
    def raycast_output(self, resp, all_ids):

        raycast = Raycast(resp)
        #print("raycast from ", raycast.get_raycast_id(), raycast.get_hit(), raycast.get_hit_object(), raycast.get_object_id() in self.graspable_objects, str(raycast.get_raycast_id()) in self.user_magnebots_ids, raycast.get_object_id())
             
        o_id = raycast.get_object_id()
        
        if raycast.get_hit() and raycast.get_hit_object() and o_id in self.graspable_objects and str(raycast.get_raycast_id()) in self.user_magnebots_ids: #If ray hits an object
        
            
            pos_idx = len(all_ids)+self.graspable_objects.index(o_id)
            
            
            u_idx = self.user_magnebots_ids.index(str(raycast.get_raycast_id()))
            
            
            if not self.user_magnebots[u_idx].grasping: #Grasping also uses raycasting but we don't want to use this code for that situation
            
                self.user_magnebots[u_idx].screen_positions["position_ids"].append(pos_idx)
                self.user_magnebots[u_idx].screen_positions["positions"].append(TDWUtils.array_to_vector3(raycast.get_point()))
                self.user_magnebots[u_idx].screen_positions["duration"].append(100)
                
                #print("raycasted ", raycast.get_object_id(), raycast.get_point())
                
                self.user_magnebots[u_idx].focus_object = o_id
                
                if o_id not in self.user_magnebots[u_idx].item_info:
                    self.user_magnebots[u_idx].item_info[o_id] = {}
                    
                self.user_magnebots[u_idx].item_info[o_id]['weight'] = int(self.required_strength[o_id])
                self.user_magnebots[u_idx].item_info[o_id]['time'] = self.timer
                self.user_magnebots[u_idx].item_info[o_id]['location'] = self.object_manager.transforms[o_id].position.tolist()
                
                if 'sensor' not in self.user_magnebots[u_idx].item_info[o_id]:
                    self.user_magnebots[u_idx].item_info[o_id]['sensor'] = {}

                self.raycast_request.append(str(self.user_magnebots[u_idx].robot_id))
                '''
                if not self.local:
                    self.sio.emit('objects_update', (str(self.user_magnebots[u_idx].robot_id),self.user_magnebots[u_idx].item_info))
                '''

    #Get screen coordinates of objects
    def screen_output(self, resp, screen_data, all_magnebots, all_ids):

        scre = ScreenPosition(resp)
                    
        if not scre.get_avatar_id() == "a":
            idx = self.user_magnebots_ids.index(scre.get_avatar_id())
        #print(scre.get_id(), all_magnebots[idx].screen_positions['position_ids'], scre.get_avatar_id(), idx)

        if scre.get_avatar_id() == "a" or scre.get_id() in all_magnebots[idx].screen_positions['position_ids']: #Screen coordinate was requested by particular magnebot
        
            scre_coords = scre.get_screen()

            scre_coords = (scre_coords[0],height-scre_coords[1],scre_coords[2])
            
            #print(scre_coords)
            
            if not (scre_coords[0] < 0 or scre_coords[0] > width or scre_coords[1] < 0 or scre_coords[1] > height or scre_coords[2] < 0): #Screen coordinates should not surpass limits
            
                
                temp_all_ids = all_ids + self.graspable_objects
                mid = temp_all_ids[scre.get_id()]
                color = (255, 255, 255)                        
                #print(mid)

                #Coordinates can be for a magnebot or object
                if mid in self.ai_magnebots_ids:
                    
                    viewable = False
                    
                    if not scre.get_avatar_id() == "a":
                        ai_idx = self.ai_magnebots_ids.index(mid)
                        if np.linalg.norm(all_magnebots[idx].dynamic.transform.position - self.ai_magnebots[ai_idx].dynamic.transform.position) < 12:
                            viewable = True
                        else:
                            mid = ""
                    else:
                        viewable = True
                        
                    if viewable:
                        mid = 'A_'+self.robot_names_translate[mid]
                elif mid in self.user_magnebots_ids:

                    viewable = False
                    if not scre.get_avatar_id() == "a":
                        u_idx = self.user_magnebots_ids.index(mid)
                        if np.linalg.norm(all_magnebots[idx].dynamic.transform.position - self.ai_magnebots[u_idx].dynamic.transform.position) < 12:
                            viewable = True
                        else:
                            mid = ""
                    else:
                        viewable = True
                        
                    if viewable:
                        mid = 'U_'+self.robot_names_translate[mid]
                else: #For object

                    if not scre.get_avatar_id() == "a":
                        avatar = self.user_magnebots[self.user_magnebots_ids.index(scre.get_avatar_id())]
                        
                        mid = self.object_names_translate[mid]
                        if mid in avatar.danger_estimates:
                            danger_estimate = avatar.danger_estimates[mid]
                        else:
                            danger_estimate = 0
             
                        mid = str(mid)
                    else:
                        danger_estimate = self.danger_level[mid]
                        mid = str(self.object_names_translate[mid]) + "_" +str(self.required_strength[mid])
                        
                    
      
                    if danger_estimate >= 2: #Different color for different danger estimate
                        color = (255, 0, 0)
                    elif danger_estimate == 1:
                        color = (0, 0, 255)
                    else:
                        color = (255, 255, 255)

                if mid and scre.get_avatar_id() not in screen_data:
                    screen_data[scre.get_avatar_id()] = {}
                    screen_data[scre.get_avatar_id()]['coords'] = [scre_coords]
                    screen_data[scre.get_avatar_id()]['ids'] = [mid]
                    screen_data[scre.get_avatar_id()]['color'] = [color]
                elif mid:
                    screen_data[scre.get_avatar_id()]['coords'].append(scre_coords)
                    screen_data[scre.get_avatar_id()]['ids'].append(mid)
                    screen_data[scre.get_avatar_id()]['color'].append(color)


    '''
    #Process keyboard presses
    def keyboard_output(self, resp, extra_commands, duration, keys_time_unheld, all_ids, messages):

        keys = KBoard(resp)
        
        #print(keys.get_num_pressed(), keys.get_num_held(), keys.get_num_released(), self.frame_num)

        # Listen for events where the key was first pressed on the previous frame.
    '''
    
    def checkCollision(self, m_idx, reverse):
     
        # Finding the distance of line 
        # from center.
        
        radius = 0.5
        
        y_rot = QuaternionUtils.quaternion_to_euler_angles(self.user_magnebots[m_idx].dynamic.transform.rotation)[1]
        pos = self.user_magnebots[m_idx].dynamic.transform.position
        a = math.tan((y_rot+180*int(reverse)) * math.pi / 180)
        b = 1
        c = 0
        

        for edg in self.wall_edges:
        
            distance2 = np.linalg.norm([edg[0] - pos[0], edg[1] - pos[2]])
            if distance2 < radius:
        
                dist = ((abs(a * (edg[0] - pos[0]) + b * (edg[1] - pos[2]) + c)) /
                        math.sqrt(a * a + b * b))
             
                # Checking if the distance is less 
                # than, greater than or equal to radius.
                if (radius >= dist):
                    return True
                    
        return False
            
    
    #Process keyboard presses
    def keyboard_output(self, key_pressed, key_hold, extra_commands, duration, keys_time_unheld, all_ids, messages, fps):
    
        if len(self.user_magnebots) <= 2:
            max_time_unheld_lin = 5#2
            max_time_unheld_rot = 1
        elif len(self.user_magnebots) >= 5:
            max_time_unheld_lin = 5
            max_time_unheld_rot = 5
        else:
            max_time_unheld_lin = 10#10#5 #3
            max_time_unheld_rot = 10


        #print(keys_time_unheld, key_pressed)

        
        #for j in range(keys.get_num_pressed()):
        for j in range(len(key_pressed)):
            idx = -1
            
            if key_pressed[j] in self.keys_set[0]: #Advance
                idx = self.keys_set[0].index(key_pressed[j])
                
                self.max_time_unheld = max_time_unheld_lin
                #print(self.user_magnebots[idx].action.status)
                #if self.user_magnebots[0].action.status != ActionStatus.ongoing:
                #if self.user_magnebots[idx].key_pressed != key_pressed[j] or self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                
                collision = self.checkCollision(idx, False)
                
                if collision:
                    txt = self.user_magnebots[idx].ui.add_text(text="Too close to wall, move away!",
                             position={"x": 0, "y": 0},
                             color={"r": 1, "g": 0, "b": 0, "a": 1},
                             font_size=20
                             )
                    messages.append([idx,txt,0])
                
                if not self.user_magnebots[idx].resetting_arm and not collision:
                    if self.user_magnebots[idx].key_pressed != key_pressed[j] or self.user_magnebots[idx].action.status != ActionStatus.ongoing or num_users < 5:
                        self.user_magnebots[idx].move_by(distance=10)
                    
                    if keys_time_unheld[idx] > self.max_time_unheld:
                        keys_time_unheld[idx] = int(-0.5*fps)
                    elif keys_time_unheld[idx] >= 0:
                        keys_time_unheld[idx] = 0      

                    #keys_time_unheld[idx] = -20
                    
                    self.user_magnebots[idx].key_pressed = key_pressed[j]
                    

            elif key_pressed[j] in self.keys_set[1]: #Back
                
                idx = self.keys_set[1].index(key_pressed[j])
                
                self.max_time_unheld = max_time_unheld_lin
                #print(self.user_magnebots[idx].action.status)
                #if self.user_magnebots[idx].key_pressed != key_pressed[j] or self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                
                collision = self.checkCollision(idx, True)
                
                if collision:
                    txt = self.user_magnebots[idx].ui.add_text(text="Too close to wall, move away!",
                             position={"x": 0, "y": 0},
                             color={"r": 1, "g": 0, "b": 0, "a": 1},
                             font_size=20
                             )
                    messages.append([idx,txt,0])

                
                if not self.user_magnebots[idx].resetting_arm and not collision:
                    if self.user_magnebots[idx].key_pressed != key_pressed[j] or self.user_magnebots[idx].action.status != ActionStatus.ongoing or num_users < 5:
                        self.user_magnebots[idx].move_by(distance=-10)
                    
                    if keys_time_unheld[idx] > self.max_time_unheld:
                        keys_time_unheld[idx] = int(-0.5*fps)
                    elif keys_time_unheld[idx] >= 0:
                        keys_time_unheld[idx] = 0                       
                    #keys_time_unheld[idx] = -20
                    
                    self.user_magnebots[idx].key_pressed = key_pressed[j]

            elif key_pressed[j] in self.keys_set[2]: #Right
                idx = self.keys_set[2].index(key_pressed[j])
                
                self.max_time_unheld = max_time_unheld_rot
                #print(self.user_magnebots[idx].action.status)
                #if self.user_magnebots[idx].key_pressed != key_pressed[j] or self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                
                if not self.user_magnebots[idx].resetting_arm:
                    if self.user_magnebots[idx].key_pressed != key_pressed[j] or self.user_magnebots[idx].action.status != ActionStatus.ongoing or num_users < 5:
                        self.user_magnebots[idx].turn_by(179)
                    
                    if keys_time_unheld[idx] > self.max_time_unheld:
                        keys_time_unheld[idx] = int(-0.4*fps)
                    elif keys_time_unheld[idx] >= 0:
                        keys_time_unheld[idx] = 0                        
                    #keys_time_unheld[idx] = -20
                    
                    self.user_magnebots[idx].key_pressed = key_pressed[j]

            elif key_pressed[j] in self.keys_set[3]: #Left
                idx = self.keys_set[3].index(key_pressed[j])
                
                self.max_time_unheld = max_time_unheld_rot
                #print(self.user_magnebots[idx].action.status)
                #if self.user_magnebots[idx].key_pressed != key_pressed[j] or self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                
                if not self.user_magnebots[idx].resetting_arm:
                    if self.user_magnebots[idx].key_pressed != key_pressed[j] or self.user_magnebots[idx].action.status != ActionStatus.ongoing or num_users < 5:
                        self.user_magnebots[idx].turn_by(-179)
                    
                    if keys_time_unheld[idx] > self.max_time_unheld:
                        keys_time_unheld[idx] = int(-0.4*fps)
                    elif keys_time_unheld[idx] >= 0:
                        keys_time_unheld[idx] = 0                     
                    #keys_time_unheld[idx] = -20
                    
                    self.user_magnebots[idx].key_pressed = key_pressed[j]

            elif key_pressed[j] in self.keys_set[4] or key_pressed[j] in self.keys_set[5]: #Pick up/Drop with one of the arms
                if key_pressed[j] in self.keys_set[4]:
                    arm = Arm.left
                    arm2 = Arm.right
                    key_idx = 4
                else:
                    arm = Arm.right
                    arm2 = Arm.left
                    key_idx = 5
                    
                idx = self.keys_set[key_idx].index(key_pressed[j])
                
                #print(self.user_magnebots[idx].resetting_arm)
                
                if self.user_magnebots[idx].dynamic.held[arm].size > 0 and not self.user_magnebots[idx].resetting_arm: #Press once to pick up, twice to drop
                
                    object_id = self.user_magnebots[idx].dynamic.held[arm][0]
                
                    self.user_magnebots[idx].drop(target=object_id, arm=arm, wait_for_object=False)
                    self.user_magnebots[idx].grasping = False
                    
                    """
                    if self.danger_level[object_id] == 2 and np.linalg.norm(self.object_manager.transforms[object_id].position[[0,2]]) >= float(self.cfg["goal_radius"]):
                        robot_ids,sort_indices = self.get_involved_teammates(self.user_magnebots[idx].current_teammates)
                            
                        all_magnebots = [*self.user_magnebots,*self.ai_magnebots]
                            
                        for sidx in range(int(self.required_strength[object_id])-1):
                            all_magnebots[robot_ids[sort_indices[sidx]]].stats.dropped_outside_goal.append(self.object_names_translate[object_id])
                    
                        self.user_magnebots[idx].stats.dropped_outside_goal.append(self.object_names_translate[object_id])
                    """
                    
                    
                    print("dropping", self.user_magnebots[idx].dynamic.held[arm], arm, self.required_strength[self.user_magnebots[idx].dynamic.held[arm][0]], self.danger_level[self.user_magnebots[idx].dynamic.held[arm][0]])
                    
                    self.object_dropping.append([int(self.user_magnebots[idx].dynamic.held[arm][0]),time.time(),self.user_magnebots[idx],arm])
                   

                    
                    #self.communicate([])
                    '''
                    if arm == Arm.right:
                        extra_commands.append({"$type": "detach_from_magnet", "object_id": int(self.user_magnebots[idx].dynamic.held[arm][0]), "arm": "right", "id": int(self.user_magnebots[idx].robot_id)})
                    else:
                        extra_commands.append({"$type": "detach_from_magnet", "object_id": int(self.user_magnebots[idx].dynamic.held[arm][0]), "arm": "left", "id": int(self.user_magnebots[idx].robot_id)})
                        
                    duration.append(1)
                    '''
                    '''
                    extra_commands.append({"$type":"send_raycast",
                   "origin": TDWUtils.array_to_vector3(source),
                   "destination": TDWUtils.array_to_vector3(destination),
                   "id": str(self.user_magnebots[idx].robot_id)}) 
                
                    duration.append(1)
                    self.user_magnebots[idx].dynamic.held[arm][0]
                    '''
                else: #Pick up object you have focused in
                    
                    
                    
                    
                    #Object can be too heavy to carry alone, or you may have picked the wrong object (dangerous)
                    grasp_object = self.user_magnebots[idx].focus_object

                    
                    if grasp_object and all(grasp_object not in um.dynamic.held[arm] for um in self.user_magnebots for arm in [Arm.left,Arm.right]) and not self.user_magnebots[idx].resetting_arm: #grasp_object not in self.user_magnebots[idx].dynamic.held[arm2] and grasp_object not in grabbed_objects_list:
                        print("grasping", grasp_object, arm, idx)
                        self.user_magnebots[idx].stats.grab_attempts += 1
                        
                        if self.user_magnebots[idx].strength < self.required_strength[grasp_object]:
                        
                            '''
                            if grasp_object in self.dangerous_objects:

                                pass

                                """
                                txt = self.user_magnebots[idx].ui.add_text(text="Failure! Dangerous object picked up!",
                                 position={"x": 0, "y": 0},
                                 color={"r": 1, "g": 0, "b": 0, "a": 1},
                                 font_size=20
                                 )
                                messages.append([idx,txt,0])
                                    
                                #self.sio.emit("disable", (self.robot_names_translate[str(self.user_magnebots[idx].robot_id)]))
                                self.user_magnebots[idx].disabled = True
                                self.user_magnebots[idx].stats.end_time = self.timer
                                self.user_magnebots[idx].stats.failed = 1
                                """
                                
                                #self.reset_message = True
                            else:
                            '''
                            txt = self.user_magnebots[idx].ui.add_text(text="Too heavy!!",
                             position={"x": 0, "y": 0},
                             color={"r": 1, "g": 0, "b": 0, "a": 1},
                             font_size=20
                             )
                            messages.append([idx,txt,0])
                        else:
                        
                            object_recently_dropped = False
                            for od_idx,od in enumerate(self.object_dropping):
                                if od[0] == int(grasp_object):
                                    object_recently_dropped = True
                                    break
                            
                            if object_recently_dropped:
                                del self.object_dropping[od_idx]
                            else:
                                try:
                                    print("grasping object 2")
                                    extra_commands.append({"$type": "set_mass", "mass": 1, "id": grasp_object})
                                    duration.append(1)
                                except:
                                    print("grasped object", grasp_object)

                            self.user_magnebots[idx].grasp(target=grasp_object, arm=arm)
                            self.user_magnebots[idx].grasping = True
                            self.user_magnebots[idx].grasping_time = time.time()
                            self.user_magnebots[idx].resetting_arm = True
                            
                            
                            
                            #self.communicate([])


                            """
                            #If dangerous object carried without being accompanied by an ai if human or by a human if an ai, ends the simulation
                            
                            if grasp_object in self.dangerous_objects and self.user_magnebots[idx].strength < 2:
                                
                                for um_idx,um in enumerate(self.user_magnebots):
                                    txt = um.ui.add_text(text="Failure! Dangerous object picked up!",
                                     position={"x": 0, "y": 0},
                                     color={"r": 1, "g": 0, "b": 0, "a": 1},
                                     font_size=20
                                     )
                                    messages.append([um_idx,txt,0])
                                
                                self.reset_message = True
                            
                            """

                    
            elif key_pressed[j] in self.keys_set[6]: #Move camera down
                idx = self.keys_set[6].index(key_pressed[j])
                self.user_magnebots[idx].rotate_camera(pitch=10)

            elif key_pressed[j] in self.keys_set[7]: #Move camera up
                idx = self.keys_set[7].index(key_pressed[j])
                self.user_magnebots[idx].rotate_camera(pitch=-10)

            elif key_pressed[j] in self.keys_set[8]: #Estimate danger level
                idx = self.keys_set[8].index(key_pressed[j])
                self.danger_sensor_request.append(str(self.user_magnebots[idx].robot_id))

                '''
                idx,item_info = self.danger_sensor_reading(self.user_magnebots[idx].robot_id)
                if not self.local:
                    self.sio.emit('objects_update', (str(self.user_magnebots[idx].robot_id),item_info))
                '''
                
            
            elif key_pressed[j] in self.keys_set[9]: #Focus on object, use raycasting, needs adjustment
                idx = self.keys_set[9].index(key_pressed[j])
                
                '''
                #print(angle, x_new, y_new, z_new, real_camera_position)
                camera_position_relative = np.array([-0.1838, 0.053+0.737074, 0])
                
                #print({"x": x_new, "y": y_new, "z": z_new}, self.user_magnebots[idx].dynamic.transform.position)
                r1 = Rotation.from_quat(self.user_magnebots[idx].dynamic.transform.rotation)
                r2 = Rotation.from_euler('zxy', self.user_magnebots[idx].camera_rpy, degrees=True)
                r3 = r2*r1
                print(r3.inv().apply([0,0,1])*np.array([-1,-1,1]) +self.user_magnebots[idx].dynamic.transform.position,self.user_magnebots[idx].dynamic.transform.position)
                print(r1.as_euler('xyz', degrees=True))
                
                print(r2.as_euler('zyx', degrees=True))
                new_camera_position_relative = r1.inv().apply(camera_position_relative)
                source = r3.inv().apply([0,0,0])*np.array([-1,-1,1])+self.user_magnebots[idx].dynamic.transform.position+new_camera_position_relative
                destination = r3.inv().apply([0,0,1])*np.array([-1,-1,1])+self.user_magnebots[idx].dynamic.transform.position+new_camera_position_relative

                print(source, destination, [self.object_manager.transforms[o].position for o in self.graspable_objects])
                
                extra_commands.append({"$type":"send_raycast",
                   "origin": TDWUtils.array_to_vector3(source),
                   "destination": 
                   "id": str(self.user_magnebots[idx].robot_id)}) 
                
                
                extra_commands.append({"$type":"send_raycast",
                   "origin": TDWUtils.array_to_vector3(source),
                   "destination": TDWUtils.array_to_vector3(destination),
                   "id": str(self.user_magnebots[idx].robot_id)}) 
                '''
                   
                
                #print(self.user_magnebots[idx].robot_id, idx, key_pressed, self.keys_set)
                '''
                extra_commands.append({"$type": "send_mouse_raycast",
                              "id": str(self.user_magnebots[idx].robot_id),
                              "avatar_id": str(self.user_magnebots[idx].robot_id)})
                '''
                extra_commands.append({"$type": "set_pass_masks",
                              "avatar_id": str(self.user_magnebots[idx].robot_id),
                              "pass_masks": ["_id", "_img"]})
                
                
                duration.append(1)
                
            elif key_pressed[j] in self.keys_set[10]: #Move camera left
                idx = self.keys_set[10].index(key_pressed[j])
                self.user_magnebots[idx].rotate_camera(yaw=-10)
            elif key_pressed[j] in self.keys_set[11]: #Move camera right
                idx = self.keys_set[11].index(key_pressed[j])
                self.user_magnebots[idx].rotate_camera(yaw=10)
            #elif key_pressed[j] in self.keys_set[10]: 
            #    idx = self.keys_set[9].index(key_pressed[j])
            
            #elif key_pressed[j] == 'P':
            #    self.reset = True
            
                      


        # Listen for keys currently held down. This is mainly for movement keys


        for j in range(len(key_hold)):
            #print(key_hold[j])
            idx = -1
            
            if key_hold[j] in self.keys_set[0]: #Advance
                idx = self.keys_set[0].index(key_hold[j])
                print(self.user_magnebots[idx].action.status)
                #if self.user_magnebots[0].action.status != ActionStatus.ongoing:
                #print(self.user_magnebots[idx].action.status)
                if self.user_magnebots[idx].action.status != ActionStatus.ongoing and not self.user_magnebots[idx].resetting_arm:
                    self.user_magnebots[idx].move_by(distance=10)
                
            elif key_hold[j] in self.keys_set[1]: #Back
                idx = self.keys_set[1].index(key_hold[j])
                print(self.user_magnebots[idx].action.status)
                if self.user_magnebots[idx].action.status != ActionStatus.ongoing and not self.user_magnebots[idx].resetting_arm:
                    self.user_magnebots[idx].move_by(distance=-10)
            elif key_hold[j] in self.keys_set[2]: #Right
                idx = self.keys_set[2].index(key_hold[j])
                print(self.user_magnebots[idx].action.status)
                if self.user_magnebots[idx].action.status != ActionStatus.ongoing and not self.user_magnebots[idx].resetting_arm:
                    self.user_magnebots[idx].turn_by(179)
            elif key_hold[j] in self.keys_set[3]: #Left
                idx = self.keys_set[3].index(key_hold[j])
                print(self.user_magnebots[idx].action.status)
                if self.user_magnebots[idx].action.status != ActionStatus.ongoing and not self.user_magnebots[idx].resetting_arm:
                    self.user_magnebots[idx].turn_by(-179)
         
       
            if idx >= 0:
                keys_time_unheld[idx] = 0
            
        '''
        # Listen for keys that were released. DOESN'T WORK
        for j in range(keys.get_num_released()):
            pdb.set_trace()
            if keys.get_released(j) == 'UpArrow':
                #if self.user_magnebots[0].action.status != ActionStatus.ongoing:
                print('stop')
                self.user_magnebots[0].stop()

            elif keys.get_released(j) == 'DownArrow':
             
                self.user_magnebots[0].stop()
            elif keys.get_released(j) == 'RightArrow':
                self.user_magnebots[0].stop()
            elif keys.get_released(j) == 'LeftArrow':
                self.user_magnebots[0].stop()

        '''
        
        if len(key_hold) == 0: #After some time unheld we stop the current action
            #print(keys_time_unheld)
            for um_idx in range(len(self.user_magnebots)):
                keys_time_unheld[um_idx] += 1
                #print(keys_time_unheld[um_idx])
                if keys_time_unheld[um_idx] == self.max_time_unheld and not self.user_magnebots[um_idx].resetting_arm: #3
                    print("stop magnebot")
                    self.user_magnebots[um_idx].stop()


    

    def danger_sensor_reading(self, robot_id):
    
        all_ids = [*self.user_magnebots_ids,*self.ai_magnebots_ids]
        all_magnebots = [*self.user_magnebots,*self.ai_magnebots]
        idx = all_ids.index(str(robot_id))
        ego_magnebot = all_magnebots[idx]
    
        near_items_pos = []
        near_items_idx = []
        danger_estimates = {}
        possible_danger_levels = [1,2]
        
        if ego_magnebot.refresh_sensor >= global_refresh_sensor: #Check if our sensor is refreshed
            ego_magnebot.refresh_sensor = 0
            
            ego_magnebot.stats.sensor_activation += 1

            #Roooms
            ego_room = -1
            for r in self.rooms_limits.keys():
                #print([ego_magnebot.dynamic.transform.position[0], ego_magnebot.dynamic.transform.position[2]], self.rooms[r][0],self.rooms[r][-1])
                if ego_magnebot.dynamic.transform.position[0] >= self.rooms_limits[r][0][0] and ego_magnebot.dynamic.transform.position[2] >= self.rooms_limits[r][0][1] and ego_magnebot.dynamic.transform.position[0] <= self.rooms_limits[r][-1][0] and ego_magnebot.dynamic.transform.position[2] <= self.rooms_limits[r][-1][1]:
                    ego_room = r
                    break
            
            for o_idx,o in enumerate(self.graspable_objects): #Sensor only actuates over objects that are in a certain radius
            
                #Roooms
                object_room = -1
                for r in self.rooms_limits.keys():
                    if self.object_manager.transforms[o].position[0] >= self.rooms_limits[r][0][0] and self.object_manager.transforms[o].position[2] >= self.rooms_limits[r][0][1] and self.object_manager.transforms[o].position[0] <= self.rooms_limits[r][-1][0] and self.object_manager.transforms[o].position[2] <= self.rooms_limits[r][-1][1]:
                        object_room = r
                        break
            
                #print(object_room, ego_room)
                if np.linalg.norm(self.object_manager.transforms[o].position - ego_magnebot.dynamic.transform.position) < int(self.cfg['sensing_radius']) and not any(doIntersect([self.object_manager.transforms[o].position[0],self.object_manager.transforms[o].position[2]],[ego_magnebot.dynamic.transform.position[0],ego_magnebot.dynamic.transform.position[2]],[self.walls[w_idx][0][0],self.walls[w_idx][0][1]],[self.walls[w_idx][-1][0],self.walls[w_idx][-1][1]]) for w_idx in range(len(self.walls))) and ego_room == object_room:
                        
                    #print([doIntersect([self.object_manager.transforms[o].position[0],self.object_manager.transforms[o].position[2]],[ego_magnebot.dynamic.transform.position[0],ego_magnebot.dynamic.transform.position[2]],[self.walls[w_idx][0][0],self.walls[w_idx][0][1]],[self.walls[w_idx][-1][0],self.walls[w_idx][-1][1]]) for w_idx in range(len(self.walls))])
                    
                    #pdb.set_trace()
                        
                    near_items_idx.append(len(all_ids)+o_idx)
                    near_items_pos.append(TDWUtils.array_to_vector3(self.object_manager.transforms[o].position))
                    actual_danger_level = self.danger_level[o]
                    
                    o_translated = self.object_names_translate[o]
                    
                    if o_translated not in ego_magnebot.item_info:
                        ego_magnebot.item_info[o_translated] = {}
                        
                    ego_magnebot.item_info[o_translated]['weight'] = int(self.required_strength[o])
                    
                    if 'sensor' not in ego_magnebot.item_info[o_translated]:
                        ego_magnebot.item_info[o_translated]['sensor'] = {}
                    
                    
                    robot_id_translated = self.robot_names_translate[str(ego_magnebot.robot_id)]
                    
                    #print("Sensed",robot_id_translated, o_translated, ego_magnebot.item_info[o_translated])
                    
                    #Get danger estimation, value and confidence level
                    if robot_id_translated not in ego_magnebot.item_info[o_translated]['sensor']:
                        possible_danger_levels_tmp = possible_danger_levels.copy()
                        possible_danger_levels_tmp.remove(actual_danger_level)
                    
                        if actual_danger_level == 1:
                            true_accuracy = ego_magnebot.p11
                        elif actual_danger_level == 2:
                            true_accuracy = ego_magnebot.p22
                        #estimate_accuracy = float(np.random.uniform(0.5, 1))
                    
                        danger_estimate = random.choices([actual_danger_level,*possible_danger_levels_tmp],weights=[true_accuracy*100,(1-true_accuracy)*100])
                        danger_estimates[o_translated] = danger_estimate[0]
                        
                        if danger_estimate[0] == 1:
                            estimate_accuracy = ego_magnebot.p11
                            other_estimate_accuracy = 1-ego_magnebot.p22
                            
                            prior = 1 - len(self.dangerous_objects)/len(self.graspable_objects)
                            other_prior = 1 - prior
                            
                        elif danger_estimate[0] == 2:
                            estimate_accuracy = ego_magnebot.p22
                            other_estimate_accuracy = 1-ego_magnebot.p11
                            
                            prior = len(self.dangerous_objects)/len(self.graspable_objects) 
                            other_prior = 1 - prior
                        
                        ego_magnebot.item_info[o_translated]['sensor'][robot_id_translated] = {}
                        ego_magnebot.item_info[o_translated]['sensor'][robot_id_translated]['value'] = int(danger_estimate[0])
                        ego_magnebot.item_info[o_translated]['sensor'][robot_id_translated]['confidence'] = (estimate_accuracy*0.5)/(0.5*estimate_accuracy + 0.5*other_estimate_accuracy)

                        
                    else: #If we already have a danger estimation reuse that one
                        danger_estimates[o_translated] = ego_magnebot.item_info[o_translated]['sensor'][robot_id_translated]['value']
                        
                        
                    ego_magnebot.item_info[o_translated]['time'] = self.timer
                    ego_magnebot.item_info[o_translated]['location'] = self.object_manager.transforms[o].position.tolist()
                        
            #If objects were detected
            if near_items_pos:
                
                #To have the information displayed in the screen
                ego_magnebot.screen_positions["position_ids"].extend(near_items_idx)
                ego_magnebot.screen_positions["positions"].extend(near_items_pos)
                ego_magnebot.screen_positions["duration"].extend([100]*len(near_items_idx)) 
                
                ego_magnebot.danger_estimates = danger_estimates
        '''         
                if not self.local:
                    #print("objects_update", (idx,ego_magnebot.item_info))
                    self.sio.emit('objects_update', (idx,ego_magnebot.item_info))
            else:
                if not self.local:
                    self.sio.emit('objects_update', (idx,ego_magnebot.item_info))
        else:
            self.sio.emit('objects_update', (idx,ego_magnebot.item_info))
        '''
        
        return idx, ego_magnebot.item_info
            
    def send_occupancy_map(self, magnebot_id):
    
        self.occupancy_map_request.append(magnebot_id)
        #print("occupancy request", self.occupancy_map_request)
        
        
    def send_objects_held_status(self,magnebot_id):
    
        self.objects_held_status_request.append(magnebot_id)
        
    def send_danger_sensor_reading(self, magnebot_id):
    
        self.danger_sensor_request.append(magnebot_id)
            
    def get_occupancy_map(self, magnebot_id):
                    
        
        m_idx = self.ai_magnebots_ids.index(magnebot_id)
        locations_magnebot_map = {}
        
        magnebots_locations = np.where(self.object_type_coords_map == 3)
        
        locations_magnebot_map = {str(j[1]):[magnebots_locations[0][i],magnebots_locations[1][i]] for i in range(len(magnebots_locations[0])) for j in self.object_attributes_id[str(magnebots_locations[0][i])+'_'+str(magnebots_locations[1][i])]}


        try:
            x = locations_magnebot_map[self.robot_names_translate[str(magnebot_id)]][0]
            y = locations_magnebot_map[self.robot_names_translate[str(magnebot_id)]][1]
        except:
            pdb.set_trace()
        
        
        #print("Hello Sending occupancy map", self.occupancy_map_request, magnebot_id, self.ai_magnebots[m_idx].view_radius)
        if magnebot_id in self.occupancy_map_request and self.ai_magnebots[m_idx].view_radius:
            
            print("Sending occupancy map")
            self.occupancy_map_request.remove(magnebot_id)
        

            view_radius = self.ai_magnebots[m_idx].view_radius
            
            

            x_min = max(0,x-view_radius)
            y_min = max(0,y-view_radius)
            x_max = min(self.object_type_coords_map.shape[0]-1,x+view_radius)
            y_max = min(self.object_type_coords_map.shape[1]-1,y+view_radius)
            #limited_map = np.zeros_like(self.static_occupancy_map.occupancy_map)
            
            #Magnebot is at the center of the occupancy  map always or not
            if self.ai_magnebots[m_idx].centered_view:
                limited_map = np.zeros((view_radius*2+1,view_radius*2+1)) #+1 as we separately count the row/column where the magnebot is currently in
                #limited_map[:,:] = self.static_occupancy_map.occupancy_map[x_min:x_max+1,y_min:y_max+1]
                limited_map[:,:] = self.object_type_coords_map[x_min:x_max+1,y_min:y_max+1]
                
                objects_locations = np.where(limited_map > 1)
                reduced_metadata = {}
                limited_map[x-x_min,y-y_min] = 5

                for ol in range(len(objects_locations[0])):
                    rkey = str(objects_locations[0][ol]+x_min)+'_'+str(objects_locations[1][ol]+y_min)
                    rkey2 = str(objects_locations[0][ol])+'_'+str(objects_locations[1][ol])
                    reduced_metadata[rkey2] = self.object_attributes_id[rkey]
            else:
                limited_map = np.ones_like(self.object_type_coords_map)*(-2)
                limited_map[[0,limited_map.shape[0]-1],:] = -1
                limited_map[:,[0,limited_map.shape[1]-1]] = -1
                limited_map[x_min:x_max+1,y_min:y_max+1] = self.object_type_coords_map[x_min:x_max+1,y_min:y_max+1]
                
                if self.ai_magnebots[m_idx].visibility_matrix and self.ai_magnebots[m_idx].visibility_matrix[x][y]:
                    m = np.array(self.ai_magnebots[m_idx].visibility_matrix[x][y])
                    m_inds = np.where(m == -2)  
                    if m_inds[0].size:
                        limited_map[x_min:x_max+1,y_min:y_max+1][m_inds] = -2
                
                objects_locations = np.where(limited_map > 1)
                
                reduced_metadata = {}
                
                if self.options.agents_localized:
                    agents = np.where(self.object_type_coords_map == 3)
                    for ag_idx in range(len(agents[0])):
                        if (agents[0][ag_idx] < x_min or agents[0][ag_idx] > x_max) or (agents[1][ag_idx] < y_min or agents[1][ag_idx] > y_max):
                            limited_map[agents[0][ag_idx],agents[1][ag_idx]] = 3
                            
                            rkey = str(agents[0][ag_idx])+'_'+str(agents[1][ag_idx])
                            
                            elements = []
                            
                            for element in self.object_attributes_id[rkey]:
                                if element[0]:
                                    elements.append(element)
                            
                            reduced_metadata[rkey] = elements
                    
                    #Make sure to track objects that have been picked up by others
                    for metadata_key in self.object_attributes_id.keys():
                        for el in self.object_attributes_id[metadata_key]:
                            if not el[0] and el[4]:
                                if metadata_key not in reduced_metadata.keys():
                                    reduced_metadata[metadata_key] = []
                                reduced_metadata[metadata_key].append(el)
                                    
                            
                    
                                
                limited_map[x,y] = 5
                
                
                
                for ol in range(len(objects_locations[0])):
                    rkey = str(objects_locations[0][ol])+'_'+str(objects_locations[1][ol])
                    reduced_metadata[rkey] = self.object_attributes_id[rkey]
                    
                
                walls = np.where(self.object_type_coords_map == 1)
                
                for w in range(len(walls[0])):
                    limited_map[walls[0][w],walls[1][w]] = 1
                
            """
            for ol in range(len(objects_locations[0])):
                rkey = str(objects_locations[0][ol]+x_min)+str(objects_locations[1][ol]+y_min)
                pdb.set_trace()
                if magnebot_id in object_attributes_id[rkey]:
                    limited_map[x,y] = 5
                else:
                    limited_map[objects_locations[0][ol],objects_locations[1][ol]] = reduced_object_type_coords_map[objects_locations[0][ol],objects_locations[1][ol]]
                    rkey2 = str(objects_locations[0][ol])+str(objects_locations[1][ol])
                    
                    reduced_metadata[rkey2] = object_attributes_id[rkey]
            
            
            for om in range(len(magnebots_locations[0])):
                if magnebots_locations[0][om] >= x_min and magnebots_locations[0][om] <= x_max and magnebots_locations[1][om] >= y_min and magnebots_locations[1][om] <= y_max:
                    rkey = str(magnebots_locations[0][om])+str(magnebots_locations[1][om])
                    if not magnebot_id in object_attributes_id[rkey]:
                        limited_map[magnebots_locations[0][om]-x_min,magnebots_locations[1][om]-y_min] = 3
                        rkey2 = str(magnebots_locations[0][om]-x_min)+str(magnebots_locations[1][om]-y_min)
                        reduced_metadata[rkey2] = object_attributes_id[rkey]
                
            """
            #print(limited_map)

            #pdb.set_trace()
            #limited_map[x_min:x_max+1,y_min:y_max+1] = self.static_occupancy_map.occupancy_map[x_min:x_max+1,y_min:y_max+1]
            
            
       
            
            #self.sio.emit('occupancy_map', (all_idx, json_numpy.dumps(limited_map), reduced_metadata, objects_held))
            
        else: #If only location of robot
            limited_map = np.ones_like(self.object_type_coords_map)*(-2)
            
            reduced_metadata = {}
            
            '''
            if self.options.agents_localized:
                agents = np.where(self.object_type_coords_map == 3)
                for ag_idx in range(len(agents[0])):
                    limited_map[agents[0][ag_idx],agents[1][ag_idx]] = 3
                    
                    rkey = str(agents[0][ag_idx])+'_'+str(agents[1][ag_idx])
                    
                    elements = []
                    
                    for element in self.object_attributes_id[rkey]:
                        if element[0]:
                            elements.append(element)
                    
                    reduced_metadata[rkey] = elements
            '''
            
            limited_map[x,y] = 5
            #objects_locations = np.where(self.object_type_coords_map[x-1:x+2:y-1:y+2] == 2) #Object metadata only for surrounding objects
            objects_locations = np.where(self.object_type_coords_map > 1)
            for ol in range(len(objects_locations[0])):
                rkey = str(objects_locations[0][ol])+'_'+str(objects_locations[1][ol])
                reduced_metadata[rkey] = self.object_attributes_id[rkey]
            
            '''
            held_objects = np.where(self.object_type_coords_map == 4)
            
            held_objects_ego = [] #Get objects held by this robot
            for arm in [Arm.left,Arm.right]:
                held_objects_ego.extend(self.ai_magnebots[m_idx].dynamic.held[arm])
           
            if held_objects_ego: #If there are objects held by this robot, display it in the occupancy map
                for ho_idx in range(len(held_objects[0])):
                    for obi in self.object_attributes_id[str(held_objects[0][ho_idx])+'_'+str(held_objects[1][ho_idx])]:
                        if obi[0] in held_objects_ego:
                            limited_map[held_objects[0][ho_idx],held_objects[1][ho_idx]] = 4
            '''

            
        return limited_map, reduced_metadata
        
    def get_objects_held_state(self, all_magnebots, all_ids, magnebot_id):
    
        #Check the objects held in each arm
        objects_held = [0,0]
        m_idx = all_ids.index(magnebot_id)
        
        if magnebot_id in self.objects_held_status_request:
            self.objects_held_status_request.remove(magnebot_id)
            
        for arm_idx, arm in enumerate([Arm.left,Arm.right]):
            
            if all_magnebots[m_idx].dynamic.held[arm].size > 0:
                objects_held[arm_idx] = self.object_names_translate[int(all_magnebots[m_idx].dynamic.held[arm][0])]

        #print("objects_held", objects_held, type(objects_held[0]))
        return objects_held
    
    
    def track_objects_carried(self, all_magnebots, all_idx, item_info, messages):
    
        #Track objects being carried
        
        for arm in [Arm.left,Arm.right]:
            if all_magnebots[all_idx].dynamic.held[arm].size > 0:
                object_id = all_magnebots[all_idx].dynamic.held[arm][0]
                object_id_translated = self.object_names_translate[object_id]
                
                if object_id_translated not in all_magnebots[all_idx].item_info:
                    all_magnebots[all_idx].item_info[object_id_translated] = {}
                
                all_magnebots[all_idx].item_info[object_id_translated]["time"] = self.timer
                all_magnebots[all_idx].item_info[object_id_translated]["location"] = self.object_manager.transforms[object_id].position.tolist()
                all_magnebots[all_idx].item_info[object_id_translated]["weight"] = int(self.required_strength[object_id])
                

                if object_id not in all_magnebots[all_idx].stats.objects_in_goal and any(np.linalg.norm(self.object_manager.transforms[object_id].position[[0,2]]- np.array(goal[1])) < goal[0] for goal in self.goal_area):
                    if "sensor" not in all_magnebots[all_idx].item_info[object_id_translated]:
                        all_magnebots[all_idx].item_info[object_id_translated]["sensor"] = {}
                    
                    robot_id_translated = self.robot_names_translate[str(all_magnebots[all_idx].robot_id)]
                    
                    if robot_id_translated not in all_magnebots[all_idx].item_info[object_id_translated]["sensor"]:
                        all_magnebots[all_idx].item_info[object_id_translated]["sensor"][robot_id_translated] = {}
                        
                    all_magnebots[all_idx].item_info[object_id_translated]["sensor"][robot_id_translated]['value'] = int(self.danger_level[object_id])
                    all_magnebots[all_idx].item_info[object_id_translated]["sensor"][robot_id_translated]['confidence'] = 1
                        
                    if self.object_names_translate[object_id] not in self.already_collected and self.object_names_translate[object_id] not in all_magnebots[all_idx].stats.objects_in_goal:
                    
                        all_magnebots[all_idx].stats.objects_in_goal.append(self.object_names_translate[object_id])
                        self.already_collected.append(self.object_names_translate[object_id])
                        if self.danger_level[object_id] == 1 and all_magnebots[all_idx].ui_elements:
                            messages.append(self.info_message_ui(all_magnebots, all_idx, "Penalty! Benign object disposed!", "red"))
                        
                        
                        
                        if int(self.required_strength[object_id]) > 1: #Add teammates contribution
                        
                            robot_ids,sort_indices = self.get_involved_teammates(all_magnebots[all_idx].current_teammates, self.object_names_translate[object_id])
                            
                            for sidx in range(int(self.required_strength[object_id])-1):
                            
                                if sidx < len(sort_indices):
                            
                                    all_magnebots[robot_ids[sort_indices[sidx]]].stats.objects_in_goal.append(self.object_names_translate[object_id])
                                    if self.danger_level[object_id] == 1 and all_magnebots[robot_ids[sort_indices[sidx]]].ui_elements:
                                        messages.append(self.info_message_ui(all_magnebots, robot_ids[sort_indices[sidx]], "Penalty! Benign object disposed!", "red"))
                    
                        if self.danger_level[object_id] == 2 and self.object_names_translate[object_id] not in all_magnebots[all_idx].stats.dangerous_objects_in_goal:
                        
                            
                            all_magnebots[all_idx].stats.dangerous_objects_in_goal.append(self.object_names_translate[object_id])
                            
                            if all_magnebots[all_idx].ui_elements:
                                messages.append(self.info_message_ui(all_magnebots, all_idx, "Reward! Dangerous object disposed!", "blue"))
                            
                            
                            if int(self.required_strength[object_id]) > 1: #Add teammates contribution
                            
                                for sidx in range(int(self.required_strength[object_id])-1):
                                    all_magnebots[robot_ids[sort_indices[sidx]]].stats.dangerous_objects_in_goal.append(self.object_names_translate[object_id])
                                    if all_magnebots[robot_ids[sort_indices[sidx]]].ui_elements:
                                        messages.append(self.info_message_ui(all_magnebots, robot_ids[sort_indices[sidx]], "Reward! Dangerous object disposed!", "blue"))
                            
                      



                item_info[self.object_names_translate[object_id]] = all_magnebots[all_idx].item_info[self.object_names_translate[object_id]]
                
     
    class Tutorial_State(Enum):
        start = 0
        move = 1
        move_to_object = 2
        activate_sensor = 3
        pickup_object = 4
        move_to_goal = 5
        drop_object = 6
        move_to_heavy_object = 7
        activate_sensor_heavy = 8
        move_to_agent = 9
        ask_sensor = 10
        end = 11
          
                
    def tutorial_state_machine(self, all_magnebots, commands):
    
        
        for um_idx,um in enumerate(all_magnebots):
        
            if str(um.robot_id) in self.user_magnebots_ids:
            
                cell_size = self.cfg['cell_size']
                
                if self.state_machine[um_idx] == self.Tutorial_State.start:
                
                    xn = self.wall_length[0]*um_idx + self.wall_length[0]/2 -self.scenario_size[0]/2
                    zn = self.wall_length[1]/2 -self.scenario_size[1]/2
                    
                    commands.append(self.get_add_visual_effect(name="fire", 
                                               position={"x": xn, "y": 0, "z": zn},
                                               effect_id=um.robot_id))
                                               
                    self.state_machine[um_idx] = self.Tutorial_State.move
                    
                    self.sio.emit("tutorial", (self.robot_names_translate[str(all_magnebots[um_idx].robot_id)], "move"))
    
                elif self.state_machine[um_idx] == self.Tutorial_State.move:
                
                    xn = self.wall_length[0]*um_idx + self.wall_length[0]/2 -self.scenario_size[0]/2
                    zn = self.wall_length[1]/2 -self.scenario_size[1]/2
                
                    if np.linalg.norm(np.array([xn,zn]) - um.dynamic.transform.position[[0,2]]) < 1: #Arrived to location
                    
                        self.state_machine[um_idx] = self.Tutorial_State.move_to_object
                        commands.append({"$type": "destroy_visual_effect", "id": um.robot_id})
                        
                        
                        xn = self.wall_length[0]*um_idx + cell_size*3 - self.scenario_size[0]/2
                        zn = cell_size*7 - self.scenario_size[1]/2
                        
                        commands.append(self.get_add_visual_effect(name="fire", 
                                               position={"x": xn, "y": 0, "z": zn},
                                               effect_id=um.robot_id))
                                               
                        self.sio.emit("tutorial", (self.robot_names_translate[str(all_magnebots[um_idx].robot_id)], "move_to_object"))
                        
                elif self.state_machine[um_idx] == self.Tutorial_State.move_to_object:
                
                    xn = self.wall_length[0]*um_idx + cell_size*3 - self.scenario_size[0]/2
                    zn = cell_size*7 - self.scenario_size[1]/2     
                    
                    if np.linalg.norm(np.array([xn,zn]) - um.dynamic.transform.position[[0,2]]) < 1: #Arrived to location   
                        self.state_machine[um_idx] = self.Tutorial_State.move_to_object
                        commands.append({"$type": "destroy_visual_effect", "id": um.robot_id})
                        
                        self.state_machine[um_idx] = self.Tutorial_State.activate_sensor
                        
                        self.sio.emit("tutorial", (self.robot_names_translate[str(all_magnebots[um_idx].robot_id)], "activate_sensor"))
                        
                elif self.state_machine[um_idx] == self.Tutorial_State.activate_sensor:
                
                    o_translated1 = str(1+um_idx*4)
                    o_translated2 = str(2+um_idx*4)
                    o_translated3 = str(3+um_idx*4)
                    
                    if o_translated1 in um.item_info and "sensor" in um.item_info[o_translated1] and um.item_info[o_translated1]['sensor'] and o_translated2 in um.item_info and "sensor" in um.item_info[o_translated2] and um.item_info[o_translated2]['sensor'] and o_translated3 in um.item_info and "sensor" in um.item_info[o_translated3] and um.item_info[o_translated3]['sensor']:
                    
                        self.state_machine[um_idx] = self.Tutorial_State.pickup_object
                        
                        object_id = list(self.object_names_translate.keys())[list(self.object_names_translate.values()).index(str(2+um_idx*4))]
                        xn = float(self.object_manager.transforms[object_id].position[0] + 0.5)
                        zn = float(self.object_manager.transforms[object_id].position[2])
                        
                        commands.append(self.get_add_visual_effect(name="fire", 
                                               position={"x": xn, "y": 0, "z": zn},
                                               effect_id=um.robot_id))
                        
                        self.sio.emit("tutorial", (self.robot_names_translate[str(all_magnebots[um_idx].robot_id)], "pickup_object"))
                        
                elif self.state_machine[um_idx] == self.Tutorial_State.pickup_object:
                
                    object_id = list(self.object_names_translate.keys())[list(self.object_names_translate.values()).index(str(2+um_idx*4))]
                            
                    for arm in [Arm.left,Arm.right]:
                        if object_id in um.dynamic.held[arm]:
                            
                            xn = self.wall_length[0]*um_idx + self.wall_length[0]/2 -self.scenario_size[0]/2
                            zn = self.wall_length[1]/2 -self.scenario_size[1]/2
                            commands.append({"$type": "destroy_visual_effect", "id": um.robot_id})
                            commands.append(self.get_add_visual_effect(name="fire", 
                                               position={"x": xn, "y": 0, "z": zn},
                                               effect_id=um.robot_id))
                        
                            self.state_machine[um_idx] = self.Tutorial_State.move_to_goal
                            
                            self.sio.emit("tutorial", (self.robot_names_translate[str(all_magnebots[um_idx].robot_id)], "move_to_goal"))
                            
                elif self.state_machine[um_idx] == self.Tutorial_State.move_to_goal:
                
                    xn = self.wall_length[0]*um_idx + self.wall_length[0]/2 -self.scenario_size[0]/2
                    zn = self.wall_length[1]/2 -self.scenario_size[1]/2
                    
                    object_id = list(self.object_names_translate.keys())[list(self.object_names_translate.values()).index(str(2+um_idx*4))]
                    
                    for arm in [Arm.left,Arm.right]:
                        if object_id in um.dynamic.held[arm]:
                            if np.linalg.norm(np.array([xn,zn]) - um.dynamic.transform.position[[0,2]]) < 1: #Arrived to location
                                commands.append({"$type": "destroy_visual_effect", "id": um.robot_id})
                                self.state_machine[um_idx] = self.Tutorial_State.drop_object
                                
                                self.sio.emit("tutorial", (self.robot_names_translate[str(all_magnebots[um_idx].robot_id)], "drop_object"))
                        
                elif self.state_machine[um_idx] == self.Tutorial_State.drop_object:
                
                    still_carrying = False
                    
                    object_id = list(self.object_names_translate.keys())[list(self.object_names_translate.values()).index(str(2+um_idx*4))]
                    
                    for arm in [Arm.left,Arm.right]:
                        if object_id in um.dynamic.held[arm]:
                            still_carrying = True
                            
                    if not still_carrying:
                    
                        xn = self.wall_length[0]*um_idx + cell_size*1.5 - self.scenario_size[0]/2
                        zn = cell_size*2 - self.scenario_size[1]/2
                    
                        commands.append(self.get_add_visual_effect(name="fire", 
                                               position={"x": xn, "y": 0, "z": zn},
                                               effect_id=um.robot_id))
                    
                        self.state_machine[um_idx] = self.Tutorial_State.move_to_heavy_object
                        
                        self.sio.emit("tutorial", (self.robot_names_translate[str(all_magnebots[um_idx].robot_id)], "move_to_heavy_object"))
                        
                elif self.state_machine[um_idx] == self.Tutorial_State.move_to_heavy_object:
                
                    xn = self.wall_length[0]*um_idx + cell_size*1.5 - self.scenario_size[0]/2
                    zn = cell_size*2 - self.scenario_size[1]/2
                
                    if np.linalg.norm(np.array([xn,zn]) - um.dynamic.transform.position[[0,2]]) < 1: #Arrived to location
                        commands.append({"$type": "destroy_visual_effect", "id": um.robot_id})
                        self.state_machine[um_idx] = self.Tutorial_State.activate_sensor_heavy
                        
                        self.sio.emit("tutorial", (self.robot_names_translate[str(all_magnebots[um_idx].robot_id)], "activate_sensor_heavy"))
                        
                elif self.state_machine[um_idx] == self.Tutorial_State.activate_sensor_heavy:
                
                    o_translated = str(um_idx*4)
                    
                    if o_translated in um.item_info and "sensor" in um.item_info[o_translated]:
                        
                        xn = self.wall_length[0]*um_idx + cell_size*4.5 - self.scenario_size[0]/2
                        zn = cell_size*4.5 - self.scenario_size[1]/2
                        
                        commands.append(self.get_add_visual_effect(name="fire", 
                                               position={"x": xn, "y": 0, "z": zn},
                                               effect_id=um.robot_id))
                    
                        self.state_machine[um_idx] = self.Tutorial_State.move_to_agent
                        
                        self.sio.emit("tutorial", (self.robot_names_translate[str(all_magnebots[um_idx].robot_id)], "move_to_agent"))
                        
                elif self.state_machine[um_idx] == self.Tutorial_State.move_to_agent:
                
                    xn = self.wall_length[0]*um_idx + cell_size*4.5 - self.scenario_size[0]/2
                    zn = cell_size*4.5 - self.scenario_size[1]/2

                    if np.linalg.norm(np.array([xn,zn]) - um.dynamic.transform.position[[0,2]]) < 1: #Arrived to location
                        commands.append({"$type": "destroy_visual_effect", "id": um.robot_id})
                        self.state_machine[um_idx] = self.Tutorial_State.ask_sensor
                        self.sio.emit("tutorial", (self.robot_names_translate[str(all_magnebots[um_idx].robot_id)], "ask_sensor"))
                        
                elif self.state_machine[um_idx] == self.Tutorial_State.ask_sensor:
                
                    xn = self.wall_length[0]*um_idx + self.wall_length[0]/2 -self.scenario_size[0]/2
                    zn = self.wall_length[1]/2 -self.scenario_size[1]/2
                    
                    
                    
                    object_id = list(self.object_names_translate.keys())[list(self.object_names_translate.values()).index(str(um_idx*4))]
                    
                    still_carrying = False
                    for arm in [Arm.left,Arm.right]:
                        if object_id in um.dynamic.held[arm]:
                            still_carrying = True
                    
                    #print(still_carrying, self.object_manager.transforms[object_id].position[[0,2]], np.linalg.norm(np.array([xn,zn]) - self.object_manager.transforms[object_id].position[[0,2]]))
                    if not still_carrying and np.linalg.norm(np.array([xn,zn]) - self.object_manager.transforms[object_id].position[[0,2]]) < 1: #Arrived to location
                        self.sio.emit("tutorial", (self.robot_names_translate[str(all_magnebots[um_idx].robot_id)], "end"))
                        self.state_machine[um_idx] = self.Tutorial_State.end
                        
                               



    #### Main Loop
    def run(self):
    
        global game_finished
    
        done = False
        commands = []
        key = ""
        messages = []
        
        
        estimated_fps = 0
        past_time = time.time()
        self.frame_num = 0
        past_timer = self.timer
        goal_counter = 0
        sim_elapsed_time = 0
        disabled_robots = []
        dropped_objects = []
        dropped_objects_message_counter = 0
        
        
        keys_time_unheld = [0]*len(self.user_magnebots_ids)
        all_ids = [*self.user_magnebots_ids,*self.ai_magnebots_ids]
        all_magnebots = [*self.user_magnebots,*self.ai_magnebots]

        
        
        #Include the positions of other magnebots in the view of all user magnebots
        for um in self.user_magnebots:
            um.screen_positions["position_ids"].extend(list(range(0,len(all_ids))))
            um.screen_positions["positions"].extend([-1]*len(all_ids))
            um.screen_positions["duration"].extend([-1]*len(all_ids))
        
            
        time_gone = time.time()
        
        print("User ids: ", self.user_magnebots_ids, "AI ids: ", self.ai_magnebots_ids)
        

        #Loop until simulation ends
        while not done:
            start_time = time.time()
            
            screen_positions = {"position_ids":[],"positions":[]}
            
            #Track magnebots positions
            user_magnebots_positions = [TDWUtils.array_to_vector3(um.dynamic.transform.position + np.array([0,0.5,0])) for um in self.user_magnebots]
            ai_magnebots_positions = [TDWUtils.array_to_vector3(um.dynamic.transform.position + np.array([0,0.5,0])) for um in self.ai_magnebots]
            
            
            held_objects = []
            held_objects_agent = {}
            for um in all_magnebots:
                for arm in [Arm.left,Arm.right]:
                    if um.dynamic.held[arm].size > 0:
                        held_objects.append(um.dynamic.held[arm][0])
                        held_objects_agent[um.dynamic.held[arm][0]] = self.robot_names_translate[str(um.robot_id)]
                
            
            #Prepare occupancy maps and associated metadata
            #object_attributes_id stores the ids of the objects and magnebots
            #object_type_coords_map creates a second occupancy map with objects and magnebots

            self.object_type_coords_map = np.copy(self.static_occupancy_map.occupancy_map)
            min_pos = [float(self.static_occupancy_map.positions[0,0,0]),float(self.static_occupancy_map.positions[0,0,1])] #self.static_occupancy_map.get_occupancy_position(0,0)[0]
            multiple = self.cfg['cell_size']
            self.object_attributes_id = {}
            
            try:
                #Environmental objects
                for o in self.env_objects:
                    pos = self.object_manager.transforms[o].position
                    pos_new = [round((pos[0]+abs(min_pos[0]))/multiple), round((pos[2]+abs(min_pos[1]))/multiple)]
                    self.object_type_coords_map[pos_new[0],pos_new[1]] = 1
            except:
                pdb.set_trace()
            
            #Graspable objects
            for o in self.graspable_objects:
                pos = self.object_manager.transforms[o].position
                pos_new = [round((pos[0]+abs(min_pos[0]))/multiple), round((pos[2]+abs(min_pos[1]))/multiple)]
                #2 is for objects
                
                ob_type = 2
                
                if o in held_objects:
                    ob_type = 4
                    
                try:
                    if not (self.object_type_coords_map[pos_new[0],pos_new[1]] and ob_type == 4): #If robot carrying object, do not substitute cell with held object
                        self.object_type_coords_map[pos_new[0],pos_new[1]] = ob_type
                except:
                    pdb.set_trace()

                if str(pos_new[0])+'_'+str(pos_new[1]) not in self.object_attributes_id:
                    self.object_attributes_id[str(pos_new[0])+'_'+str(pos_new[1])] = []
                    
                if o in held_objects_agent.keys():
                    carried_by = held_objects_agent[o]
                else:
                    carried_by = ""
                
                self.object_attributes_id[str(pos_new[0])+'_'+str(pos_new[1])].append((0,self.object_names_translate[o],self.required_strength[o],int(self.danger_level[o]), carried_by))
            
            if self.options.save_map and not self.timer:
                map_f = open("maps/map"+ str(self.reset_number) + ".json", "w")
                json.dump({"map": self.object_type_coords_map.tolist(), "attributes": self.object_attributes_id}, map_f)
                map_f.close()
                
                
                
                
            #Magnebots
            for o in [*self.user_magnebots,*self.ai_magnebots]:
                pos = o.dynamic.transform.position
                pos_new = [round((pos[0]+abs(min_pos[0]))/multiple), round((pos[2]+abs(min_pos[1]))/multiple)]
                #3 is for other magnebots
                
                #if o.disabled: #3 if active, 1 if not active
                #    self.object_type_coords_map[pos_new[0],pos_new[1]] = 1
                #else:
                self.object_type_coords_map[pos_new[0],pos_new[1]] = 3
                    
                if str(pos_new[0])+'_'+str(pos_new[1]) not in self.object_attributes_id:
                    self.object_attributes_id[str(pos_new[0])+'_'+str(pos_new[1])] = []
                self.object_attributes_id[str(pos_new[0])+'_'+str(pos_new[1])].append((1,self.robot_names_translate[str(o.robot_id)],o.disabled))

            
            if self.options.log_state and self.enable_logs: # and self.timer - past_timer > 1:
                past_timer = self.timer
                object_metadata = []
                
                """
                for key in self.object_attributes_id.keys():
                    xy = key.split('_')

                    type_object = int(self.object_type_coords_map[int(xy[0]),int(xy[1])])

                    object_metadata.append([int(xy[0]),int(xy[1]),type_object,self.object_attributes_id[key]])
                """
                for go in self.graspable_objects:
                    pos = self.object_manager.transforms[go].position
                    object_metadata.append([0,self.object_names_translate[go], round(float(pos[0]),2),round(float(pos[2]),2), int(self.required_strength[go]), int(self.danger_level[go])])
                    
                for magn in [*self.user_magnebots,*self.ai_magnebots]:
                    pos = magn.dynamic.transform.position
                    arms_held = [-1,-1]
                    if magn.dynamic.held[Arm.left].size > 0:
                        arms_held[0] = self.object_names_translate[magn.dynamic.held[Arm.left][0]]
                    if magn.dynamic.held[Arm.right].size > 0:
                        arms_held[1] = self.object_names_translate[magn.dynamic.held[Arm.right][0]]
                    
                    rot = QuaternionUtils.quaternion_to_euler_angles(magn.dynamic.transform.rotation)[1]
                        
                    object_metadata.append([1,self.robot_names_translate[str(magn.robot_id)],round(float(pos[0]),2),round(float(pos[2]),2),magn.disabled,arms_held[0],arms_held[1],magn.strength,round(float(rot),2)])
                
                
                self.sio.emit("log_output", (json.dumps({'metadata':object_metadata}),past_timer))
            
            
            #Set a visual target whenever the user wants to help
            if self.target:
                temp_all_ids = all_ids + self.graspable_objects
                position_ids = []
                agent_ids = []
                positions = []
                
                for t in self.target.keys():
                
                    position_id = temp_all_ids.index(int(self.target[t]))
                    if not position_id in self.user_magnebots[t].screen_positions["position_ids"]:
                        self.user_magnebots[t].screen_positions["position_ids"].append(position_id)
                        self.user_magnebots[t].screen_positions["positions"].append(TDWUtils.array_to_vector3(self.object_manager.transforms[int(self.target[t])].position))
                        self.user_magnebots[t].screen_positions["duration"].append(-1)
                    else:
                        position_index = self.user_magnebots[t].screen_positions["position_ids"].index(position_id)
                        self.user_magnebots[t].screen_positions["positions"][position_index] = TDWUtils.array_to_vector3(self.object_manager.transforms[int(self.target[t])].position)
                        
                
                
              
                    
                    
            commands_time = time.time()
            
            #Some extra commands to send and when to remove them
            to_eliminate = []
            for ex_idx in range(len(extra_commands)):
                duration[ex_idx] -= 1
                if not duration[ex_idx]:
                    to_eliminate.append(ex_idx)
                commands.append(extra_commands[ex_idx])
            
            to_eliminate.reverse()
            for e in to_eliminate:
                del duration[e]
                del extra_commands[e]
                
                
                

            #We update timer
            """
            if self.timer_limit:
                mins, remainder = divmod(self.timer_limit-self.timer, 60)
                secs,millisecs = divmod(remainder,1)
            else:
            """
            mins = 0
            secs = 0

            mins, remainder = divmod(self.timer, 60)
            secs,millisecs = divmod(remainder,1)

                    
                    
            object_info_update = []

            
            #print(QuaternionUtils.quaternion_to_euler_angles(all_magnebots[0].dynamic.transform.rotation)[0])
            
            
            
            #Update all stats related with closeness of magnebots, like strength factor
            #Iterate over all magnebots
            for idx in range(len(all_magnebots)):
                robot_id = all_magnebots[idx].robot_id
                all_magnebots[idx].strength = 1
                company = {}
                
                #print(all_magnebots[idx].screen_positions["position_ids"])
                
                pos1 = all_magnebots[idx].dynamic.transform.position[[0,2]] #Not interested in height coordinate
                
                if all_magnebots[idx].last_position.size > 0:
                    all_magnebots[idx].stats.distance_traveled += float(np.linalg.norm(pos1 - all_magnebots[idx].last_position))
                
                all_magnebots[idx].last_position = pos1
                
                if all_magnebots[idx].robot_id not in disabled_robots and all_magnebots[idx].disabled:
                
                    #start a fire
                    commands.append(self.get_add_visual_effect(name="fire", 
                                       position=TDWUtils.array_to_vector3(all_magnebots[idx].dynamic.transform.position),
                                       effect_id=self.get_unique_id()))
                
                    disabled_robots.append(all_magnebots[idx].robot_id)
                    robot_id_translated = self.robot_names_translate[str(all_magnebots[idx].robot_id)]
                  
                    for object_id_translated in all_magnebots[idx].item_info.keys():
                        if "sensor" in all_magnebots[idx].item_info[object_id_translated] and robot_id_translated in all_magnebots[idx].item_info[object_id_translated]["sensor"]:
                            #all_magnebots[idx].stats.objects_sensed += 1
                            if object_id_translated not in all_magnebots[idx].stats.objects_sensed:
                                all_magnebots[idx].stats.objects_sensed.append(object_id_translated)
                            
                            #len(list(all_magnebots[idx].item_info[object_id_translated]["sensor"].keys()))
                            
                    
                    
                    
                    #for k in all_magnebots[idx].stats.time_with_teammates.keys():
                    #    all_magnebots[idx].stats.time_with_teammates[k] = round(all_magnebots[idx].stats.time_with_teammates[k],1)
                    #all_magnebots[idx].stats.distance_traveled = round(all_magnebots[idx].stats.distance_traveled,1)
                    #all_magnebots[idx].stats.end_time = round(all_magnebots[idx].stats.end_time,1)
                    
                    if self.scenario == 2:
                        #Disable ai companion
                        if str(all_magnebots[idx].robot_id) in self.user_magnebots_ids:
                            agent_idx = self.user_magnebots_ids.index(str(all_magnebots[idx].robot_id))
                            self.ai_magnebots[agent_idx].disabled = True
                            self.ai_magnebots[agent_idx].stats.end_time = self.timer    
                        
                    
                    #Last magnebot to be sent stats. We also send to everyone the stats related to team performance
                    if len(disabled_robots) == len(all_magnebots):
                    
                        game_finished += 1
                    
                        failure_reasons = {self.robot_names_translate[str(am.robot_id)]:am.stats.failed for am in all_magnebots}
                        
                        num_sensed = {}
                        
                        for go in self.graspable_objects:
                            o = self.object_names_translate[go]
                            
                            for am in all_magnebots:
                                robot_id_translated2 = self.robot_names_translate[str(am.robot_id)]
                                if o in am.item_info and "sensor" in am.item_info[o] and robot_id_translated2 in am.item_info[o]["sensor"]:
                                    if o not in num_sensed.keys():
                                        num_sensed[o] = []
                                    num_sensed[o].append(robot_id_translated2)
                            
                                  
                        human_team_effort = 0   
                        reported_objects = {}            
                        for um in self.user_magnebots:
                        
                            robot_id_translated2 = self.robot_names_translate[str(um.robot_id)]
                            
                            for o in num_sensed.keys():
                                
                                if o in um.item_info and "sensor" in um.item_info[o] and robot_id_translated2 in um.item_info[o]["sensor"]:
                                
                                    um.stats.effort += 1/len(num_sensed[o])
                                    
                            um.stats.effort /= len(self.graspable_objects)
                            human_team_effort += um.stats.effort
                            
                            for ro in um.reported_objects:
                                
                                if ro not in reported_objects.keys():
                                    reported_objects[ro] = 0
                                    
                                reported_objects[ro] += 1
                            
                        for ro in reported_objects.keys(): #Count reported objects as collected
                            if reported_objects[ro] == len(all_magnebots):
                                for um in all_magnebots:
                                    object_id = list(self.object_names_translate.keys())[list(self.object_names_translate.values()).index(ro)]
                                    um.stats.objects_in_goal.append(ro)
                                    danger_level = self.danger_level[object_id]
                                    if danger_level == 2:
                                        um.stats.dangerous_objects_in_goal.append(ro)

                                
                        #Quality of work
                        
                        for um in self.user_magnebots:
                        
                            number_dangerous_objects_in_goal = 0
                            number_benign_objects_in_goal = 0
                            number_dropped_objects = 0
                            
                            for ob in um.stats.objects_in_goal:
                            
                                object_id = list(self.object_names_translate.keys())[list(self.object_names_translate.values()).index(ob)]
                            
                                weight = self.required_strength[object_id]
                                
                                if weight > len(all_magnebots): #Applicable for objects that are heavier than team size
                                    weight = len(all_magnebots)
                            
                                if ob in um.stats.dangerous_objects_in_goal:
                                    number_dangerous_objects_in_goal += 1/weight                            
                                else:
                                    number_benign_objects_in_goal += 1/weight
                                    
                                    
                            for ob in um.stats.dropped_outside_goal:
                                object_id = list(self.object_names_translate.keys())[list(self.object_names_translate.values()).index(ob)]
                                weight = self.required_strength[object_id]
                                number_dropped_objects += 1/weight
                                
                            
                            #number_dangerous_objects_in_goal = len(um.stats.dangerous_objects_in_goal)
                            
                            if len(self.dangerous_objects):
                                um.stats.quality_work = max(0,(number_dangerous_objects_in_goal - number_benign_objects_in_goal - number_dropped_objects)/len(self.dangerous_objects))
                            else:
                                um.stats.quality_work = 1
                        
                        
                        team_quality_work = sum([am.stats.quality_work for am in all_magnebots])        
                        
                        max_human_team_payment = self.payment*len(self.user_magnebots)
                        actual_human_team_payment = max_human_team_payment*(team_quality_work+human_team_effort)/2
                        
                        all_stats = []
                        
                        end_time = all_magnebots[idx].stats.end_time
                            
                        if end_time > self.timer_limit:
                            end_time = self.timer_limit
                        
                        if self.timer_limit:
                            team_speed_work = self.timer_limit/(max(self.timer_limit/10,min(self.timer_limit,end_time)))
                        else:
                            team_speed_work = 0
                            
                        team_achievement = team_speed_work * team_quality_work
                        
                        for idx2 in range(len(all_magnebots)):
                        
                        
                            if team_quality_work+human_team_effort > 0:
                                individual_contribution = (all_magnebots[idx2].stats.quality_work + all_magnebots[idx2].stats.effort)/(team_quality_work+human_team_effort) #human quality work
                            else:
                                individual_contribution = 0
                                
                            individual_payment = actual_human_team_payment*individual_contribution
                        
                        
                            all_magnebots[idx2].stats.team_dangerous_objects_in_goal = goal_counter
                            all_magnebots[idx2].stats.total_dangerous_objects = len(self.dangerous_objects)
                            all_magnebots[idx2].stats.team_end_time = end_time
                            all_magnebots[idx2].stats.team_failure_reasons = failure_reasons
                            all_magnebots[idx2].stats.team_quality_work = team_quality_work
                            all_magnebots[idx2].stats.team_speed_work = team_speed_work
                            all_magnebots[idx2].stats.team_achievement = team_achievement
                            all_magnebots[idx2].stats.human_team_effort = human_team_effort
                            all_magnebots[idx2].stats.team_payment = actual_human_team_payment
                            all_magnebots[idx2].stats.individual_payment = individual_payment
                            
                            
                            self.sio.emit("stats", (self.robot_names_translate[str(all_magnebots[idx2].robot_id)], all_magnebots[idx2].stats.__dict__, self.timer, True))
                            
                            all_stats.append(all_magnebots[idx2].stats.__dict__)
                            
                            
                        if self.options.log_results:
                            if not os.path.exists(self.options.log_results):
                                os.makedirs(self.options.log_results)
                            
                        
                            log_results_f = open(self.options.log_results + "/" + dateTime + "_" + str(game_finished) + '_results.txt', "w")
                            json.dump({"results": all_stats, "seed": self.seed_value}, log_results_f)
                            log_results_f.close()
                            
                        #Reset automatically for tutorial
                        if self.scenario == 2:
                            self.previous_scenario = self.scenario
                            self.scenario = 1
                            self.reset = True
                        elif self.options.no_human_test:
                            self.previous_scenario = self.scenario
                            #self.scenario = 1
                            self.reset = True
                            
                            if len(self.user_magnebots) > 0:
                                self.timer_limit = 0
                                self.waiting = True #When is this necessary
                                self.enable_logs = False
                        else:
                            self.enable_logs = False
                            self.timer_limit = 0   
                            self.previous_scenario = self.scenario
                            self.scenario = 2
                            self.reset = True
                        
                            
                    else:
                        self.sio.emit("stats", (self.robot_names_translate[str(all_magnebots[idx].robot_id)], all_magnebots[idx].stats.__dict__, self.timer, False))
                
                for idx2 in range(len(all_magnebots)):
                    if idx == idx2:
                        continue
                        
                    
                    pos2 = all_magnebots[idx2].dynamic.transform.position[[0,2]]
                    distance = np.linalg.norm(pos1 - pos2)
                    
                    if not all_magnebots[idx2].disabled and distance < int(self.cfg['strength_distance_limit']) and not any(doIntersect([pos2[0],pos2[1]],[pos1[0],pos1[1]],[self.walls[w_idx][0][0],self.walls[w_idx][0][1]],[self.walls[w_idx][-1][0],self.walls[w_idx][-1][1]]) for w_idx in range(len(self.walls))): #Check if robot is close enough to influence strength
                        all_magnebots[idx].strength += 1 #Increase strength
                        
                        robot2 = self.robot_names_translate[str(all_magnebots[idx2].robot_id)]
                        if robot2 not in all_magnebots[idx].stats.time_with_teammates:
                            all_magnebots[idx].stats.time_with_teammates[robot2] = 0
                        all_magnebots[idx].stats.time_with_teammates[robot2] += float(sim_elapsed_time)
                        
                        
                        for arm in [Arm.left,Arm.right]:
                            if all_magnebots[idx].dynamic.held[arm].size > 0:
                                object_id = self.object_names_translate[all_magnebots[idx].dynamic.held[arm][0]]
                                if object_id not in all_magnebots[idx].current_teammates:
                                    all_magnebots[idx].current_teammates[object_id] = {}
                                    
                                if idx2 not in all_magnebots[idx].current_teammates[object_id]:
                                    all_magnebots[idx].current_teammates[object_id][idx2] = 0
                                all_magnebots[idx].current_teammates[object_id][idx2] += float(sim_elapsed_time)
                        
                        
                    if distance < int(self.cfg['communication_distance_limit']): #Check if robot is close enough to communicate
                    
                        company[self.robot_names_translate[str(all_magnebots[idx2].robot_id)]] = (all_magnebots[idx2].controlled_by, pos2.tolist(), float(distance), all_magnebots[idx2].disabled) #Add information about neighbors
                        
                all_magnebots[idx].company = company 
                
                 
                        
                        
                '''
                        #Update object info entries when nearby
                        if not all_magnebots[idx].item_info == all_magnebots[idx2].item_info:
                            for it_element in all_magnebots[idx2].item_info.keys():
                                if it_element not in all_magnebots[idx].item_info:
                                    all_magnebots[idx].item_info[it_element] = all_magnebots[idx2].item_info[it_element]
                                else:
                                    if 'sensor' in all_magnebots[idx2].item_info[it_element]:
                                        if 'sensor' in all_magnebots[idx].item_info[it_element]:
                                            all_magnebots[idx].item_info[it_element]['sensor'].update(all_magnebots[idx2].item_info[it_element]['sensor'])
                                        else:
                                            all_magnebots[idx].item_info[it_element]['sensor'] = all_magnebots[idx2].item_info[it_element]['sensor']
                                            
                                #Newest information based on time
                                if all_magnebots[idx].item_info[it_element]['time'] > all_magnebots[idx2].item_info[it_element]['time']:
                                    all_magnebots[idx].item_info[it_element]['time'] = all_magnebots[idx2].item_info[it_element]['time']
                                    all_magnebots[idx].item_info[it_element]['location'] = all_magnebots[idx2].item_info[it_element]['location']
                                
                            all_magnebots[idx2].item_info = all_magnebots[idx].item_info 
                            object_info_update.extend([idx,idx2])
                '''
                        
                '''      
                #Transmit neighbors info
                if not all_magnebots[idx].company == company:
                    all_magnebots[idx].company = company          
                    if not self.local:      
                        self.sio.emit('neighbors_update', (idx,all_magnebots[idx].company)) 
                '''
                #Refresh danger level sensor             
                if all_magnebots[idx].refresh_sensor < global_refresh_sensor:
                    all_magnebots[idx].refresh_sensor += 1              
                            
                if all_magnebots[idx].ui_elements: #For user magnebots, update user interface
                    
                    all_magnebots[idx].ui.set_text(ui_id=all_magnebots[idx].ui_elements[1],text=f"My Strength: {all_magnebots[idx].strength}")
                    all_magnebots[idx].ui.set_size(ui_id=all_magnebots[idx].ui_elements[0], size={"x": int(self.progress_bar_size["x"] * self.progress_bar_scale["x"] * (all_magnebots[idx].strength)/len(all_magnebots)),    "y": int(self.progress_bar_size["y"] * self.progress_bar_scale["y"])})


                    
                    
                    all_magnebots[idx].ui.set_text(ui_id=all_magnebots[idx].ui_elements[2],text='{:02d}:{:02d}'.format(int(mins), int(secs)))
                    
                    #We modify action status
                    all_magnebots[idx].ui.set_text(ui_id=all_magnebots[idx].ui_elements[3],text="My Action Status: " + all_magnebots[idx].action.status.name)
                    
                    all_magnebots[idx].ui.set_text(ui_id=all_magnebots[idx].ui_elements[4],text="Targets in goal: " + str(goal_counter))
                    
                    all_magnebots[idx].ui.set_text(ui_id=all_magnebots[idx].ui_elements[5],text="(" + str(round(all_magnebots[idx].dynamic.transform.position[0],1)) + "," + str(round(all_magnebots[idx].dynamic.transform.position[2],1)) + ")")
                    
                    left_arm = ""
                    right_arm = ""
                    
                    if all_magnebots[idx].dynamic.held[Arm.left].size > 0:
                        left_arm = str(self.object_names_translate[all_magnebots[idx].dynamic.held[Arm.left][0]])
                    if all_magnebots[idx].dynamic.held[Arm.right].size > 0:
                        right_arm = str(self.object_names_translate[all_magnebots[idx].dynamic.held[Arm.right][0]])
                    
                    all_magnebots[idx].ui.set_text(ui_id=all_magnebots[idx].ui_elements[6],text="L:" + left_arm + " R:" + right_arm)
                    
                    """
                    if np.linalg.norm(all_magnebots[idx].dynamic.transform.position[[0,2]]) < float(self.cfg["goal_radius"]):
                        in_goal_txt = "In goal area"
                    else:
                        in_goal_txt = "Out of goal area"
                    all_magnebots[idx].ui.set_text(ui_id=all_magnebots[idx].ui_elements[7],text=in_goal_txt)
                    """
                    
                    #Add screen position markers requested by each particular user magnebot
                    
                    to_delete = []
                    for sc_idx in range(len(all_magnebots[idx].screen_positions['positions'])):
                        
                        if all_magnebots[idx].screen_positions['duration'][sc_idx] > 0:
                            all_magnebots[idx].screen_positions['duration'][sc_idx] -= 1
                        
                            if all_magnebots[idx].screen_positions['duration'][sc_idx] == 0:
                                to_delete.append(sc_idx)
                            
                            elif all_magnebots[idx].screen_positions['position_ids'][sc_idx] not in screen_positions['position_ids']:

                                screen_positions["position_ids"].append(all_magnebots[idx].screen_positions['position_ids'][sc_idx])
                                screen_positions["positions"].append(all_magnebots[idx].screen_positions['positions'][sc_idx])
                            
                                
                                
                                
                    
                        
                    to_delete.reverse()
                    for e in to_delete:
                        
                        del all_magnebots[idx].screen_positions['position_ids'][e]                           
                        del all_magnebots[idx].screen_positions['positions'][e]    
                        del all_magnebots[idx].screen_positions['duration'][e]    

                        
                        

                for arm in [Arm.left,Arm.right]:
                
                    if all_magnebots[idx].action.status == ActionStatus.cannot_reach or all_magnebots[idx].action.status == ActionStatus.failed_to_grasp or (all_magnebots[idx].controlled_by == "human" and all_magnebots[idx].resetting_arm and time.time() - all_magnebots[idx].grasping_time > 5):
                        all_magnebots[idx].grasping = False
                        all_magnebots[idx].resetting_arm = False
                        all_magnebots[idx].reset_arm(arm)
                        
                        if all_magnebots[idx].ui_elements:
                            txt = all_magnebots[idx].ui.add_text(text="Cannot grasp this way!",
                                             position={"x": 0, "y": 0},
                                             color={"r": 0, "g": 0, "b": 0, "a": 1},
                                             font_size=20
                                             )
                            messages.append([idx,txt,0])
                
                    if all_magnebots[idx].dynamic.held[arm].size > 0:
                        
                        
                        if all_magnebots[idx].resetting_arm and all_magnebots[idx].action.status != ActionStatus.ongoing:
                            all_magnebots[idx].resetting_arm = False
                            
                            if str(all_magnebots[idx].robot_id) in self.user_magnebots_ids:
                                all_magnebots[idx].reset_arm(arm)
              
                            print("Resetting arm")
                            
                            all_magnebots[idx].stats.grabbed_objects += 1
                            
                        #Drop object if strength decreases
                        if self.required_strength[all_magnebots[idx].dynamic.held[arm][0]] > all_magnebots[idx].strength:
                            
                         
                            self.object_dropping.append([int(all_magnebots[idx].dynamic.held[arm][0]),time.time(),all_magnebots[idx], arm])
                        
                            grasped_object = all_magnebots[idx].dynamic.held[arm][0]
                            all_magnebots[idx].drop(target=grasped_object, arm=arm)
                            all_magnebots[idx].grasping = False
                            

                            
                            
                            if any(np.linalg.norm(self.object_manager.transforms[grasped_object].position[[0,2]] - np.array(goal[1])) >= goal[0] for goal in self.goal_area):
                            
                                robot_ids,sort_indices = self.get_involved_teammates(all_magnebots[idx].current_teammates, self.object_names_translate[grasped_object])
                            
                                for sidx in range(int(self.required_strength[grasped_object])-1):
                                
                                    if sidx < len(sort_indices):
                                        all_magnebots[robot_ids[sort_indices[sidx]]].stats.dropped_outside_goal.append(self.object_names_translate[grasped_object])
                                        robot_id = self.robot_names_translate[str(all_magnebots[robot_ids[sort_indices[sidx]]].robot_id)]
                                        if (robot_id,self.object_names_translate[grasped_object]) not in dropped_objects:
                                            dropped_objects.append((robot_id,self.object_names_translate[grasped_object]))
                                        
                                        if all_magnebots[robot_ids[sort_indices[sidx]]].ui_elements: 
                                            messages.append(self.info_message_ui(all_magnebots, robot_ids[sort_indices[sidx]], "Penalty! Object accidentaly dropped!", "red"))
                            
                                all_magnebots[idx].stats.dropped_outside_goal.append(self.object_names_translate[grasped_object])
                                robot_id = self.robot_names_translate[str(all_magnebots[idx].robot_id)]
                                
                                if (robot_id,self.object_names_translate[grasped_object]) not in dropped_objects:
                                    dropped_objects.append((robot_id,self.object_names_translate[grasped_object]))
                                print("Dropped object!", dropped_objects)
                                if all_magnebots[idx].ui_elements:
                                    messages.append(self.info_message_ui(all_magnebots, idx, "Penalty! Object accidentaly dropped!", "red"))
                            
                            """
                            if grasped_object in self.dangerous_objects:
                                
                                if all_magnebots[idx].ui_elements:
                                    txt = all_magnebots[idx].ui.add_text(text="Failure! Dangerous object dropped!",
                                     position={"x": 0, "y": 0},
                                     color={"r": 1, "g": 0, "b": 0, "a": 1},
                                     font_size=20
                                     )
                                    messages.append([idx,txt,0])
                                
                                #self.reset_message = True
                                
                                #self.sio.emit("disable", (self.robot_names_translate[str(all_magnebots[idx].robot_id)]))
                                all_magnebots[idx].disabled = True
                                all_magnebots[idx].stats.end_time = self.timer
                                
                                all_magnebots[idx].stats.failed = 2
                            """
                        #Terminate game if dangerous object held alone
                        '''
                        if all_magnebots[idx].dynamic.held[arm][0] in self.dangerous_objects:
                            
                            if (all_magnebots[idx].controlled_by == 'ai' and 'human' not in all_magnebots[idx].company.values()) or (all_magnebots[idx].controlled_by == 'human' and 'ai' not in all_magnebots[idx].company.values()):
                                for um in self.user_magnebots:
                                    txt = um.ui.add_text(text="Dangerous object picked without help!",
                                     position={"x": 0, "y": 0},
                                     color={"r": 0, "g": 0, "b": 1, "a": 1},
                                     font_size=20
                                     )
                                    messages.append([idx,txt,0])
                                self.terminate = True
                        '''

                #Transmit ai controlled robots status
                if not self.local and all_magnebots[idx] in self.ai_magnebots:
                    if all_magnebots[idx].action.status != all_magnebots[idx].past_status:
                        all_magnebots[idx].past_status = all_magnebots[idx].action.status
                        #self.sio.emit("ai_status", (idx,all_magnebots[idx].action.status.value))


            '''
                            
            #Share object info
            object_info_update = list(set(object_info_update))

            if not self.local:
                for ob_idx in object_info_update:
                    self.sio.emit('objects_update', (ob_idx,all_magnebots[ob_idx].item_info)) 
            '''
             
            #If all robots have been disabled
            if len(disabled_robots) == len(all_magnebots):
                pass #Do something when game finishes
            
             
            #Ask for the given screen positions of certain objects/magnebots     
            screen_positions["position_ids"].extend(list(range(0,len(all_ids))))
            screen_positions["positions"].extend([*user_magnebots_positions,*ai_magnebots_positions])
            
            
            

            
            #For top down view
            if self.options.showall:
                new_screen_positions = {}
                new_screen_positions["position_ids"] = [*screen_positions["position_ids"],*list(range(len(all_ids),len(all_ids)+len(self.graspable_objects)))]
                new_screen_positions["positions"] = [*screen_positions["positions"],*[TDWUtils.array_to_vector3(self.object_manager.transforms[o].position) for o in self.graspable_objects]]
                
                commands.append({"$type": "send_screen_positions", "position_ids": new_screen_positions["position_ids"], "positions":new_screen_positions["positions"], "ids": ["a",*self.user_magnebots_ids], "frequency": "once"})
                
            elif self.user_magnebots_ids:
            
                #print(screen_positions["position_ids"], screen_positions["positions"], self.user_magnebots_ids)
                commands.append({"$type": "send_screen_positions", "position_ids": screen_positions["position_ids"], "positions":screen_positions["positions"], "ids": [*self.user_magnebots_ids], "frequency": "once"})
                
                
            #print(commands)
            
            
            
            if self.scenario == 2:
                self.tutorial_state_machine(all_magnebots, commands)
            
            
            if num_users >= 5:
                commands.append({"$type": "step_physics", "frames": 1})
            
            try:
                #commands.append({"$type": "step_physics", "frames": 1})
                resp = self.communicate(commands)
            except Exception as e:
                print("Error communication")
                #pdb.set_trace()
                self.sio.emit("sim_crash", self.timer)
                if hasattr(e, 'message'):
                    print(e.message)
                else:
                    print(e)
                    #if not self.local:
                    #    self.reset_agents()
                    return -1
            
            
            
            self.initializing = False

            duration_fps = time.time()-past_time
            estimated_fps = (estimated_fps + 1/duration_fps)/2
            #print(estimated_fps)
            past_time = time.time()


            commands.clear()

            
            
            
            

            screen_data = {}
            magnebot_images = {}
 
            key_pressed = []
            key_hold = []

            ############ Output data processing
            for i in range(len(resp) - 1):
                
                r_id = OutputData.get_data_type_id(resp[i])

                # Get Images output data.
                if r_id == "imag":
                    images = Images(resp[i])
                    # Determine which avatar captured the image. In this case, the third person camera
                    if images.get_avatar_id() == "a":
                        # Iterate throught each capture pass.
                        for j in range(images.get_num_passes()):
                            # This is the _img pass.

                            if images.get_pass_mask(j) == "_img":
                                #image_arr = images.get_image(j)

                                # Get a PIL image.
                                pil_image = ImageOps.flip(TDWUtils.get_pil_image(images=images, index=j))
                                #img_image = np.asarray(pil_image)
                                magnebot_images[images.get_avatar_id()] = np.asarray(pil_image)
                                '''
                                if cams:
                                    cams[0].send(img_image)
                                    if self.options.create_video:
                                        video[0].write(img_image)
                                '''
                                #cv2.imshow('frame',np.asarray(pil_image))
                                #cv2.waitKey(1)
                                
                            
                    
                    #Process images from user magnebot cameras
                    elif images.get_avatar_id() in self.user_magnebots_ids:
                        idx = self.user_magnebots_ids.index(images.get_avatar_id())
                        pil_images = self.user_magnebots[idx].dynamic.get_pil_images()
                        
                        if 'id' in pil_images: #This is for focusing on an object for human users
                        
                            commands.append({"$type": "set_pass_masks",
                              "avatar_id": str(self.user_magnebots[idx].robot_id),
                              "pass_masks": ["_img"]})
                            p_size = pil_images['id'].size
                            pointer_position = (round(pil_images['id'].size[0]/2),round(pil_images['id'].size[1]/2))
                            color_center = pil_images['id'].getpixel(pointer_position)
                            
                            if color_center in self.segmentation_colors:
                            
                                o_id = self.segmentation_colors[color_center]
                                
                                if o_id in self.graspable_objects:
                                    pos_idx = len(all_ids)+self.graspable_objects.index(o_id)
 
                                
                                    self.user_magnebots[idx].screen_positions["position_ids"].append(pos_idx)
                                    self.user_magnebots[idx].screen_positions["positions"].append(TDWUtils.array_to_vector3(self.object_manager.transforms[o_id].position))
                                    self.user_magnebots[idx].screen_positions["duration"].append(100)
                                    
                                    self.user_magnebots[idx].focus_object = o_id
                                    

                        
                                    o_translated = self.object_names_translate[o_id]
                                    if o_translated not in self.user_magnebots[idx].item_info:
                                        self.user_magnebots[idx].item_info[o_translated] = {}
                                        
                                    self.user_magnebots[idx].item_info[o_translated]['weight'] = int(self.required_strength[o_id])
                                    self.user_magnebots[idx].item_info[o_translated]['time'] = self.timer
                                    self.user_magnebots[idx].item_info[o_translated]['location'] = self.object_manager.transforms[o_id].position.tolist()
                                    
                                    if 'sensor' not in self.user_magnebots[idx].item_info[o_translated]:
                                        self.user_magnebots[idx].item_info[o_translated]['sensor'] = {}

                                    self.raycast_request.append(str(self.user_magnebots[idx].robot_id))
                        
                        img_image = np.asarray(ImageOps.flip(pil_images['img']))

                        magnebot_images[images.get_avatar_id()] = img_image
                    #Process images from ai magnebot cameras
                    elif images.get_avatar_id() in self.ai_magnebots_ids:
                        idx = self.ai_magnebots_ids.index(images.get_avatar_id())
                        img_image = np.asarray(ImageOps.flip(self.ai_magnebots[idx].dynamic.get_pil_images()['img']))
                        magnebot_images[images.get_avatar_id()] = img_image
                        
                    
                        
    
                elif r_id == "scre": #Get screen coordinates from objects
                    self.screen_output(resp[i], screen_data, all_magnebots, all_ids)

                        
                    
                elif r_id == "rayc":   #Raycast information (given ray vector, which objects are in its path) Needs adjustment. Activated when focusing on an object
      
                    self.raycast_output(resp[i], all_ids)
                    
                    

                elif r_id == "keyb":#For each keyboard key pressed
                    keys = KBoard(resp[i])
                    key_pressed_tmp = [keys.get_pressed(j) for j in range(keys.get_num_pressed())]
                    key_hold_tmp = [keys.get_held(j) for j in range(keys.get_num_held())]
                    
                    key_pressed.extend(key_pressed_tmp)
                    key_hold.extend(key_hold_tmp)
                    
                    
                elif r_id == "fram":
                    framerate = Framerate(resp[i])
                    sim_elapsed_time = framerate.get_frame_dt()
                    #print(1/sim_elapsed_time)
                    
                elif r_id == "quit":
                    print("Error quitting")
                    self.sio.emit("sim_crash", self.timer)
                    return -1
                             
            ##################
                           
            if self.object_dropping:
                to_remove = []
                for o_idx,od in enumerate(self.object_dropping): #Increase mass of objects when dropped
                    if not od[0] in od[2].dynamic.held[od[3]]:
                        if time.time() - od[1] > 1:
                        
                            try:
                                print("grasping object 1")
                                commands.append({"$type": "set_mass", "mass": 1000, "id": od[0]})
                                to_remove.append(o_idx)
                            except:
                                print("grasped object2", od[0])
                    else:
                        print("Can't drop object")
                        self.object_dropping[o_idx][1] = time.time()
                    
                if to_remove:
                    to_remove.reverse()
                    for tr in to_remove:
                        del self.object_dropping[tr]    

            
                
            
                                
            #Process keyboard output
            key_pressed.extend(self.extra_keys_pressed)
            self.extra_keys_pressed = []
            #if key_pressed:
            #    print(key_pressed)
            if num_users > 0:    
                self.keyboard_output(key_pressed, key_hold, extra_commands, duration, keys_time_unheld, all_ids, messages, estimated_fps)
            
            
            #Destroy messages in the user interface after some time
            to_eliminate = []
            for m_idx in range(len(messages)):
                messages[m_idx][2] += 1
                if messages[m_idx][2] == 100:
                    self.uis[messages[m_idx][0]].destroy(messages[m_idx][1])
                    to_eliminate.append(m_idx)
                    if self.terminate:
                        done = True
                    if self.reset_message:
                        self.reset_message = False
                        self.reset = True
                        self.previous_scenario = self.scenario
                        #self.scenario = 1
                        
                        
                
            to_eliminate.reverse()
            for te in to_eliminate:
                try:
                    del messages[te]
                except:
                    pdb.set_trace()


            #Draw ui objects
            for key in magnebot_images.keys():
                if key in screen_data:
                    if key == "a":
                        self.add_ui(magnebot_images[key], screen_data[key], True)
                    else:
                        self.add_ui(magnebot_images[key], screen_data[key], False)


            #Game ends when all dangerous objects are left in the rug
            goal_counter = 0
            
            
            for sd in self.dangerous_objects:
                if any(np.linalg.norm(self.object_manager.transforms[sd].position[[0,2]] - np.array(goal[1])) < goal[0] for goal in self.goal_area):
                    goal_counter += 1
                     
            
            
            """
            if goal_counter == len(self.dangerous_objects):
                for idx,um in enumerate(self.user_magnebots):
                    txt = um.ui.add_text(text="Success!",
                                         position={"x": 0, "y": 0},
                                         color={"r": 0, "g": 1, "b": 0, "a": 1},
                                         font_size=20
                                         )
                    messages.append([idx,txt,0])
                self.reset_message = True
            """
                        
            
            #Show view of magnebot
            if self.local and self.user_magnebots and str(self.user_magnebots[0].robot_id) in magnebot_images:
                cv2.imshow('frame', magnebot_images[str(self.user_magnebots[0].robot_id)])
                cv2.waitKey(1)
            
            
            to_remove = []
            #Execute delayed actions
            for qa_idx in range(len(self.queue_perception_action)):
                if not self.queue_perception_action[qa_idx][2]:
                    self.queue_perception_action[qa_idx][2] -= 1
                else:
                    self.queue_perception_action[qa_idx][0](*self.queue_perception_action[qa_idx][1])
                    #print(self.queue_perception_action[qa_idx])
                    to_remove.append(qa_idx)
                    
            to_remove.reverse()
            for tr in to_remove:
                del self.queue_perception_action[tr]
            
            
            
            #Send frames to virtual cameras in system and occupancy maps if required, and all outputs needed

            if self.options.create_video and self.enable_logs:
                video_meta_f.write(str(self.timer)+'\n') #Given variable frame rate, save timestamp of frame

            #For top down view
            if cams and not self.no_debug_camera:
                img_image = magnebot_images["a"]
                cams[0].send(img_image)
                """
                im = Image.fromarray(img_image)
                im.save("top_view.png")
                """
                if self.options.create_video and self.enable_logs:
                    img_image_rgb = cv2.cvtColor(img_image, cv2.COLOR_BGR2RGB)
                    video[0].write(img_image_rgb)


            #Send human output data
            idx = 0
            for magnebot_id in self.user_magnebots_ids:
            
                if cams and magnebot_id in magnebot_images:
                    cams[idx+1].send(magnebot_images[magnebot_id])
                    
                    """
                    if not idx:
                        im = Image.fromarray(magnebot_images[magnebot_id])
                        im.save("first_view.png")
                    """
                    
                    if self.options.create_video and self.enable_logs:
                        #pdb.set_trace()
                        img_image_rgb = cv2.cvtColor(magnebot_images[magnebot_id], cv2.COLOR_BGR2RGB)
                        video[idx+1].write(img_image_rgb)
                    
                item_info = {}
                all_idx = all_ids.index(str(magnebot_id))
                
                if magnebot_id in self.danger_sensor_request:
                    _,item_info = self.danger_sensor_reading(magnebot_id)
                    self.danger_sensor_request.remove(magnebot_id)
                    
                if magnebot_id in self.raycast_request:

                    item_info = all_magnebots[all_idx].item_info
                    self.raycast_request.remove(magnebot_id)
                    
                #Track objects being carried
                if not item_info:
                    self.track_objects_carried(all_magnebots, all_idx, item_info, messages)
                            
                
                objects_held = self.get_objects_held_state(all_magnebots, all_ids, magnebot_id)
                
                if not self.local and not all_magnebots[idx].last_output:
                
                    self.sio.emit('human_output', (all_idx, all_magnebots[idx].dynamic.transform.position.tolist(), item_info, all_magnebots[all_idx].company, self.timer, all_magnebots[idx].disabled,dropped_objects, objects_held))
                    if all_magnebots[idx].disabled:
                        all_magnebots[idx].last_output = True
                    
                    
                idx += 1

            
            #Send AI output data
            
            for m_idx, magnebot_id in enumerate(self.ai_magnebots_ids):
                all_idx = all_ids.index(str(magnebot_id)) 
                if not all_magnebots[all_idx].skip_frames or not (self.frame_num % all_magnebots[all_idx].skip_frames):
                    if cams and magnebot_id in magnebot_images:
                        cams[idx+1].send(magnebot_images[magnebot_id])
                        
                        if self.options.create_video and self.enable_logs:
                            video[idx+1].write(magnebot_images[magnebot_id])

                    #Occupancy maps

                    extra_status = [0]*3
                    
                    if magnebot_id in self.occupancy_map_request:
                         extra_status[0] = 1
                   
                    if not all_magnebots[all_idx].disabled:
                        limited_map, reduced_metadata = self.get_occupancy_map(magnebot_id)
                    else:
                        limited_map = []
                        reduce_metadata = {}
                    
                    
                    if magnebot_id in self.objects_held_status_request:
                        extra_status[2] = 1
                    
                    objects_held = self.get_objects_held_state(all_magnebots, all_ids, magnebot_id)
                    
                    
                    
                    if magnebot_id in self.danger_sensor_request:
                        _,item_info = self.danger_sensor_reading(magnebot_id)
                        self.danger_sensor_request.remove(magnebot_id)
                        extra_status[1] = 1
                    else:
                        item_info = {}
                        
                        
                       
                        
                    #Track objects being carried
                    if not item_info:
                        self.track_objects_carried(all_magnebots, all_idx, item_info, messages)
                        
                    
                    
                    
                    #if all_idx in self.ai_status_request:
                    ai_status = all_magnebots[all_idx].past_status.value
                        
                        
                    #print(limited_map)
                    if not self.local and not all_magnebots[all_idx].last_output:
                        #if any(extra_status):
                        #    print("Sending extra status")
                        self.sio.emit('ai_output', (all_idx, json_numpy.dumps(limited_map), reduced_metadata, objects_held, item_info, ai_status, extra_status, all_magnebots[all_idx].strength, self.timer, all_magnebots[all_idx].disabled, [all_magnebots[all_idx].dynamic.transform.position.tolist(), QuaternionUtils.quaternion_to_euler_angles(all_magnebots[all_idx].dynamic.transform.rotation).tolist()],dropped_objects))
                        if all_magnebots[all_idx].disabled:
                            all_magnebots[all_idx].last_output = True
    
                        if dropped_objects:
                            dropped_objects_message_counter += 1
                            if dropped_objects_message_counter >= len(self.ai_magnebots_ids):
                                dropped_objects = []
                                dropped_objects_message_counter = 0
                idx += 1
            
             
            
            #If timer expires end simulation, else keep going
            if self.timer_limit and self.timer_limit-self.timer <= 0:
                for idx,um in enumerate(all_magnebots):
                
                    """
                    txt = um.ui.add_text(text="Session Ended",
                                         position={"x": 0, "y": 0},
                                         color={"r": 1, "g": 0, "b": 0, "a": 1},
                                         font_size=20
                                         )
                    messages.append([idx,txt,0])
                    """
                
                    #self.sio.emit("disable", (self.robot_names_translate[str(all_magnebots[idx].robot_id)]))
                    all_magnebots[idx].disabled = True
                    all_magnebots[idx].stats.end_time = self.timer
                    

            else:
                new_time = time.time()
                self.timer += new_time-self.real_timer
                self.real_timer = new_time



            

            #Reset world
            if self.reset:
            
            
                print("Resetting...")
                
                for am in all_magnebots:
                    self.sio.emit("reset_announcement", (self.robot_names_translate[str(am.robot_id)]))
            
                self.reset_world()
                
                self.reset_number += 1
                
                if not self.local:
                    self.reset_agents()
                
                all_ids = [*self.user_magnebots_ids,*self.ai_magnebots_ids]
                all_magnebots = [*self.user_magnebots,*self.ai_magnebots]
                #Include the positions of other magnebots in the view of all user magnebots
                for um in self.user_magnebots:
                    um.screen_positions["position_ids"].extend(list(range(0,len(all_ids))))
                    um.screen_positions["positions"].extend([-1]*len(all_ids))
                    um.screen_positions["duration"].extend([-1]*len(all_ids))
                    #um.last_output = False
                    #um.disabled = False
                    
                self.reset = False
                #pdb.set_trace()
                print("Reset complete")
                sim_elapsed_time = 0
                disabled_robots = []
                dropped_objects = []
   
            elif self.reset_partial:
                self.partial_reset()
                all_ids = [*self.user_magnebots_ids,*self.ai_magnebots_ids]
                for um in self.user_magnebots:
                    um.screen_positions["position_ids"].extend(list(range(0,len(all_ids))))
                    um.screen_positions["positions"].extend([-1]*len(all_ids))
                    um.screen_positions["duration"].extend([-1]*len(all_ids))
                self.reset_partial = False
   
            self.frame_num +=1
            
        self.communicate({"$type": "terminate"})
        
        return 0


    def reset_agents(self):
    
        if self.waiting and self.options.no_human_test:
            all_magnebots = [*self.user_magnebots]
        else:
            all_magnebots = [*self.user_magnebots,*self.ai_magnebots]
            
        extra_config = self.extra_config_population()
        
        for am in all_magnebots:
            self.sio.emit("agent_reset", (self.robot_names_translate[str(am.robot_id)],self.timer,self.real_timer, extra_config))

    def reset_world(self):
        
        if self.options.seed > -1:
            self.seed_value = self.options.seed
        else:
            self.seed_value = random.randrange(sys.maxsize)
        random.seed(self.seed_value)
        print("SEED:", self.seed_value)
        
        commands = []
        
        for go in self.graspable_objects:
            commands.append({"$type": "destroy_object", "id": go})
            
        for env_obj in self.env_objects:
            commands.append({"$type": "destroy_object", "id": env_obj})
            
        if not self.no_debug_camera:
            commands.append({"$type": "destroy_avatar", "avatar_id": 'a'})   
            
        self.communicate(commands)
            
        commands = []
        
        
            
        if self.previous_scenario == 2 and not self.scenario == 2:
            for u_idx in range(len(self.ai_magnebots)):   
                commands.append({"$type": "destroy_avatar", "id": str(self.ai_magnebots[u_idx].robot_id)})
                self.sio.emit("agent_delete", (self.robot_names_translate[str(self.ai_magnebots[u_idx].robot_id)]))
                self.add_ons.remove(self.ai_magnebots[u_idx])
                del self.robot_names_translate[str(self.ai_magnebots[u_idx].robot_id)]
            '''    
            for u_idx in range(len(self.user_magnebots)):   
                commands.append({"$type": "destroy_avatar", "id": str(self.user_magnebots[u_idx].robot_id)})
            '''    
            self.ai_magnebots = []
            self.ai_magnebots_ids = []
            
        if self.waiting and self.options.no_human_test:
            for u_idx in range(len(self.ai_magnebots)):
                self.sio.emit("agent_delete", (self.robot_names_translate[str(self.ai_magnebots[u_idx].robot_id)]))
            
        '''
        #Reset user magnebots
        for u_idx in range(len(self.user_magnebots)):
                print("Destroying: ", u_idx)
                self.user_magnebots[u_idx].ui.destroy_all(destroy_canvas=True)
                self.communicate([])
        for u_idx in range(len(self.user_magnebots)):    
                print("Destroying: ", self.user_magnebots[u_idx].robot_id)
                self.communicate({"$type": "destroy_avatar", "id": str(self.user_magnebots[u_idx].robot_id)})
                self.communicate({"$type": "set_render_order", "render_order": 100, "sensor_name": "SensorContainer", "avatar_id": "a"})
        print("Fin")
        '''
                
        #Reset ai magnebots
        #for u_idx in range(len(self.ai_magnebots)):
        #        self.communicate({"$type": "destroy_avatar", "id": str(self.ai_magnebots[u_idx].robot_id)})
        # self.user_magnebots[u_idx].reset(position=user_spawn_positions[u_idx])
        #self.user_magnebots[0].reset(position=user_spawn_positions[0])
            
        #self.user_magnebots[1].ui.destroy_all(destroy_canvas=True)
        #self.communicate([])

        #self.scenario = 1

        self.object_manager.initialized = False
        

        
        commands = []
        commands.extend(self.create_scene())

        self.ai_spawn_positions = self.ai_original_spawn_positions.copy()
        self.user_spawn_positions = self.user_original_spawn_positions.copy()
        
        global agents_good_sensors
        agents_good_sensors = [1,2,3,4]
        random.shuffle(agents_good_sensors)
        
        if self.scenario != 2:
            random.shuffle(self.ai_spawn_positions)
            random.shuffle(self.user_spawn_positions)
            ai_spawn_positions = self.ai_spawn_positions
            user_spawn_positions = self.user_spawn_positions
        else:
            cell_size = self.cfg['cell_size']
            user_spawn_positions = []
            ai_spawn_positions = []
            extra_ai_agents = num_users
            for um in range(num_users):
                #c1 = [self.wall_length[0]*um + cell_size*1.5 - self.scenario_size[0]/2, cell_size*1.5 - self.scenario_size[1]/2]
                user_spawn_positions.append({"x": self.wall_length[0]*um + cell_size*3.5 - self.scenario_size[0]/2, "y": 0, "z": -(self.wall_length[1] - cell_size*1.5 - self.scenario_size[1]/2)})
                ai_spawn_positions.append({"x": self.wall_length[0]*um + cell_size*4.5 - self.scenario_size[0]/2, "y": 0, "z": -(self.wall_length[1] - cell_size*5.5 - self.scenario_size[1]/2)})
        
        
            if self.previous_scenario != 2:
                #Create ai magnebots
                for ai_idx in range(extra_ai_agents):  
                    robot_id = self.get_unique_id()                                 
                    self.ai_magnebots.append(Enhanced_Magnebot(robot_id=robot_id, position=ai_spawn_positions[ai_idx],image_frequency=ImageFrequency.never, controlled_by='ai', difficulty_level=self.options.level))
                    #self.ai_magnebots.append(Enhanced_Magnebot(robot_id=robot_id, position=ai_spawn_positions[ai_idx],image_frequency=ImageFrequency.always, pass_masks=['_img'], controlled_by='ai'))
                    self.robot_names_translate[str(robot_id)] = chr(ord('A') + ai_idx + num_users)
                
                print(self.robot_names_translate)
                    
                print("Before:", self.add_ons)
                self.add_ons.extend([*self.ai_magnebots])
                print("After:", self.add_ons)
                self.ai_magnebots_ids = [str(um.robot_id) for um in self.ai_magnebots]
        
        
        for u_idx in range(len(self.ai_magnebots)):
            self.ai_magnebots[u_idx].reset(position=ai_spawn_positions[u_idx])
            print(self.robot_names_translate[str(self.ai_magnebots[u_idx].robot_id)])
        
        for u_idx in range(len(self.user_magnebots)):
            self.user_magnebots[u_idx].reset(position=user_spawn_positions[u_idx])
            #self.user_magnebots[u_idx].ui.initialized = False
            commands.append({"$type": "destroy_visual_effect", "id": self.user_magnebots[u_idx].robot_id})
            
            if self.scenario != 2:
                self.user_magnebots[u_idx].stats.token = ''.join(random.choices(string.ascii_lowercase + string.digits + string.ascii_uppercase, k=8))
            
            print(self.robot_names_translate[str(self.user_magnebots[u_idx].robot_id)])

        commands.extend(self.populate_world())
        
        
        self.communicate(commands)

        self.log_init_data()
        
        #Reattach canvas
        for um in self.user_magnebots:
            um.ui.attach_canvas_to_avatar(avatar_id=str(um.robot_id))
        self.communicate([])
        
        self.segmentation_colors = self.get_segmentation_colors()
        
        
        self.static_occupancy_map.generate(cell_size=self.cfg['cell_size']) #Get occupancy map only with walls
        
        self.communicate([])
        
        self.manual_occupancy_map()
        print(self.static_occupancy_map.occupancy_map)
        
    def partial_reset(self):

        
        commands = []
        
        object_list = []
        all_magnebots = [*self.user_magnebots,*self.ai_magnebots]
        
        print(self.object_names_translate)
        for go_idx in range(len(self.graspable_objects)):
            object_id = self.graspable_objects[go_idx]
            object_list.append([self.object_names_translate[object_id], self.required_strength[object_id], self.danger_level[object_id], self.object_manager.transforms[object_id].position])
            commands.append({"$type": "destroy_object", "id": object_id})
            
        for env_obj in self.env_objects:
            commands.append({"$type": "destroy_object", "id": env_obj})
            
        if not self.no_debug_camera:
            commands.append({"$type": "destroy_avatar", "avatar_id": 'a'})   
            
        agent_locations = []
        for idx in range(len(all_magnebots)):
            agent_locations.append(all_magnebots[idx].dynamic.transform.position)
            
        self.communicate(commands)
            
        commands = []
        
        self.object_manager.initialized = False
        

        
        commands = []
        commands.extend(self.create_scene())
        

        for idx in range(len(all_magnebots)):
            all_magnebots[idx].partial_reset(position={"x":float(agent_locations[idx][0]),"y":0,"z":float(agent_locations[idx][2])})

        commands.extend(self.populate_world(object_list))
        #pdb.set_trace()
        
        self.communicate(commands)
        
        #self.log_init_data()
        
        #Reattach canvas
        for um in self.user_magnebots:
            um.ui.attach_canvas_to_avatar(avatar_id=str(um.robot_id))
        self.communicate([])
        
        self.segmentation_colors = self.get_segmentation_colors()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', help='run locally only')
    parser.add_argument('--no_virtual_cameras', action='store_true', help='do not stream frames to virtual cameras')
    parser.add_argument('--address', type=str, default='https://172.17.15.69:4000' ,help='adress to connect to')
    parser.add_argument('--sim-port', type=int, default=1071 ,help='Simulator open port')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to simulation configuration file')
    parser.add_argument('--video-index', type=int, default=0 ,help='index of the first /dev/video device to start streaming to')
    parser.add_argument('--no-debug-camera', action='store_true', help='do not instantiate debug top down camera')
    parser.add_argument('--log-state', action='store_true', help='Log occupancy maps')
    parser.add_argument('--create-video', action='store_true', help='Create videos for all views')
    parser.add_argument('--scenario', type=int, default=1, help='Choose scenario')
    parser.add_argument('--log', action='store_true', help="Log occupancy maps + create videos")
    parser.add_argument('--showall', action='store_true', help="Show everything in the top view")
    parser.add_argument('--single-weight', type=int, default=0, help="Make all objects of the specified weight")
    parser.add_argument('--single-danger', action='store_true', help="Make all objects dangerous")
    parser.add_argument('--seed', type=int, default=-1, help="Input seed value")
    parser.add_argument('--log-results', type=str, default='', help='Directory where to log results')
    parser.add_argument('--save-map', action='store_true', help="Save the occupancy map")
    parser.add_argument('--no-launch-build', action='store_true', help="Do not launch build")
    parser.add_argument('--sim-binary', default='', help="Location of binary if not auto-launched")
    parser.add_argument('--no-human-test', action='store_true', help="Do not run human tests")
    parser.add_argument('--single-object', action='store_true', help="Single object")
    parser.add_argument('--ai-vision', action='store_true', help="Activate cameras for AI agents")
    parser.add_argument('--level', type=int, default=0, help="Difficulty level [1,2,3]")
    parser.add_argument('--no-block', action='store_true', help="No object will block the way")
    parser.add_argument('--agents-localized', action='store_true', help="Agents will always be localized by other agents")
    
    
    
    
    args = parser.parse_args()
    
    if args.log:
        args.log_state = True
        args.create_video = True
    

    print("Simulator starting")
    

    with open(args.config, 'r') as file:
        cfg = yaml.safe_load(file)

    num_users = cfg['num_humans']
    num_ais = cfg['num_ais']
    
    width = cfg['width']
    height = cfg['height']
    
    global_refresh_sensor = cfg['sensor_waiting_time']

    #The web interface expects to get frames from camera devices. We simulate this by using v4l2loopback to create some virtual webcams to which we forward the frames generated in here
    if not args.no_virtual_cameras:
        for user in range(args.video_index,args.video_index+num_users+1): #One extra camera for the debug video
            cams.append(pyvirtualcam.Camera(width=width, height=height, fps=20, device='/dev/video'+str(user)))
        for ai in range(args.video_index+num_users+1,args.video_index+num_users+1+num_ais):
            cams.append(pyvirtualcam.Camera(width=width, height=height, fps=20, device='/dev/video'+str(ai)))
            
        if args.create_video:
            log_dir = './videos/'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            video_meta_f = open(log_dir+dateTime+'_meta.txt', 'w')

            for c_idx in range(len(cams)):
                video_name = log_dir+dateTime+ '_' + str(c_idx) + '.mp4'
                video_tmp = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), 60, (width,height))
                video.append(video_tmp)
                print("Created ", video_name)
                
            

    address = args.address

    if args.no_launch_build and args.sim_binary:
        Popen([args.sim_binary, "-port " + str(args.sim_port)])


    c = Simulation(args, cfg, launch_build=not args.no_launch_build, port=args.sim_port)

    result = c.run()
    

    while True:       
        print("Simulation ended with result: ", result)
         
        c.__init__(args, cfg, restart=True)

        result = c.run()

#TODO: remove fires, make goal locations valid in tutorial    
