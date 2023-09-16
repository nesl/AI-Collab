from llm_control import LLMControl
from gym_collab.envs.action import Action
import numpy as np
from deepq_control import ReplayMemory
import torch
import time
import pdb
from enum import Enum
import re
import random
import itertools

class HeuristicControl:

    def __init__(self, goal_coords, num_steps, robot_id, env, role, planning):
        self.goal_coords = goal_coords.copy()
        
        self.extended_goal_coords = goal_coords.copy()
        

        self.extended_goal_coords.extend([(g[0]+op[0],g[1]+op[1]) for g in self.goal_coords for op in [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]] if [g[0]+op[0],g[1]+op[1]] not in self.goal_coords])
        

        self.memory_replay = ReplayMemory(10000)
        
        self.num_steps = num_steps
        
        self.robot_id = robot_id
        
        self.env = env
        
        self.wait_time_limit = 5
        self.wait_free_limit = 10
        self.help_time_limit2 = 30
        self.help_time_limit = random.randrange(self.wait_time_limit,30)
        
        self.ending_locations = [[x,y] for x in range(8,13) for y in range(15,19)] #ending locations
        self.ending_locations.remove([12,18]) #To ensure all locations are within communication range
        self.ending_locations.remove([8,18])
        
        self.other_agents = [self.Other_Agent() for r in range(env.action_space["robot"].n-1)]
        
        
        self.original_role = role
        
        
        self.original_planning = planning
        
        self.room_distance = 7
        
    class State(Enum):
        get_closest_object = 0
        init_move = 0 #Exchangable
        sense_area = 1
        init_check_items = 2
        check_items = 3
        move_and_pickup = 4
        pickup_and_move_to_goal = 5
        drop_object = 6
        move_end = 7
        wait_message = 8
        check_neighbors = 9
        follow = 10
        wait_random = 11
        wait_free = 12
        obey = 13
        end_meeting = 14
        init_move_complete = 14 #Exchangable
        waiting_order = 15
        sense_compute = 16
        sense_order = 17
        collect_order = 18
        
    class Other_Agent:
        
        def __init__(self):
            self.current_location = []
            self.next_location = []
            self.next_goal = []
            self.team = ""
            self.carrying = False
            self.items = {}
            self.assignment = "None"
    
        
        
    class MessagePattern:
        @staticmethod
        def location(goal_x,goal_y,next_loc_x,next_loc_y,convert_to_real_coordinates, current_location, carrying, helping):
            real_goal = convert_to_real_coordinates([goal_x,goal_y])
            real_next_location = convert_to_real_coordinates([next_loc_x,next_loc_y])
            real_current_location = convert_to_real_coordinates(current_location)
            output_string = "My goal is (" + str(real_goal[0]) + "," + str(real_goal[1]) + "), I'm moving towards (" + str(real_next_location[0]) + "," + str(real_next_location[1]) + "). My current location is (" + str(real_current_location[0]) + "," + str(real_current_location[1]) + "). "
            
            if carrying:
                output_string += "Carrying object. "
                
            if helping:
                output_string += "Helping " + str(helping[0]) + ". "
                
                
            return output_string
                
            
        @staticmethod
        def location_regex():
            return "My goal is (\(-?\d+\.\d+,-?\d+\.\d+\)), I'm moving towards (\(-?\d+\.\d+,-?\d+\.\d+\)). My current location is (\(-?\d+\.\d+,-?\d+\.\d+\)).( Carrying object.)?( Helping (\w+))?"
            
        @staticmethod
        def item(items,item_idx,object_id, convert_to_real_coordinates):
    
            """ Example
            Object 1 (weight: 1) Last seen in (5.5,5.5) at 00:57. Status Danger: benign, Prob. Correct: 88.1%
            """
        
            message_text = ""
                                
            if items[item_idx]["item_weight"]:
                                
                item_loc = items[item_idx]['item_location']
            
                try:
                    mins, remainder = divmod(items[item_idx]["item_time"][0], 60)
                except:
                    pdb.set_trace()
                secs,millisecs = divmod(remainder,1)
                
                
                time_formatted = '{:02d}:{:02d}'.format(int(mins), int(secs))
                
                real_location = convert_to_real_coordinates(item_loc)
            
                message_text = "Object " + str(object_id) + " (weight: " +  str(items[item_idx]["item_weight"]) + ") Last seen in (" + str(real_location[0]) + "," + str(real_location[1]) + ") at " + time_formatted + ". "
                                    
                if items[item_idx]['item_danger_level'] > 0:
                                    
                    message_text +=  "Status Danger: "
                    if items[item_idx]['item_danger_level'] == 1:
                        message_text += "benign, "
                    else:
                        message_text += "dangerous, "
                        
                    message_text += "Prob. Correct: " + str(round(items[item_idx]["item_danger_confidence"][0]*100,1)) + "%. "
            


            return message_text
            
        @staticmethod
        def item_regex_partial():
            return "Object (\d+) \(weight: (\d+)\) Last seen in (\(-?\d+\.\d+,-?\d+\.\d+\)) at (\d+:\d+)"
        @staticmethod
        def item_regex_full():
            return "Object (\d+) \(weight: (\d+)\) Last seen in (\(-?\d+\.\d+,-?\d+\.\d+\)) at (\d+:\d+).( Status Danger: (\w+), Prob. Correct: (\d+\.\d+)%)?"
        
        @staticmethod
        def sensing_help(object_id):
            return "What do you know about object " + str(object_id) + ". "
            
        @staticmethod
        def sensing_help_regex():
            return "What do you know about object (\d+)" 
            
        @staticmethod    
        def sensing_help_negative_response(object_id):
            return "I know nothing about object " + str(object_id) + ". "
            
        @staticmethod    
        def sensing_help_negative_response_regex():
            return "I know nothing about object (\d+)"
            
        @staticmethod
        def carry_help(object_id, num_robots):
            return "I need " + str(num_robots) + " more robots to help carry object " + str(object_id) + ". "
            
        @staticmethod
        def carry_help_regex():
            return "I need (\d+) more robots to help carry object (\d+)"
            
        @staticmethod
        def carry_help_accept(robot_id):
            return "I can help you " + str(robot_id) + ". "
            
        @staticmethod
        def carry_help_accept_regex():
            return "I can help you (\w+)"
            
        @staticmethod
        def carry_help_participant_reject(robot_id):
            return "I cannot help you " + str(robot_id) + ". "
            
        @staticmethod
        def carry_help_participant_reject_regex():
            return "I cannot help you (\w+)"
            
        @staticmethod
        def carry_help_reject(robot_id):
            return "Nevermind " + str(robot_id) + ". "
            
        @staticmethod
        def carry_help_reject_regex():
            return "Nevermind (\w+)"
            
        @staticmethod
        def carry_help_finish():
            return "No need for more help. "
            
        @staticmethod
        def carry_help_cancel():
            return "Nevermind. "
            
        @staticmethod
        def carry_help_complain(): #(robot_id):
            return "Thanks for nothing. " # " + str(robot_id)
            
        @staticmethod
        def carry_help_complain_regex():
            return "Thanks for nothing (\w+)"
            
        @staticmethod
        def follow(robot_id, teammate_number):
            return "Thanks, follow me " + str(robot_id) + ". You are number " + str(teammate_number) + ". "
            
        @staticmethod
        def follow_regex():
            return "Thanks, follow me (\w+). You are number (\d+)"
            
        @staticmethod
        def wait(robot_id):
            return "I'm going to wait for " + str(robot_id) + " to pass. " 
            
        @staticmethod
        def wait_regex():
            return "I'm going to wait for (\w+) to pass"
            
            
        @staticmethod
        def move_request(robot_id):
            return "Hey " + str(robot_id) + ", I need you to move. " 
            
        @staticmethod
        def move_request_regex():
            return "Hey (\w+), I need you to move"
            
        @staticmethod
        def move_order(robot_id, location, convert_to_real_coordinates):
        
            real_location = convert_to_real_coordinates([location[0],location[1]])
            
            return str(robot_id) + ", move to (" + str(real_location[0]) + "," + str(real_location[1]) + "). " 
            
        @staticmethod
        def move_order_regex():
            return "(\w+), move to (\(-?\d+\.\d+,-?\d+\.\d+\))"
            
        @staticmethod
        def explanation_question(robot_id):
            return "What are you doing " + str(robot_id) + ". "
            
        @staticmethod
        def explanation_question_regex():
            return "What are you doing (\w+)"
            
        @staticmethod
        def pickup(object_id):
            return "Going to pick up object " + str(object_id) + ". "
            
        @staticmethod
        def pickup_regex():
            return "Going to pick up object (\d+)"
            
        @staticmethod
        def sensing():
            return "Sensing area. "
            
        @staticmethod
        def returning():
            return "Returning to goal area. "
            
        @staticmethod
        def explanation_follow(robot_id):
            return "I'm following " + str(robot_id) + ". "
            
        @staticmethod
        def explanation_response(action_index):
        
            response = ""
        
            if State.get_closest_object.value == action_index:
                response = "Figuring out my next objective. "
            elif State.sense_area.value == action_index:
                response = "Sensing area. "
            elif State.init_check_items.value == action_index:
                response = "Going to check my sensing results. "
            elif State.check_items.value == action_index:
                response = "Checking sensing results and selecting best objective. "
            elif State.move_and_pickup.value == action_index:
                response = "Moving to pick up object. "
            elif State.pickup_and_move_to_goal.value == action_index:
                response = "Picking object. "
            elif State.drop_object.value == action_index:
                response = "Dropping object in goal area. "
            elif State.move_end.value == action_index:
                response = "Just dropped object. "
            elif State.wait_message.value == action_index:
                response = "Waiting for others to respond. "
            elif State.check_neighbors.value == action_index:
                response = "Checking nearby robots. "
            elif State.follow.value == action_index:
                response = "Following. "
            else:
                response = "Can't explain. "
            
            return response
            
        @staticmethod
        def order_sense(robot_id, location, convert_to_real_coordinates):
       
            real_location = convert_to_real_coordinates(location)
       
            return str(robot_id) + ", sense location (" + str(real_location[0]) + "," + str(real_location[1]) + "). "
        
        @staticmethod    
        def order_sense_regex():
            return "(\w+), sense location (\(-?\d+\.\d+,-?\d+\.\d+\))"
            
        @staticmethod
        def order_collect(robot_id, object_id):

            return str(robot_id) + ", collect object " + str(object_id) + ". "
        
        @staticmethod    
        def order_collect_regex():
            return "(\w+), collect object (\d+)"
            
        @staticmethod
        def order_collect_group(robot_id, other_robot_ids, object_id):
        
            
            output_string = "Team leader: " + str(robot_id) + ". Helpers: ["
            
            for ori_idx,other_robot_id in enumerate(other_robot_ids):
                if ori_idx:
                    output_string += "," + other_robot_id
                else:
                    output_string += other_robot_id
                    
            output_string += "]. Collect object " + str(object_id) + ". "

            return output_string 
            
        @staticmethod    
        def order_collect_group_regex():
            return "Team leader: (\w+). Helpers: \[(\w+)(,\w+)*\]. Collect object (\d+)"
            
        @staticmethod
        def order_finished():
            return "Order completed. "
            
        @staticmethod
        def task_finished():
            return "Task finished. "
            
        @staticmethod
        def finish():
            return "Let's finish. "
            
        @staticmethod
        def finish_reject():
            return "Wait, not yet. "
        
            
        
    def start(self):
    
        self.action_index = 0
        self.last_action_index = -1
        self.last_action = -1
        self.retries = 0
        self.ignore_object = []
        self.ignore_go_location = []
        self.previous_go_location = []
        self.go_retries = 0
        self.not_dangerous_objects = []
        self.item_index = -1
        self.sensed_items = []
        self.message_text = ""
        self.heavy_objects = {"index": [], "weight": []}
        self.asked_help = False
        self.asked_time = 0
        self.being_helped = []
        self.being_helped_locations = []
        self.being_helped_combinations = []
        self.helping = []
        self.accepted_help = ""
        self.target_location = []
        self.chosen_heavy_object = -1
        self.stuck_retries = 0
        self.message_send_time = float('inf')
        self.next_loc = []
        self.ignore_robots = []
        self.wait_requester = ""
        self.wait_locations = []
        self.previous_message = []
        self.occupied_locations = []
        self.pending_location = []
        self.stuck_time = 0
        self.previous_next_loc = []
        self.potential_occupied_locations = []
        self.stuck_too_much = 0
        self.stuck_moving = 0
        self.too_stuck = 0
        self.stuck_wait_moving = 0
        self.target_object_idx = -1
        self.assigned_target_location = []
        self.just_started = True
        self.planning = self.original_planning
        self.role = self.original_role
        self.finished_robots = []
        
        #self.other_agents = {n:Other_Agent() for n in agents_ids}
        

    def compute_real_distance(self,neighbor_location,ego_location):
    
        res = np.linalg.norm(np.array([neighbor_location[0],neighbor_location[1]]) - np.array([ego_location[0],ego_location[1]]))*self.env.map_config['cell_size']
        
        return res
        
    def get_combinations(self,lst): # creating a user-defined method
        combination = [] # empty list 
        for r in range(1, len(lst) + 1):
            # to generate combination
            combination.extend(itertools.combinations(lst, r))
          
        return combination
        
    def go_to_location(self, x, y, occMap, robotState, info, ego_location, end=False,checking=False):
    
        occMap_clean = np.copy(occMap)
        locations_to_test = [[1,0],[0,1],[1,1],[-1,0],[0,-1],[-1,-1],[-1,1],[1,-1]]

        all_movements = not robotState.object_held
        action = -1 #For checking
        
        try:
            path_to_follow = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),occMap,ignore=self.ignore_go_location, all_movements=all_movements)
        except:
            pdb.set_trace()
        
        
        
        if x == ego_location[0][0] and y == ego_location[1][0]: #In case we are already in the destination
            action = []
            if not checking:
                self.stuck_retries = 0
        elif not path_to_follow: #This means there is no feasible path, maybe because robots are blocking the path
        

            if self.helping:
                agent_idx = info['robot_key_to_index'][self.helping[0]]
                helping_robot_location = robotState.robots[agent_idx]["neighbor_location"]
                occMap_clean[helping_robot_location[0],helping_robot_location[1]] = 1
        
            if self.ignore_robots: #Modify the occMap by removing the robots, to see if that works
            
                nearby_robots = []
                for rb_idx in self.ignore_robots:
                    rb = robotState.robots[rb_idx]["neighbor_location"]
                    
                    if self.helping and rb == helping_robot_location: #Do not ignore the robot you are helping out
                        continue
                        
                    occMap_clean[rb[0],rb[1]] = 0
                    if self.compute_real_distance(rb,[ego_location[0][0],ego_location[1][0]]) < self.env.map_config['communication_distance_limit']:
                        nearby_robots.append(rb)
                    
                
                path_to_follow = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),occMap_clean,ignore=self.ignore_go_location, all_movements=all_movements)
                
                if path_to_follow and occMap[path_to_follow[0][0],path_to_follow[0][1]] != 0 and not checking: #Removing robots does work, make other robots move
                    robot_combinations = self.get_combinations(nearby_robots) 
                    allowed_robots_blocking = (-1,0)
                    for rc_idx,rc in enumerate(robot_combinations): #Check all possible combinations for moving robots (maybe moving one is enough or two are needed)
                        t_occMap = np.copy(occMap_clean)
                        for robot_loc in rc:
                            t_occMap[robot_loc[0],robot_loc[1]] = 1 #We create a mask with locations of robots
                            
                        temp_path_to_follow = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),t_occMap,ignore=self.ignore_go_location, all_movements=all_movements)
                        
                        if temp_path_to_follow and len(rc) >= allowed_robots_blocking[1]:
                            allowed_robots_blocking = (rc_idx,len(rc)) #Save combination index, move the least number of robots.
                    
                    order_robots = []
                            
                            
                    robot_index_to_key = list(info['robot_key_to_index'].keys())
                    for rb_idx in self.ignore_robots:
                        rb = robotState.robots[rb_idx]["neighbor_location"]
                        if (allowed_robots_blocking[0] == -1 or (allowed_robots_blocking[0] > -1 and rb not in robot_combinations[allowed_robots_blocking[0]])) and not (self.helping and rb == helping_robot_location): #If the robot is not in the combination, move it
                            #order_robots.append(rb)
                            for nrobot_idx in range(len(robotState.robots)):
                                if robotState.robots[nrobot_idx]["neighbor_location"] == rb:
                                    robot_id = robot_index_to_key[list(info['robot_key_to_index'].values()).index(nrobot_idx)] #Get the id of the robot
                                    break
                                    
 
                            
                            if self.helping and robot_id == self.helping[0]:
                                pdb.set_trace()
                            self.message_text += self.MessagePattern.move_request(robot_id)
                            if not self.action_index == self.State.wait_free.value and not self.action_index == self.State.wait_random.value:
                                self.last_action_index = self.action_index
                            self.action_index = self.State.wait_free.value
                            self.wait_locations.append(rb)
                            self.asked_time = time.time()
                            
                        
                    action = -1
                    print("Waiting: moving", x,y, path_to_follow)
                    
                    self.stuck_wait_moving += 1
                    
                    if self.stuck_wait_moving == 100:
                        #pdb.set_trace()
                        print("WAIT TOO MUCH!!!")
                        
                    if not re.search(self.MessagePattern.move_request_regex(),self.message_text):
                        pdb.set_trace()
                    
                    
                elif path_to_follow and occMap[path_to_follow[0][0],path_to_follow[0][1]] == 0: #If the next step has no robot move until you are next to a blocking robot
                    action = LLMControl.position_to_action([ego_location[0][0],ego_location[1][0]],path_to_follow[0],False)
                    
                    self.stuck_wait_moving = 0
                else: #We need to wait
                    action = -1
                    if not checking:
                        print("Waiting: Couldn't go to", x,y, path_to_follow)
                        self.ignore_go_location = []
                    
                    #self.stuck_retries += 1
                    #if not path_to_follow:
                    #    pdb.set_trace()
                    #pdb.set_trace()
                    self.stuck_wait_moving = 0
            else:
        
                action = -1
                
                if not checking:
                    print("Couldn't go to", x,y)
                
                    self.stuck_retries += 1
                
                    if self.stuck_retries >= random.randrange(5,20):
                        self.ignore_go_location = []
                        self.stuck_retries = 0
                    
            
                
        elif x == path_to_follow[0][0] and y == path_to_follow[0][1] and occMap[x,y] and not end: #Next location is our destination. Actually we never arrive to the destination if there is already something there, we just stay one cell before.
            action = []
            if not checking:
                self.stuck_retries = 0
        #elif self.helping and (x == path_to_follow[1][0] and y == path_to_follow[1][1] and occMap[x,y]):
        #    action = []
        elif not checking:
            
            self.stuck_retries = 0
            current_location = [ego_location[0][0],ego_location[1][0]]
            
            if self.previous_go_location and path_to_follow[0][0] == self.previous_go_location[0] and path_to_follow[0][1] == self.previous_go_location[1]: #If it gets stuck at location
                if self.go_retries == 5:#2:

                    self.ignore_go_location.append(path_to_follow[0])
                    path_to_follow = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),occMap, ignore=self.ignore_go_location, all_movements=all_movements)
                    print(path_to_follow, self.ignore_go_location)
                    if not path_to_follow: #stuck
                        action = -1
                        print("I'm stuck!")
                        
                    self.go_retries = 0
                else:
                    self.go_retries += 1
            else:
                self.go_retries = 0
                self.ignore_go_location = []
            
            if path_to_follow:
                self.previous_go_location = [path_to_follow[0][0],path_to_follow[0][1]]
                action = LLMControl.position_to_action(current_location,path_to_follow[0],False)
                       
        if not checking:
            print("Retreis:", self.go_retries)   
            
            if action == -1:
                self.stuck_moving += 1
            else:
                self.stuck_moving = 0
            
        return action,path_to_follow
        
    def drop(self):
        return Action.drop_object.value
        
    def pick_up(self, occMap, item_location, ego_location):
        
    
        action = LLMControl.position_to_action([ego_location[0][0],ego_location[1][0]],item_location,True)
        
        return action
        
    
    def activate_sensor(self,robotState, next_observation):

        action = -1
        item = []

        if self.subaction_index == 0:
            self.subaction_index += 1
            action = Action.danger_sensing.value
        elif self.subaction_index == 1:
            if self.num_items < next_observation["num_items"]:
                self.subaction_index += 1

            
        if self.subaction_index == 2:
        
            if self.num_items < next_observation["num_items"]:
                action = Action.check_item.value
                item = self.num_items
                self.num_items += 1
                
        if self.num_items == next_observation["num_items"]:
            self.subaction_index = 0
            

        
        return action,item
        
    def process_sensor(self,robotState, next_observation):

        action = -1
        item = []

        while True:
        
            if self.item_index < next_observation['num_items']:
                
                if self.item_index not in self.sensed_items:
                    action = Action.check_item.value
                    item = self.item_index 
                    self.item_index += 1
                    break
                    
                self.item_index += 1
                
            else:
                break
                
            
        
        return action,item
        
    def send_message(self,message):
        
        action = Action.send_message.value
        
        
        return action,message
        
    
    def find_order_team_rec(self, help_ids, idx, robotState, info, goal_locations, limited_occ_map, num_agents):
    
        
        help_idx = help_ids[idx]
        agent_id = self.being_helped[help_idx]
        agent_idx = info['robot_key_to_index'][agent_id]
        agent_location = robotState.robots[agent_idx]["neighbor_location"]
            
        limited_occ_map_copy = np.copy(limited_occ_map)
        limited_occ_map_copy[agent_location[0],agent_location[1]] = 3
        
        goal_distances = [float("inf")]*len(goal_locations)
        
        for gl_idx,gl in enumerate(goal_locations):
        
            if not limited_occ_map_copy[gl[0],gl[1]] == 1 and not limited_occ_map_copy[gl[0],gl[1]] == 2 and not limited_occ_map_copy[gl[0],gl[1]] == 4:
            
                if gl == agent_location:
                    goal_distances[gl_idx] = 0
                else:
            
                    possible_path = LLMControl.findPath(np.array(agent_location),np.array([gl[0],gl[1]]),limited_occ_map_copy,all_movements=False)
                
                    if possible_path:
                        goal_distances[gl_idx] = len(possible_path)
                        
        ordered_goal_locations = [gl for gl_distance,gl in sorted(zip(goal_distances,goal_locations))]
        
        for gl_distance,gl in sorted(zip(goal_distances,goal_locations)): #recursive
      
                    
            if gl_distance < float("inf"):
            
                if idx == num_agents-1:
                    return [gl] #Return goal location
                    
                limited_occ_map_copy_copy = np.copy(limited_occ_map)
                limited_occ_map_copy_copy[agent_location[0],agent_location[1]] = 0
                limited_occ_map_copy_copy[gl[0],gl[1]] = 1
                
                
                result = self.find_order_team_rec(help_ids,idx+1,robotState,info,goal_locations,limited_occ_map_copy_copy, num_agents)
                
                if result:
                    result.append(gl)
                    return result
                        
        return []
                        
    
    def wait_for_others_func(self,occMap, info, robotState, nearby_other_agents, next_locations, ego_location):
    
        wait_for_others = False    
        combinations_found = True
        within_comms_range = True
                                
        if self.being_helped:
        
            
        
            wait_for_others = True
            
            #pdb.set_trace()

            if self.being_helped_locations:
            
                if not next_locations:
                    help_idx = len(self.being_helped_locations)-1
                else:
                    comb_idx = len(self.being_helped_locations)-1
                    help_idx = self.being_helped_combinations[comb_idx][0]
                
                agent_id = self.being_helped[help_idx]
                agent_idx = info['robot_key_to_index'][agent_id]
                previous_agent_location = robotState.robots[agent_idx]["neighbor_location"]
                
            
            elif next_locations: #At the beginning choose the order of path assignment
            
                #According to next step, we have all the possible cells helping robots may move into
        
                ego_location = [ego_location[0][0],ego_location[1][0]]
                
                res = np.array(next_locations[0]) - np.array(ego_location) 
                
        
                if int(res[0]) == 0 and res[1] > 0: #Left movement
                    range1_1 = ego_location[0]-1
                    range1_2 = ego_location[0]+2
                    range2_1 = ego_location[1]-1
                    range2_2 = ego_location[1]+3
                    additional_goal_loc = [ego_location[0],ego_location[1]-1]
                    
                elif int(res[0]) == 0 and res[1] < 0: #Right movement
                    range1_1 = ego_location[0]-1
                    range1_2 = ego_location[0]+2
                    range2_1 = ego_location[1]-2
                    range2_2 = ego_location[1]+2
                    additional_goal_loc = [ego_location[0],ego_location[1]+1]
                elif res[0] > 0 and int(res[1]) == 0: #Up movement
                    range1_1 = ego_location[0]-1
                    range1_2 = ego_location[0]+3
                    range2_1 = ego_location[1]-1
                    range2_2 = ego_location[1]+2
                    additional_goal_loc = [ego_location[0]-1,ego_location[1]]
                elif res[0] < 0 and int(res[1]) == 0: #Down movement
                    range1_1 = ego_location[0]-2
                    range1_2 = ego_location[0]+2
                    range2_1 = ego_location[1]-1
                    range2_2 = ego_location[1]+2
                    additional_goal_loc = [ego_location[0]+1,ego_location[1]]
                else:
                    pdb.set_trace()
                    
                goal_locations = [[x,y] for x in range(next_locations[0][0]-1,next_locations[0][0]+2,1) for y in range(next_locations[0][1]-1,next_locations[0][1]+2,1) if not (x == next_locations[0][0] and y == next_locations[0][1]) and not (x == ego_location[0] and y == ego_location[1])]
                
                goal_locations.append(additional_goal_loc)
                
                limited_occ_map = np.ones(occMap.shape,int)
                limited_occ_map[range1_1:range1_2,range2_1:range2_2] = occMap[range1_1:range1_2,range2_1:range2_2]
                
                limited_occ_map[ego_location[0],ego_location[1]] = 1
                limited_occ_map[next_locations[0][0],next_locations[0][1]] = 1
                
                for agent_id in self.being_helped: #locations with teammates

                    agent_idx = info['robot_key_to_index'][agent_id]
                    other_robot_location = robotState.robots[agent_idx]["neighbor_location"]
                    limited_occ_map[other_robot_location[0],other_robot_location[1]] = 1

            
                possible_permutations = list(itertools.permutations(list(range(len(self.being_helped)))))
                
                solution_found = []
                for perm in possible_permutations:

                    possible_perm = self.find_order_team_rec(perm,0,robotState,info,goal_locations,limited_occ_map, len(self.being_helped))
                    
                    if possible_perm:
                        solution_found = perm
                        break
                        
                if solution_found:
                    possible_perm.reverse()
                    
                    self.being_helped_combinations = [[solution_found[p_idx],possible_perm[p_idx]] for p_idx in range(len(possible_perm))]
                    print("Combinaionts", self.being_helped_combinations)                    

                else:
                    combinations_found = False
                    self.message_text += self.MessagePattern.carry_help_finish()
                    print("No possible combinations 1")
                
            else: #To wait for others to get into communication range
                agent_sum = 0
                robot_index_to_key = list(info['robot_key_to_index'].keys())
                for noa in nearby_other_agents:
                    robot_id = robot_index_to_key[list(info['robot_key_to_index'].values()).index(noa)]
                    if robot_id in self.being_helped:
                        agent_sum += 1
                            
                if agent_sum != len(self.being_helped):
                    within_comms_range = False
                    
                
                if within_comms_range: #Compute feasible paths
                
        
                    ego_location = [ego_location[0][0],ego_location[1][0]]
                    
                    limited_occ_map = np.copy(occMap)
                    
                    range1_1 = ego_location[0]-1
                    range1_2 = ego_location[0]+2
                    range2_1 = ego_location[1]-1
                    range2_2 = ego_location[1]+2
                    
                    goal_locations = [[x,y] for x in range(ego_location[0]-1,ego_location[0]+2,1) for y in range(ego_location[1]-1,ego_location[1]+2,1) if not (x == ego_location[0] and y == ego_location[1])]
                    limited_occ_map[ego_location[0],ego_location[1]] = 1
                    
                    for agent_id in self.being_helped: #locations with teammates

                        agent_idx = info['robot_key_to_index'][agent_id]
                        other_robot_location = robotState.robots[agent_idx]["neighbor_location"]
                        limited_occ_map[other_robot_location[0],other_robot_location[1]] = 1

            
                    possible_permutations = list(itertools.permutations(list(range(len(self.being_helped)))))
                    
                    solution_found = []
                    for perm in possible_permutations:

                        possible_perm = self.find_order_team_rec(perm,0,robotState,info,goal_locations,limited_occ_map, len(self.being_helped))
                        
                        if possible_perm:
                            solution_found = perm
                            break
                            
                    if solution_found:
                        possible_perm.reverse()
                        
                        self.being_helped_combinations = [[solution_found[p_idx],possible_perm[p_idx]] for p_idx in range(len(possible_perm))]
                        print("Combinaionts", self.being_helped_combinations)                    

                    else:
                        combinations_found = False
                        self.message_text += self.MessagePattern.carry_help_finish()
                        print("No possible combinations 2")
                
            print("Expected locations:", self.being_helped_locations)
               
        
            if within_comms_range and combinations_found and (not self.being_helped_locations or (self.being_helped_locations and self.being_helped_locations[-1] == previous_agent_location and len(self.being_helped_locations) != len(self.being_helped))):

                """
                if not next_locations:

                    
                    for agent_id in self.being_helped: #remove locations with teammates

                        agent_idx = info['robot_key_to_index'][agent_id]
                        other_robot_location = robotState.robots[agent_idx]["neighbor_location"]
                        occMap[other_robot_location[0],other_robot_location[1]] = 3

                    wait_for_others = True
                    
                    help_idx = len(self.being_helped_locations)
                    
                    agent_id = self.being_helped[help_idx]
                    agent_idx = info['robot_key_to_index'][agent_id]
                    agent_location = robotState.robots[agent_idx]["neighbor_location"]
                    new_location = self.find_location_teammate(agent_location, occMap, self.being_helped_locations, next_locations)
                    
                    
                    
                    if not new_location: #One agent is not able to get close
                        wait_for_others = True
                        print("Not able to plan for agent", agent_id)
                    else:
                        self.being_helped_locations.append(new_location)
                        self.message_text += self.MessagePattern.move_order(agent_id, new_location, self.env.convert_to_real_coordinates)
                    
                        print("NEW Location", occMap[new_location[0],new_location[1]])
                        
                        if occMap[new_location[0],new_location[1]] != 0 and occMap[new_location[0],new_location[1]] != 3:
                            pdb.set_trace()
                        
                    
                else:
                """
                comb_idx = len(self.being_helped_locations)
                new_location = self.being_helped_combinations[comb_idx][1]

                self.being_helped_locations.append(new_location)
                
                help_idx = self.being_helped_combinations[comb_idx][0]
                
                agent_id = self.being_helped[help_idx]
                self.message_text += self.MessagePattern.move_order(agent_id, new_location, self.env.convert_to_real_coordinates)
                self.asked_time = time.time()
            
            elif len(self.being_helped_locations) == len(self.being_helped) and self.being_helped_locations[-1] == previous_agent_location: #When all agents have followed orders
                wait_for_others = False
                
            

            
            
                
        return wait_for_others,combinations_found
        
    def wait_movement(self, agent_idx, agent):
    
        self.message_text = self.MessagePattern.wait(agent) #What happens when someone is carrying object
        if not self.action_index == self.State.wait_random.value and not self.action_index == self.State.wait_free.value:
            self.last_action_index = self.action_index
        self.action_index = self.State.wait_random.value
        self.wait_requester = agent_idx
        self.asked_time = time.time()
        
        
    def find_location_teammate(self, agent_location, occMap, other_locs, next_ego_location):
    
        
        ego_location = np.where(occMap == 5)
        
        ego_location = [ego_location[0][0],ego_location[1][0]]
        
        if next_ego_location:
            target_location = next_ego_location[0].tolist()
        else:
            target_location = ego_location
        
        map_side = np.array(target_location)-occMap.shape[0]/2
        
        
        #ADD two cells away
        
        if map_side[0] >= 0 and map_side[1] <= 0: #Up right
            possible_locations = [[0,-1],[1,-1],[1,0],[1,1],[-1,-1],[-1,1],[0,1],[-1,0]]
        elif map_side[0] >= 0 and map_side[1] >= 0: #Up left
            possible_locations = [[0,1],[1,-1],[1,0],[1,1],[-1,-1],[-1,1],[0,-1],[-1,0]]
        elif map_side[0] <= 0 and map_side[1] <= 0: #Down right
            possible_locations = [[0,-1],[-1,1],[-1,0],[-1,-1],[1,-1],[1,1],[0,1,],[1,0]]
        elif map_side[0] <= 0 and map_side[1] >= 0: #Down left
            possible_locations = [[0,1],[-1,1],[-1,0],[-1,-1],[1,1],[1,-1],[0,-1],[1,0]]
        else:
            pdb.set_trace()
        
        next_location = [0,0]
        
        offset_location = 0
        output_location = []
        path_length = float('inf')

        for p_idx,p in enumerate(possible_locations):
            next_location[0] = target_location[0] + p[0]
            next_location[1] = target_location[1] + p[1]
            
            if not target_location == next_location and not ego_location == next_location: #We don't want agent to block current or next position
            
                if agent_location == next_location: 
                    print("BROKE")
                    if occMap[next_location[0],next_location[1]] != 0 and occMap[next_location[0],next_location[1]] != 3:
                            pdb.set_trace()
                    output_location = next_location
                    break
                    
                elif not (other_locs and any(next_location[0] == ol[0] and next_location[1] == ol[1] for ol in other_locs)) and (occMap[next_location[0],next_location[1]] == 0 or occMap[next_location[0],next_location[1]] == 3):

                    possible_path = LLMControl.findPath(np.array(agent_location),np.array([next_location[0],next_location[1]]),occMap,all_movements=False, ignore=[ego_location])
                    
                    
                    if possible_path:
                        if len(possible_path) < path_length and all(self.compute_real_distance(target_location,node) < self.env.map_config['strength_distance_limit']  for node in possible_path):
                            output_location = next_location.copy()
                            path_length = len(possible_path)
                            
                            if occMap[output_location[0],output_location[1]] != 0 and occMap[output_location[0],output_location[1]] != 3:
                                pdb.set_trace()
                            
                            print("PATH",output_location, possible_path, path_length, ego_location, target_location)
                    
                

        return output_location
        
        
        
    """
    def find_location_teammate(self, agent_id, help_index, info, robotState, occMap, following_loc):
    
        agent_idx = info['robot_key_to_index'][agent_id]
        target_location = robotState.robots[agent_idx]["neighbor_location"]
        
        
        map_side = np.array(target_location)-occMap.shape[0]/2
        
        if map_side[0] > 0 and map_side[1] < 1: #Up right
            possible_locations = [[0,-1],[1,-1],[1,0],[1,1],[-1,-1],[-1,1],[0,1]]
        elif map_side[0] > 0 and map_side[1] > 1: #Up left
            possible_locations = [[0,1],[1,-1],[1,0],[1,1],[-1,-1],[-1,1],[0,-1]]
        elif map_side[0] < 0 and map_side[1] < 1: #Down right
            possible_locations = [[0,-1],[-1,1],[-1,0],[-1,-1],[1,-1],[1,1],[0,1]]
        elif map_side[0] < 0 and map_side[1] > 1: #Down left
            possible_locations = [[0,1],[-1,1],[-1,0],[-1,-1],[1,1],[1,-1],[0,-1]]
        else:
            pdb.set_trace()
        
        next_location = [0,0]
        
        offset_location = 0
        output_location = []
        
        for p_idx,p in enumerate(possible_locations):
            next_location[0] = target_location[0] + p[0]
            next_location[1] = target_location[1] + p[1]
            
            #possible_path = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([loc[0],loc[1]]),occMap,all_movements=(not robotState.object_held))
            
            if occMap[next_location[0],next_location[1]] != 0 and not occMap[next_location[0],next_location[1]] == 3 and not occMap[next_location[0],next_location[1]] == 5:
                offset_location += 1
            elif p_idx == help_index + offset_location:
                output_location = next_location
                break
                
        return output_location
    
    """
    
    
    def message_processing(self,received_messages, robotState, info):
    
        action = -1
        last_move_request = ""
    
        for rm in received_messages:
            
            print("Received message:", rm)


            if self.MessagePattern.carry_help_accept(self.robot_id) in rm[1]:
                self.asked_time = time.time()
                
                if self.asked_help:
                    
                    teammate_number = len(self.being_helped)
                    
                    
                    if self.chosen_heavy_object < len(self.heavy_objects["weight"]): 
                        print("Being helped by ", rm[0])
                        
                        self.being_helped.append(rm[0])
                        
                        try:
                            if len(self.being_helped)+1 >= self.heavy_objects["weight"][self.chosen_heavy_object]:
                                self.asked_help = False
                                
                                self.target_location = robotState.items[self.heavy_objects["index"][self.chosen_heavy_object]]['item_location']
                                self.target_object_idx = self.heavy_objects["index"][self.chosen_heavy_object]
                                
                                self.action_index = self.State.move_and_pickup.value
                                self.being_helped_locations = []
                        except:
                            pdb.set_trace()    
                        
                        match_pattern = re.search(self.MessagePattern.location_regex(),self.message_text)
                        
                        if match_pattern and not match_pattern.group(6):
                            self.message_text = self.message_text.replace(match_pattern.group(), match_pattern.group() + " Helping " + self.robot_id + ". ")
                            
                        self.message_text += self.MessagePattern.follow(rm[0],teammate_number)
                    else: #Something happened with the object
                        self.message_text += self.MessagePattern.carry_help_reject(rm[0])
                else:
                    self.message_text += self.MessagePattern.carry_help_reject(rm[0])
                    
                    
            if re.search(self.MessagePattern.carry_help_regex(),rm[1]) and self.role != "scout": # "I need help" in rm[1]:
                rematch = re.search(self.MessagePattern.carry_help_regex(),rm[1])
                
                if re.search(self.MessagePattern.carry_help_regex(),self.message_text): #This means the robot is preparing to ask for help and reject the help request, we shouldn't allow this
                    self.message_text = self.message_text.replace(re.search(self.MessagePattern.carry_help_regex(),self.message_text).group(), "")
                    self.asked_help = False
                    self.asked_time = time.time()
                    self.action_index = self.last_action_index

                
                if not robotState.object_held and not self.helping and not self.being_helped and not self.accepted_help and not self.asked_help: # and not self.asked_help:
                    self.message_text += self.MessagePattern.carry_help_accept(rm[0])
                    self.accepted_help = rm[0]
                    

                    #self.helping = rm[0]
                    #self.action_index = self.State.check_neighbors.value
                    
                else:
                    self.message_text += self.MessagePattern.carry_help_participant_reject(rm[0])
                    print("Cannot help", not robotState.object_held, not self.helping, not self.being_helped, not self.accepted_help, not self.asked_help)
                    
                    
                    
            if re.search(self.MessagePattern.follow_regex(),rm[1]):
            
                for rematch in re.finditer(self.MessagePattern.follow_regex(),rm[1]):
            
                    if rematch.group(1) == self.robot_id:
                
                        teammate_number = int(rematch.group(2))
                        
                        self.helping = [rm[0],teammate_number]
                        
                        self.action_index = self.State.follow.value
                        
                        
                        print("HELPING")
                        break
                
            if self.MessagePattern.carry_help_cancel() in rm[1] or self.MessagePattern.carry_help_reject(self.robot_id) in rm[1] or self.MessagePattern.carry_help_finish() in rm[1] or self.MessagePattern.carry_help_complain() in rm[1]:
            
                
                
                if self.helping and self.helping[0] == rm[0]:
                    self.accepted_help = ""
                    self.action_index = self.State.get_closest_object.value
                    print("Changed -3")
                
                    self.helping = []
                elif self.accepted_help == rm[0]:
                    self.accepted_help = ""
                
            #if self.MessagePattern.carry_help_participant_reject(self.robot_id) in rm[1]:
            #    #self.asked_help = False
            #    self.asked_time = time.time()
                
            if re.search(self.MessagePattern.sensing_help_regex(),rm[1]): #"What do you know about object " in rm[1]:
                rematch = re.search(self.MessagePattern.sensing_help_regex(),rm[1])
                object_id = rematch.group(1) #rm[1].strip().split()[-1] 
                object_idx = info['object_key_to_index'][object_id]
                
                self.message_text += self.MessagePattern.item(robotState.items,object_idx,object_id, self.env.convert_to_real_coordinates)
                
                if not self.message_text:
                     self.message_text += self.MessagePattern.sensing_help_negative_response(object_id)
            if re.search(self.MessagePattern.item_regex_full(),rm[1]):
            
                for rematch in re.finditer(self.MessagePattern.item_regex_full(),rm[1]):
                
                    object_id = rematch.group(1)
                    object_idx = info['object_key_to_index'][object_id]
                    
                    item = {}
                    
                    last_seen = list(eval(rematch.group(3)))
                    item["item_location"] = self.env.convert_to_grid_coordinates(last_seen)
                    last_time = rematch.group(4).split(":")
                    item["item_time"] = [int(last_time[1]) + int(last_time[0])*60]
                    item["item_weight"] = int(rematch.group(2))
                    item["item_danger_level"] = 0
                    item["item_danger_confidence"] = []
                    
                    
                    if rematch.group(5):
                        
                        if "benign" in rematch.group(6):
                            danger_level = 1
                            
                        else:
                            danger_level = 2
                    
                        item["item_danger_level"] = danger_level
                        item["item_danger_confidence"] = [float(rematch.group(7))/100]
                        print("update estimates", danger_level,item["item_danger_confidence"])
                        
                    robotState.update_items(item,object_idx) #Object gets updated based on higher confidence estimates
                    
                    if robotState.items[object_idx]["item_danger_level"] == 1:
                    
                        if object_idx not in self.not_dangerous_objects:
                    
                            self.not_dangerous_objects.append(object_idx)
                            
                            
                        if item["item_weight"] > 1 and object_idx in self.heavy_objects["index"]:
                            oi = self.heavy_objects["index"].index(object_idx)
                            del self.heavy_objects["index"][oi]
                            del self.heavy_objects["weight"][oi]
                        
                    elif robotState.items[object_idx]["item_danger_level"] == 2:
                    
                        if object_idx in self.not_dangerous_objects:
                            self.not_dangerous_objects.remove(object_idx)
                            
                        if item["item_weight"] > 1 and object_idx not in self.heavy_objects["index"]:
                            self.heavy_objects["index"].append(object_idx)
                            self.heavy_objects["weight"].append(item["item_weight"])
                        
                    
                    agent_idx = info['robot_key_to_index'][rm[0]]
                    
                    if object_id not in self.other_agents[agent_idx].items.keys():
                        self.other_agents[agent_idx].items[object_id] = {"danger_level":0,"confidence":0}
                        

                    self.other_agents[agent_idx].items[object_id]["danger_level"] = item["item_danger_level"] #Update estimates about what other robots know
                    
                    if item["item_danger_confidence"]:
                        self.other_agents[agent_idx].items[object_id]["confidence"] = item["item_danger_confidence"][0]
                    else:
                        self.other_agents[agent_idx].items[object_id]["confidence"] = 0


                    
            if re.search(self.MessagePattern.location_regex(),rm[1]) and not (self.helping and self.helping[0] == rm[0] and self.action_index == self.State.obey.value) and not (self.being_helped and rm[0] in self.being_helped and self.action_index == self.State.drop_object.value) and not self.action_index == self.State.wait_message.value: #"Going towards location" in rm[1]: 
                match_pattern = re.search(self.MessagePattern.location_regex(),rm[1])

                print("location_regex", self.being_helped)

                #pdb.set_trace()
                other_target_location = self.env.convert_to_grid_coordinates(eval(match_pattern.group(1)))
                other_next_step = self.env.convert_to_grid_coordinates(eval(match_pattern.group(2)))

                agent_idx = info['robot_key_to_index'][rm[0]]
                
                if match_pattern.group(6): #Register whether other agents have already a team
                    self.other_agents[agent_idx].team = match_pattern.group(6)
                else:
                    self.other_agents[agent_idx].team = ""
                    
                if match_pattern.group(4):
                    self.other_agents[agent_idx].carrying = True
                else:
                    self.other_agents[agent_idx].carrying = False
                    
                    
                if self.helping and self.helping[0] == rm[0] and not match_pattern.group(6): #This means the team leader disbanded the team without us knowing
                    self.helping = []
                    self.accepted_help = ""
                    self.action_index = self.State.get_closest_object.value
                    print("Changed -2")
                    
                
                curr_loc = tuple(self.env.convert_to_grid_coordinates(eval(match_pattern.group(3))))
                
                if curr_loc not in self.occupied_locations:
                    self.occupied_locations.append(curr_loc)
                
                if other_next_step == other_target_location: #robot stays there
                    
                    if agent_idx not in self.ignore_robots:
                        self.ignore_robots.append(agent_idx)
                        
                    if self.target_location == other_target_location and not self.action_index == self.State.follow.value and not self.action_index == self.State.obey.value: #Change destination
                        self.action_index = self.State.get_closest_object.value
                        print("Changed -1")
                        if self.being_helped:
                            self.being_helped = []
                            self.being_helped_locations = []
                            self.message_text += self.MessagePattern.carry_help_finish()
                else:
                    if self.target_location == other_target_location:
                    
                    
                        if not match_pattern.group(4) and not robotState.object_held and not match_pattern.group(5) and not self.helping and not self.being_helped: #Possible change !!!!
                    
                     
                    
                            if rm[2] <= self.message_send_time: #Message arrive at the same time or previous than this robot sent its message.  
                            
                                if rm[2] == self.message_send_time: #rules to disambiguate are based on alphabetic order
                                    
                                    if ord(rm[0]) < ord(self.robot_id): #If sender's id appears first than receiver in alphabetic order
                                        self.ignore_object.append(other_target_location)
                                        print("Changed 0")
                                        self.action_index = self.State.get_closest_object.value
                                        
                                        if self.being_helped:
                                            self.being_helped = []
                                            self.being_helped_locations = []
                                            self.message_text += self.MessagePattern.carry_help_finish()
                                        
                                   
                                else:
                                    self.ignore_object.append(other_target_location)
                                    print("Changed 1")
                                    self.action_index = self.State.get_closest_object.value
                                    if self.being_helped:
                                        self.being_helped = []
                                        self.being_helped_locations = []
                                        self.message_text += self.MessagePattern.carry_help_finish()                                
                                    
                                    
                                
                                if re.search(self.MessagePattern.location_regex(),self.message_text):
                                    self.message_text = self.message_text.replace(re.search(self.MessagePattern.location_regex(),self.message_text).group(), "")
                                    
                                    if self.message_text.isspace():    
                                        self.message_text = ""
                                    print("changing going location!!!")
                                    

                        elif (match_pattern.group(5) and (self.helping or self.being_helped)) or (match_pattern.group(4) and robotState.object_held) or (match_pattern.group(4) and (self.helping or self.being_helped)) or (match_pattern.group(5) and robotState.object_held):
                            if match_pattern.group(5):
                                other_id = match_pattern.group(6)
                            else:
                                other_id = rm[0]
                                
                            if self.helping: 
                                our_id = self.helping[0]
                            else:
                                our_id = self.robot_id
                                    
                                    
                            if ord(other_id) < ord(our_id): #If sender's id appears first than receiver in alphabetic order
                                #Move to free location
                                self.ignore_object.append(other_target_location)
                                print("Changed 2")
                                self.action_index = self.State.get_closest_object.value
                                if self.being_helped:
                                    self.being_helped = []
                                    self.being_helped_locations = []
                                    self.message_text += self.MessagePattern.carry_help_finish()
                        
                        elif match_pattern.group(5) or match_pattern.group(4):
                            self.ignore_object.append(other_target_location)
                            self.action_index = self.State.get_closest_object.value
                            print("Changed 3")
                            if self.being_helped:
                                self.being_helped = []
                                self.being_helped_locations = []
                                self.message_text += self.MessagePattern.carry_help_finish()
                                             
                    else: #If we are not going to same destination, just ignore temporarily the other location
                        self.ignore_object.append(other_target_location)
                        
                        
                
                
                    if self.next_loc:

                    
           
                
                        if (other_next_step == self.next_loc[0].tolist() or (len(self.next_loc) > 1 and other_next_step == self.next_loc[1].tolist())):
                        
                            if rm[2] <= self.message_send_time and not match_pattern.group(4) and not robotState.object_held and not match_pattern.group(5) and not self.helping and not self.being_helped: #Message arrive at the same time or previous than this robot sent its message. This condition is true only when robots have no teams and are not carrying any object
                            
                                if rm[2] == self.message_send_time: #If helping the one who send the message, automatically wait

                                    other_id = rm[0]
                                    our_id = self.robot_id
                                    
                                    if ord(other_id) < ord(our_id): #If sender's id appears first than receiver in alphabetic order
                                        #Move to free location
                                        print(rm[2],self.message_send_time)
                                        self.wait_movement(agent_idx,rm[0])
                                else:
                                    print(rm[2],self.message_send_time)
                                    self.wait_movement(agent_idx,rm[0])

                            elif (match_pattern.group(5) and (self.helping or self.being_helped)) or (match_pattern.group(4) and robotState.object_held) or (match_pattern.group(4) and (self.helping or self.being_helped)) or (match_pattern.group(5) and robotState.object_held): #Priority given to robot teamleader or robot carrying object with robot id that appears first in alphabetic order
                            
                                if match_pattern.group(5):
                                    other_id = match_pattern.group(6)
                                else:
                                    other_id = rm[0]
                                    
                                if self.helping: 
                                    our_id = self.helping[0]
                                else:
                                    our_id = self.robot_id
                                        
                                        
                                if ord(other_id) < ord(our_id): #If sender's id appears first than receiver in alphabetic order
                                    #Move to free location
                                    print(rm[2],self.message_send_time)
                                    self.wait_movement(agent_idx,rm[0])

                            elif match_pattern.group(5) or match_pattern.group(4): #If we are not carrying an object while the other is, or we are not part of a team while the other is
                                print(rm[2],self.message_send_time)
                                self.wait_movement(agent_idx,rm[0])
                                
                        else:

                            if match_pattern.group(5):
                                other_id = match_pattern.group(6)
                            else:
                                other_id = rm[0]
                                
                            if self.helping: 
                                our_id = self.helping[0]
                            else:
                                our_id = self.robot_id
                            
                            other_id = rm[0]
                            our_id = self.robot_id
                                
                            previous_index = -1
                            
                            if match_pattern.group(5) or match_pattern.group(4) or (ord(other_id) < ord(our_id) and (((match_pattern.group(5) and (self.helping or self.being_helped)) or (match_pattern.group(4) and robotState.object_held) or (match_pattern.group(4) and (self.helping or self.being_helped)) or (match_pattern.group(5) and robotState.object_held)) or not match_pattern.group(4) and not robotState.object_held and not match_pattern.group(5) and not self.helping and not self.being_helped)):
                            
                                for s_idx,s in enumerate(self.potential_occupied_locations):
                                    if s[0] == other_next_step:
                                        previous_index = s_idx
                                    
                                if previous_index == -1:
                                    self.potential_occupied_locations.append([other_next_step,time.time()])
                                    
                                else:
                                    self.potential_occupied_locations[previous_index][1] = time.time()
                                        
                                        
                    
            if self.MessagePattern.explanation_question(self.robot_id) in rm[1]:
                self.message_text += self.MessagePattern.explanation_response(self.action_index)
                
            if self.MessagePattern.wait(self.robot_id) in rm[1]:
                agent_idx = info['robot_key_to_index'][rm[0]]
                #other_robot_location = robotState.robots[agent_idx]["neighbor_location"]
                #self.ignore_robots.append(other_robot_location)
                if agent_idx not in self.ignore_robots:
                    self.ignore_robots.append(agent_idx)
            
            if self.MessagePattern.move_request(self.robot_id) in rm[1]: # and not (last_move_request and last_move_request == rm[0]):
            
                
                
                if not robotState.object_held and not self.being_helped and (not self.helping or (self.helping and self.helping[0] == rm[0])) and not self.planning == "coordinator": #This condition is true only when robots have no teams and are not carrying any object
                            
                    print("MOVING")
                    
                    #last_move_request = rm[0]
                    agent_idx = info['robot_key_to_index'][rm[0]]
                    other_robot_location = robotState.robots[agent_idx]["neighbor_location"]
                    possible_locations = [[1,0],[0,1],[-1,0],[0,-1]] #,[1,1],[1,-1],[-1,1],[-1,-1]]
                    ego_location = np.where(robotState.latest_map == 5)
                    
                    maximum_distance = 0
                    maximum_distance_with_robot = 0
                    
                    for p in possible_locations:
                        ego_location2 = [ego_location[0][0] + p[0],ego_location[1][0] + p[1]]

                        if robotState.latest_map[ego_location2[0],ego_location2[1]] == 0:
                            temp_distance = self.compute_real_distance([other_robot_location[0],other_robot_location[1]],ego_location2)
                            if temp_distance > maximum_distance:
                                next_location = p
                                maximum_distance = temp_distance
                        elif robotState.latest_map[ego_location2[0],ego_location2[1]] == 3 and ego_location2 != other_robot_location:
                            temp_distance = self.compute_real_distance([other_robot_location[0],other_robot_location[1]],ego_location2)
                            if temp_distance > maximum_distance_with_robot:
                                next_location_with_robot = p
                                maximum_distance_with_robot = temp_distance
                    
                    if maximum_distance:            
                        ego_location2 = [ego_location[0][0] + next_location[0],ego_location[1][0] + next_location[1]]
                        self.pending_location = ego_location2
                         
                        if not self.action_index == self.State.wait_random.value and not self.action_index == self.State.wait_free.value:
                            self.last_action_index = self.action_index
                        self.action_index = self.State.wait_random.value
                        self.wait_requester = agent_idx
                        self.asked_time = time.time()
                        print("MOVING", action)
                    elif maximum_distance_with_robot:
                        #object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.heavy_objects['index'][ho])]
                        robot_index_to_key = list(info['robot_key_to_index'].keys())
                        self.message_text += self.MessagePattern.move_request(robot_index_to_key[list(info['robot_key_to_index'].values()).index(agent_idx)])
                        if not self.action_index == self.State.wait_free.value and not self.action_index == self.State.wait_random.value:
                            self.last_action_index = self.action_index
                        self.action_index = self.State.wait_free.value
                        self.asked_time = time.time()
                        self.wait_locations.append(next_location_with_robot)
                        print("MOVING robot", action)
                
                
            if re.search(self.MessagePattern.move_order_regex(),rm[1]):
                rematch = re.search(self.MessagePattern.move_order_regex(),rm[1])
                
                if rematch.group(1) == self.robot_id and self.helping:
                    self.target_location = self.env.convert_to_grid_coordinates(eval(rematch.group(2)))
                    self.action_index = self.State.obey.value
                    
                    
            if re.search(self.MessagePattern.order_sense_regex(),rm[1]):
                for rematch in re.finditer(self.MessagePattern.order_sense_regex(),rm[1]):
                    if rematch.group(1) == self.robot_id:
                        self.assigned_target_location = self.env.convert_to_grid_coordinates(eval(rematch.group(2)))
                        self.role = "scout"
                        self.action_index = self.State.get_closest_object.value
                        
            if re.search(self.MessagePattern.order_collect_regex(),rm[1]):
                for rematch in re.finditer(self.MessagePattern.order_collect_regex(),rm[1]):
                    if rematch.group(1) == self.robot_id:
                        object_idx = info['object_key_to_index'][rematch.group(2)]
                        self.target_location = robotState.items[object_idx]["item_location"] #Check assignment
                        self.target_object_idx = object_idx
                        self.role = "lifter"
                        self.action_index = self.State.move_and_pickup.value
                        self.being_helped_locations = []
                        
            if re.search(self.MessagePattern.order_collect_group_regex(),rm[1]):
                for rematch in re.finditer(self.MessagePattern.order_collect_group_regex(),rm[1]):
                    if rematch.group(1) == self.robot_id:
                        object_idx = info['object_key_to_index'][rematch.group(4)]
                        self.role = "lifter"
                        self.asked_time = time.time()
                        self.being_helped.append(rematch.group(2))
                        
                        if rematch.group(3):
                            self.being_helped.extend(rematch.group(3).split(",")[1:])
                    
                        
                        self.target_location = robotState.items[object_idx]['item_location']
                        self.target_object_idx = object_idx
                        
                        self.action_index = self.State.move_and_pickup.value
                        self.being_helped_locations = []
                        
                    
                        match_pattern = re.search(self.MessagePattern.location_regex(),self.message_text)
                        
                        if match_pattern and not match_pattern.group(6):
                            self.message_text = self.message_text.replace(match_pattern.group(), match_pattern.group() + " Helping " + self.robot_id + ". ")
                            
                    elif rematch.group(2) == self.robot_id or (rematch.group(3) and self.robot_id in rematch.group(3).split(",")):
                    

                        self.role = "lifter"
                        
                        self.helping = [rematch.group(1),0]
                        
                        self.action_index = self.State.follow.value
                        
                        

            if self.MessagePattern.order_finished() in rm[1] and self.planning == "coordinator":  
                robot_idx = info['robot_key_to_index'][rm[0]]
                
                self.other_agents[robot_idx].assignment = ""
                
            if self.MessagePattern.task_finished() in rm[1] and self.planning == "coordinated":
                
                self.role = "lifter"
                self.planning = "equal"
                
                
            if self.MessagePattern.finish() in rm[1]:
                if rm[0] not in self.finished_robots:
                    self.finished_robots.append(rm[0])
                    
                    
            if self.MessagePattern.finish_reject() in rm[1]:
                if rm[0] in self.finished_robots:
                    self.finished_robots.remove(rm[0])
                
                    
        return action  
        
    def check_team_arrangement(self, item_location, occMap, weight, ego_location): #We need to know if it's even possible for a team to arrange itself in order to carry the object
        
        sub_occMap = occMap[item_location[0]-1:item_location[0]+2,item_location[1]-1:item_location[1]+2]
        
        free_locs = np.where((sub_occMap == 0) | (sub_occMap == 3))
        num_possibles = 0
        
        for fl_idx in range(len(free_locs[0])):
            
            new_fl = [free_locs[0][fl_idx] + item_location[0]-1, free_locs[1][fl_idx] + item_location[1]-1]
            
        
            possible_path = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([new_fl[0],new_fl[1]]),occMap)
            
            if possible_path:
                num_possibles += 1
                
        clean_occMap = np.copy(occMap)
        
        for sg in self.extended_goal_coords:
            clean_occMap[sg[0],sg[1]] = 0
            
        clean_occMap[item_location[0],item_location[1]] = 0
                
        possible_path = LLMControl.findPath(np.array([item_location[0],item_location[1]]),np.array([sg[0],sg[1]]),clean_occMap)
        
        free_locs_path = []
        
        for p in possible_path: #Check if there is space through all the trajectory
            sub_occMap = clean_occMap[p[0]-1:p[0]+2,p[1]-1:p[1]+2]
        
            free_locs = len(np.where((sub_occMap == 0) | (sub_occMap == 3))[0])
            
            free_locs_path.append(free_locs)
                
        return num_possibles >= weight and all(f >= weight for f in free_locs_path)

    def check_safe_direction(self, location):
        
        for ax in location:
            if ax < 0:
                return False
            elif ax >= self.env.map_config['num_cells'][0]:
                return False
                
        return True

    def cancel_cooperation(self, message=""):
    
        self.action_index = self.State.get_closest_object.value
        
        if message:
            self.message_text += message
        self.asked_time = time.time()
        self.being_helped = []
        self.being_helped_locations = []
        action = Action.get_occupancy_map.value
        
        return action
        
    def modify_occMap(self,robotState, occMap, ego_location, info):
    
        if robotState.object_held: #Eliminate all carried objects from the occupancy map if robot is carrying object
            carried_objects = np.where(occMap == 4)
            for c_idx in range(len(carried_objects[0])):
                occMap[carried_objects[0][c_idx],carried_objects[1][c_idx]] = 0
            

        print(self.occupied_locations)

        for rob_loc_idx in reversed(range(len(self.occupied_locations))): #Make sure agents don't move to locations already occupied
        
            other_robot_location = self.occupied_locations[rob_loc_idx]
            if occMap[other_robot_location[0],other_robot_location[1]] != 5:
                if occMap[other_robot_location[0],other_robot_location[1]] == 0:
                    del self.occupied_locations[rob_loc_idx]  
                elif occMap[other_robot_location[0],other_robot_location[1]] == 3 and self.next_loc and self.next_loc[0][0] == other_robot_location[0] and self.next_loc[0][1] == other_robot_location[1]:
                    occMap[other_robot_location[0],other_robot_location[1]] = 1
            else:
                del self.occupied_locations[rob_loc_idx]

        
        
        for rob_loc_idx in reversed(range(len(self.ignore_robots))): #We mark ignored robots as an object
            other_robot_location = robotState.robots[self.ignore_robots[rob_loc_idx]]["neighbor_location"]

            if self.compute_real_distance([other_robot_location[0],other_robot_location[1]],[ego_location[0][0],ego_location[1][0]]) >= self.env.map_config['communication_distance_limit']:
                del self.ignore_robots[rob_loc_idx]
            elif occMap[other_robot_location[0],other_robot_location[1]] != 5:
                occMap[other_robot_location[0],other_robot_location[1]] = 1
        
        #Make sure possible directions are not blocked by other robots
        for direction in [[0,1],[1,0],[-1,0],[0,-1]]:
            new_direction = [ego_location[0][0] + direction[0],ego_location[1][0] + direction[1]]
            
            if not self.check_safe_direction(new_direction):
                continue
            
            if occMap[new_direction[0],new_direction[1]] == 3: 
                occMap[new_direction[0],new_direction[1]] = 1
                
                
        print("Potential locations before", self.potential_occupied_locations)
        for pot_idx in reversed(range(len(self.potential_occupied_locations))): #Potentially occupied locations, eliminate after some time
            
            if time.time() - self.potential_occupied_locations[pot_idx][1] > 5 or occMap[self.potential_occupied_locations[pot_idx][0][0],self.potential_occupied_locations[pot_idx][0][1]]: #Seconds to eliminate
                del self.potential_occupied_locations[pot_idx]
            else:
                occMap[self.potential_occupied_locations[pot_idx][0][0],self.potential_occupied_locations[pot_idx][0][1]] = 1
        
        print("Potential locations", self.potential_occupied_locations)
        
        if self.action_index != self.State.drop_object.value and robotState.object_held:
            for agent_id in self.being_helped: #if you are being helped, ignore locations of your teammates

                agent_idx = info['robot_key_to_index'][agent_id]
                other_robot_location = robotState.robots[agent_idx]["neighbor_location"]
                occMap[other_robot_location[0],other_robot_location[1]] = 3
            
            
        #Make sure the ego location is always there
        
        occMap[ego_location[0][0],ego_location[1][0]] = 5
        
        
    def get_neighboring_agents(self, robotState, ego_location):
    
        nearby_other_agents = []
        #Get number of neighboring robots at communication range
        for n_idx in range(len(robotState.robots)):
            if "neighbor_location" in robotState.robots[n_idx] and self.compute_real_distance([robotState.robots[n_idx]["neighbor_location"][0],robotState.robots[n_idx]["neighbor_location"][1]],[ego_location[0][0],ego_location[1][0]]) < self.env.map_config['communication_distance_limit']:
                nearby_other_agents.append(n_idx)
                
        return nearby_other_agents

    def carry_heavy_object(self, robotState, ego_location, nearby_other_agents, info):
    
        num_neighbors = len(nearby_other_agents)
    
        for ho_idx in reversed(range(len(self.heavy_objects["index"]))): #Eliminate heavy objects if they have already been taken care of
            if tuple(robotState.items[self.heavy_objects["index"][ho_idx]]['item_location']) in self.extended_goal_coords:
                del self.heavy_objects["index"][ho_idx]
                del self.heavy_objects["weight"][ho_idx]
                            
        print("Asked help", self.asked_help)  
        if self.role != "scout" and self.heavy_objects["index"] and not robotState.object_held and not self.accepted_help and not self.being_helped and not self.asked_help and time.time() - self.asked_time > self.help_time_limit: #Ask for help to move heavy objects 
            
            
                    
            order_heavy_objects_ind = np.argsort(self.heavy_objects['weight'])[::-1] #Depening on the number of neighbors, ask for help for a specific object
                    
            unavailable_robots = sum(1 if self.other_agents[oa_idx].team or self.other_agents[oa_idx].carrying else 0 for oa_idx in range(len(self.other_agents)) if oa_idx in nearby_other_agents)
                    
            chosen_object = -1
            
            for ho in order_heavy_objects_ind: #Select the object closest to ego
            
                    
                if self.heavy_objects["index"][ho] in self.sensed_items and self.heavy_objects['weight'][ho] <= num_neighbors+1 - unavailable_robots and self.check_team_arrangement(robotState.items[self.heavy_objects["index"][ho]]['item_location'],robotState.latest_map, self.heavy_objects['weight'][ho], ego_location): #Heavy object should have been sensed by robot, there should be enough nearby robots, and it should be feasible
                
                    if chosen_object == -1 or (chosen_object > -1 and self.compute_real_distance(robotState.items[self.heavy_objects["index"][ho]]['item_location'],[ego_location[0][0],ego_location[1][0]]) < self.compute_real_distance(robotState.items[self.heavy_objects["index"][chosen_object]]['item_location'],[ego_location[0][0],ego_location[1][0]])):
                        chosen_object = ho

                    
            
            if chosen_object > -1:
                object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.heavy_objects['index'][chosen_object])]
                self.message_text += self.MessagePattern.carry_help(object_id,self.heavy_objects['weight'][chosen_object]-1)
                self.asked_help = True
                self.asked_time = time.time()
                if not self.action_index == self.State.wait_free.value and not self.action_index == self.State.wait_random.value:
                    self.last_action_index = self.action_index
                self.action_index = self.State.wait_message.value
                self.chosen_heavy_object = chosen_object
                print("ASKING HELP")        

    

    
    def exchange_sensing_info(self, robotState, info, nearby_other_agents):
    
        
        object_info_message = []
        
        for noa in nearby_other_agents:   
            
            for item_idx in range(len(robotState.items)):
                danger_level = robotState.items[item_idx]["item_danger_level"]
                
               
                if danger_level > 0:
                
                    confidence = robotState.items[item_idx]["item_danger_confidence"][0]
                
                    object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(item_idx)]
                    
                    if object_id not in self.other_agents[noa].items.keys():
                        self.other_agents[noa].items[object_id] = {"danger_level":0,"confidence":0}
                    
                    if not (self.other_agents[noa].items[object_id]["danger_level"] == danger_level and self.other_agents[noa].items[object_id]["confidence"] == confidence):
                    
                        if object_id not in object_info_message:
                            object_info_message.append(object_id)
                            self.message_text += self.MessagePattern.item(robotState.items,item_idx,object_id, self.env.convert_to_real_coordinates)
                            
                        self.other_agents[noa].items[object_id]["danger_level"] = danger_level
                        self.other_agents[noa].items[object_id]["confidence"] = confidence
            
            
    def central_planning(self, robotState, info, occMap, ego_location, nearby_other_agents):
        
        sensing_agents_per_object = 1 #How many robots to sense each room
        
        if self.action_index == self.State.init_move.value:
        
            
            action = self.return_to_meeting_point(occMap, robotState, info, ego_location)
            
        elif self.action_index == self.State.init_move_complete.value:
        
            action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
        
            neighbor_locations = np.where(occMap == 3)
        
            print([1 if not rob.assignment else 0 for rob in self.other_agents])
        
            #all_robots_in_place = all([1 if not rob.assignment else 0 for rob in self.other_agents])
            all_robots_in_place = all([1 if [neighbor_locations[0][rob_idx],neighbor_locations[1][rob_idx]] in self.ending_locations else 0 for rob_idx in range(len(neighbor_locations[0]))])
                
        
            if not action and isinstance(action, list):
            
           
                if all_robots_in_place:
                
                    for rob in self.other_agents:
                        rob.assignment = ""
                
                    self.action_index = self.State.sense_compute.value
                    self.assigned_item_locations = []
            
                action = Action.get_occupancy_map.value
                
                
    
        elif self.action_index == self.State.sense_compute.value: #Calculate all clusters of object locations
            item_locations = np.where(occMap == 2)    
            
            
            
            for assigned_item in range(len(item_locations[0])):
            
                distance = float("inf")
                
                item_loc = [item_locations[0][assigned_item],item_locations[1][assigned_item]]
                
                if self.assigned_item_locations:
                    for ail1 in self.assigned_item_locations:
                        distance = np.linalg.norm(np.array(ail1)-np.array(item_loc))

                        if distance < self.room_distance:
                            break
                            
                if distance < self.room_distance:
                    continue
                    
                _,possible_path = self.go_to_location(item_loc[0],item_loc[1],occMap,robotState,info,ego_location,checking=True)
                
                if possible_path:
                    object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(assigned_item)]
                    self.assigned_item_locations.append(item_loc)
                    
            
            for sa in range(sensing_agents_per_object-1):
                self.assigned_item_locations.extend(self.assigned_item_locations)
            
            self.action_index = self.State.sense_order.value
            action = Action.get_occupancy_map.value
            self.target_object_idx = 0
            
        elif self.action_index == self.State.sense_order.value:
        
            for rob in nearby_other_agents:
                if not self.other_agents[rob].assignment:
                
                    self.other_agents[rob].assignment = "-1" #Sensing
                    robot_index_to_key = list(info['robot_key_to_index'].keys())
                    robot_id = robot_index_to_key[list(info['robot_key_to_index'].values()).index(rob)]
                    
                    self.message_text += self.MessagePattern.order_sense(robot_id, self.assigned_item_locations[self.target_object_idx], self.env.convert_to_real_coordinates)
                
                    self.target_object_idx += 1
                    
                    if self.target_object_idx == len(self.assigned_item_locations):
                        break
            
                        
            if self.target_object_idx == len(self.assigned_item_locations):
                self.action_index = self.State.collect_order.value
                self.target_object_idx = 0
                
            action = Action.get_occupancy_map.value
                
        elif self.action_index == self.State.collect_order.value:
        
            agents_not_busy = [rob for rob in nearby_other_agents if not self.other_agents[rob].assignment]
        
            missing_objects = []
            
            all_missing_objects = []
            
            for i_idx in range(len(robotState.items)):
                item_location = robotState.items[i_idx]["item_location"]
                
                object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(i_idx)]
                        
                if tuple(item_location) not in self.extended_goal_coords and robotState.items[i_idx]["item_danger_level"] == 2 and not any(1 if rob2.assignment == str(object_id) else 0 for rob2 in self.other_agents) and len(robotState.robots) + 1 != robotState.items[i_idx]["item_weight"]:
                    missing_objects.append(i_idx) 
                    print("Item location:",object_id,item_location)
                    all_missing_objects.append(i_idx)
                #elif robotState.items[i_idx]["item_danger_level"] == 0 and not len(robotState.robots) + 1 == robotState.items[i_idx]["item_weight"]:
                #    all_missing_objects.append(i_idx)
                
        
            for rob in nearby_other_agents:
            
                if not self.other_agents[rob].assignment:
                
                    robot_index_to_key = list(info['robot_key_to_index'].keys())
                    robot_id = robot_index_to_key[list(info['robot_key_to_index'].values()).index(rob)]
                        
                    for m_idx in reversed(range(len(missing_objects))):
                        
                        i_idx = missing_objects[m_idx]
                        
                        if robotState.items[i_idx]["item_weight"] == 1: 
                        
                            object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(i_idx)]
                            self.other_agents[rob].assignment = str(object_id)
                            self.message_text += self.MessagePattern.order_collect(robot_id,object_id)
                            
                            del missing_objects[m_idx]
                            
                            break
                            
                        elif robotState.items[i_idx]["item_weight"] <= len(agents_not_busy):
                            
                            agents_needed = 0
                            
                            object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(i_idx)]
                            
                            
                            other_robot_ids = []
                            
                            agent_idx = agents_not_busy.index(rob)
                            self.other_agents[agents_not_busy[agent_idx]].assignment = str(object_id)
                            del agents_not_busy[agent_idx]

                            for agent_idx in reversed(range(robotState.items[i_idx]["item_weight"]-1)):
                                other_robot_id = robot_index_to_key[list(info['robot_key_to_index'].values()).index(agents_not_busy[agent_idx])]
                                self.other_agents[agents_not_busy[agent_idx]].assignment = str(object_id)
                                del agents_not_busy[agent_idx]
                                
                                other_robot_ids.append(other_robot_id)
                                
                                
                            self.message_text += self.MessagePattern.order_collect_group(robot_id, other_robot_ids,object_id)
                            
                            del missing_objects[m_idx]
                            
                            break
                                    
             
            print([rob2.assignment for rob2 in self.other_agents])
            if all(1 if not rob2.assignment else 0 for rob2 in self.other_agents):
                self.action_index = self.State.end_meeting.value
                self.planning = "equal" #Ends role as coordinator
                self.role = "lifter"
                
            action = Action.get_occupancy_map.value                    
                                
        
        
        elif self.action_index == self.State.wait_random.value:
                

            other_robot_location = robotState.robots[self.wait_requester]["neighbor_location"]
            
            if self.compute_real_distance(other_robot_location,[ego_location[0][0],ego_location[1][0]]) >= self.env.map_config['communication_distance_limit'] or time.time() - self.asked_time > self.help_time_limit: #Until the other robot is out of range we can move
                self.action_index = self.last_action_index
            
            if self.pending_location and self.pending_location != [ego_location[0][0],ego_location[1][0]]:
                action = LLMControl.position_to_action([ego_location[0][0],ego_location[1][0]],self.pending_location,False)
            else:
                action = Action.get_occupancy_map.value    
                self.pending_location = []
            
            
        elif self.action_index == self.State.wait_free.value: 
                    
            for loc_wait_idx in reversed(range(len(self.wait_locations))): #Wait for robots to move from location
                loc_wait = self.wait_locations[loc_wait_idx]
                if occMap[loc_wait[0],loc_wait[1]] == 0:
                    del self.wait_locations[loc_wait_idx]
            print(time.time() - self.asked_time)
            if not self.wait_locations or time.time() - self.asked_time > self.wait_time_limit:
                self.action_index = self.last_action_index
                self.wait_locations = []
                print("Last action", self.last_action_index)
            else:
                action = Action.get_occupancy_map.value
                
                
        try:
            print(action)
        except:
            pdb.set_trace()                    
                                
        return action            
        
    
    def return_to_meeting_point(self, occMap, robotState, info, ego_location):
    
        self.action_index = self.State.end_meeting.value
                            
        true_ending_locations = [loc for loc in self.ending_locations if occMap[loc[0],loc[1]] == 0]
        
        if self.target_location not in self.ending_locations:
            self.target_location = random.choice(true_ending_locations)  
        elif [ego_location[0][0],ego_location[1][0]] in self.ending_locations: #If we are already in the ending locations just stay there
            self.target_location = [ego_location[0][0],ego_location[1][0]]
        elif occMap[self.target_location[0],self.target_location[1]] == 3 or occMap[self.target_location[0],self.target_location[1]] == 2 or occMap[self.target_location[0],self.target_location[1]] == 1:
            self.target_location = random.choice(true_ending_locations)
        
        action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
        
        if self.role == "scout":
            self.item_index = 0
            
        if not action and isinstance(action, list):
            action = Action.get_occupancy_map.value
            
        return action
    
    def planner_sensing(self, robotState, reward, step_count, done, next_observation, info, received_messages):
    
        occMap = np.copy(robotState.latest_map)
        
        action = ""
        item = 0
        message = ''
        robot = 0
        num_neighbors = 0
        
        
        ego_location = np.where(occMap == 5)
        

        self.modify_occMap(robotState, occMap, ego_location, info)

        
        
        
        #Get number of neighboring robots at communication range
        #for nl_idx in range(len(neighbors_location[0])):

        #    if self.compute_real_distance([neighbors_location[0][nl_idx],neighbors_location[1][nl_idx]],[ego_location[0][0],ego_location[1][0]]) < self.env.map_config['communication_distance_limit']:
            
            
        #        num_neighbors += 1
                
                
        
        nearby_other_agents = self.get_neighboring_agents(robotState, ego_location)
        num_neighbors = len(nearby_other_agents)
        
        if not self.helping and not self.planning == "coordinator" and not self.planning == "coordinated": #not helping another robot

            self.carry_heavy_object(robotState, ego_location, nearby_other_agents, info)
                        


        
        if received_messages: #Process received messages
            self.pending_action = self.message_processing(received_messages, robotState, info)
            
            
            #for rm in received_messages:
            #    if self.MessagePattern.move_request(self.robot_id) in rm[1]:
            #        pdb.set_trace() 
            #        break
                                


        self.exchange_sensing_info(robotState, info, nearby_other_agents) #Exchange info about objects sensing measurements
            
            
        if not self.message_text: #if going to send message, skip normal execution of actions
        
        
            if self.planning != "coordinator":
            
                if self.action_index == self.State.get_closest_object.value:
                    print("New sequence")
                    item_locations = np.where(occMap == 2)
                    
                    
                    
                    
                    min_possible_path = float('inf')
                    item_location_idx = -1
                    min_path = []
                    
                    
                    
                    
                    if self.role == "general":
                        heavy_objects_location = [tuple(robotState.items[idx]['item_location']) for idx in self.heavy_objects["index"] if idx in self.sensed_items] #We only exclude objects if they were sensed by this robot
                        non_dangerous_objects_location = [tuple(robotState.items[idx]['item_location']) for idx in self.not_dangerous_objects if idx in self.sensed_items]
                        objects_to_ignore = [self.extended_goal_coords, self.ignore_object, non_dangerous_objects_location, heavy_objects_location]
                    elif self.role == "scout":
                        sensed_objects_location = [tuple(robotState.items[idx]['item_location']) for idx in self.sensed_items]
                        objects_to_ignore = [self.extended_goal_coords, self.ignore_object, sensed_objects_location]
                    elif self.role == "lifter":
                        non_sensed_objects = [tuple(robotState.items[idx]['item_location']) for idx in range(len(robotState.items)) if robotState.items[idx]["item_danger_level"] == 0]
                        non_dangerous_objects_location = [tuple(robotState.items[idx]['item_location']) for idx in self.not_dangerous_objects]
                        heavy_objects_location = [tuple(robotState.items[idx]['item_location']) for idx in self.heavy_objects["index"]]
                        objects_to_ignore = [self.extended_goal_coords, self.ignore_object, non_dangerous_objects_location, heavy_objects_location, non_sensed_objects]
                    
                    
                    if not self.planning == "coordinated" or (self.planning == "coordinated" and self.role != "lifter" and not self.assigned_target_location and not self.just_started):
                    
                        for it_idx in range(len(item_locations[0])): #Check which object in the map to go to
                            loc = (item_locations[0][it_idx],item_locations[1][it_idx])
                            
                            if any(1 if loc in group else 0 for group in objects_to_ignore): 
                                continue
                                
                                
                            try:
                                _,possible_path = self.go_to_location(loc[0],loc[1],occMap,robotState,info,ego_location,checking=True)
                                #possible_path = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([loc[0],loc[1]]),occMap,all_movements=(not robotState.object_held))
                            except:
                                pdb.set_trace()
                            
                            if possible_path:
                                possible_path_len = len(possible_path)
                            
                                if possible_path_len < min_possible_path:
                                    min_possible_path = possible_path_len
                                    min_path = possible_path
                                    item_location_idx = it_idx
                            
                    if item_location_idx >= 0 or self.assigned_target_location: #If there is an object to go to
                        self.stuck_too_much = 0
                        
                        
                        if not self.assigned_target_location:
                            
                            self_location = [ego_location[0][0],ego_location[1][0]]
                        
                            if self_location in self.ending_locations or step_count == 1: #If it's in the meeting location, we should not care about returning there immediately
                                previous_target_location = []
                            else:
                                previous_target_location = self_location
                            
                            self.target_location = [item_locations[0][item_location_idx],item_locations[1][item_location_idx]]
                        else:
                            previous_target_location = []
                            self.target_location = self.assigned_target_location
                            self.assigned_target_location = []
                        

                        
                        
                        
                        action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
                        
                        
                        if self.role == "lifter":
                            already_scanned = True
                        elif self.role == "scout":
                            already_scanned = False
                        else:
                        
                            print(action)
                            already_scanned = False
                            
                            
                            for obj_idx,obj in enumerate(robotState.items): #If we have already scanned it, no need to do it again
                                if obj_idx in self.sensed_items and obj["item_location"][0] == self.target_location[0] and obj["item_location"][1] == self.target_location[1]:
                                    self.action_index = self.State.move_and_pickup.value
                                    self.target_object_idx = obj_idx
                                    already_scanned = True
                                    
                                    print("Scanned", robotState.items[obj_idx],heavy_objects_location, self.heavy_objects)
      
                                    
                                    break


                        if not already_scanned:
                            if not action and isinstance(action, list): #If already near target, start sensing
                                action = Action.danger_sensing.value
                                print("Object:", self.target_location, robotState.items)
                                self.target_location = []
                                self.action_index = self.State.init_check_items.value 
                            elif self.role == "scout" and (previous_target_location and np.linalg.norm(np.array(previous_target_location) - np.array(self.target_location)) > self.room_distance): #Whenever it finishes sensing a room return to meeting location. CHECK THIS!!!
                                action = self.return_to_meeting_point(occMap, robotState, info, ego_location)
                            else:
                                self.action_index = self.State.sense_area.value
                        else:
                            if not action and isinstance(action, list) : #If already near target

                                wait_for_others,_ = self.wait_for_others_func(occMap, info, robotState, nearby_other_agents, [], ego_location)
                                self.being_helped_locations = []
                                if not wait_for_others:
                                    action = self.pick_up(occMap, self.target_location, ego_location)
                                    self.action_index = self.State.pickup_and_move_to_goal.value
                                    
                                    if action < 0:
                                        action = Action.get_occupancy_map.value
          
                                else:
                                    action = Action.get_occupancy_map.value
                                    
                                self.asked_time = time.time()
                                #ego_location = np.where(occMap == 5)
                                #self.past_location = [ego_location[0][0],ego_location[1][0]]
                                self.retries = 0
                                
                            
                        self.past_location = [ego_location[0][0],ego_location[1][0]]       
                        
                           
        
                    else: #No objective found

                        no_more_objects = True
                        for obj_idx,obj in enumerate(robotState.items): #If objects have not been scanned or objects of weight 1 have not been carried
                            
                            if tuple(obj["item_location"]) not in self.extended_goal_coords:
                                if obj_idx not in self.sensed_items and self.role != "lifter":
                                    print("Object missing scan", obj)
                                    no_more_objects = False
                                    break
                                elif obj["item_weight"] == 1 and obj_idx not in self.not_dangerous_objects and self.role != "scout" and not (obj_idx in self.ignore_object and self.role == "lifter"):
                                    no_more_objects = False
                                    print("Object missing", obj)
                                    break
                            
                                    
                        if no_more_objects or self.stuck_too_much >= 100 or self.planning == "coordinated":

                            print("FINISHED")

                            action = self.return_to_meeting_point(occMap, robotState, info, ego_location)
                            
                            
                        else:
                            self.stuck_too_much += 1
                            
                            #if self.stuck_too_much == 100:
                            #    pdb.set_trace()
                            
                            
                        
                            action = Action.get_occupancy_map.value
                           
                elif self.action_index == self.State.sense_area.value:
                
                    
                    action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
                    

                    if self.stuck_moving > 10:
                        self.stuck_moving = 0
                        self.action_index = self.State.get_closest_object.value
                        self.ignore_object.append(self.target_location)
                        print("Getting stuck moving!", self.ignore_object)
                    else:
                        self.ignore_object = []
                
                    if occMap[self.target_location[0],self.target_location[1]] == 0: #The package was taken or something happened
                        self.action_index = self.State.get_closest_object.value

                 
                    if not action and isinstance(action, list):
                        action = Action.danger_sensing.value
                        self.target_location = []
                        self.action_index = self.State.init_check_items.value

                       
                        
                elif self.action_index == self.State.init_check_items.value:
                    self.ignore_object = []
                    self.item_index = 0
                    self.target_location = []
                    action,item = self.process_sensor(robotState, next_observation)
                    if action < 0: #No new sensing measurements

                        self.action_index = self.State.get_closest_object.value
                        action = Action.get_occupancy_map.value
                    else:
                        self.action_index = self.State.check_items.value

                elif self.action_index == self.State.check_items.value:
                
                    
                    if robotState.items[self.item_index-1]['item_danger_level'] > 0: #Consider only those objects we obtain measurements from
                        print(robotState.items[self.item_index-1])

                        
                        if robotState.items[self.item_index-1]['item_danger_level'] == 1: #If not dangerous
                            if self.item_index-1 not in self.sensed_items:
                                self.not_dangerous_objects.append(self.item_index-1) #tuple(robotState.items[self.item_index-1]['item_location']))
                        elif robotState.items[self.item_index-1]['item_weight'] == 1 and tuple(robotState.items[self.item_index-1]['item_location']) not in self.extended_goal_coords: #If dangerous and their weight is 1
                            self.target_location = robotState.items[self.item_index-1]['item_location']
                            self.target_object_idx = self.item_index-1
                        else: #If dangerous and heavy
                            if self.item_index-1 not in self.sensed_items and self.item_index-1 not in self.heavy_objects["index"]:
                                #self.heavy_objects["location"].append(tuple(robotState.items[self.item_index-1]['item_location']))
                                self.heavy_objects["index"].append(self.item_index-1)
                                self.heavy_objects["weight"].append(robotState.items[self.item_index-1]['item_weight'])

                        if self.item_index-1 not in self.sensed_items: #Create a list of sensed objects
                            self.sensed_items.append(self.item_index-1)

                            object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.item_index-1)]
                            
                            #self.message_text += self.MessagePattern.item(robotState.items,self.item_index-1,object_id, self.env.convert_to_real_coordinates)
         

                
                    action,item = self.process_sensor(robotState, next_observation)
                    if action < 0: #finished processing sensor measurements
                    
                        if not self.target_location or self.role == "scout": #in case there is no object sensed
                            self.action_index = self.State.get_closest_object.value
                            action = Action.get_occupancy_map.value
                            
                            if self.role != "scout":
                                self.too_stuck += 1
                                if self.too_stuck > 100:
                                    print("Too stuck")
                                    #pdb.set_trace()
                            

                        else: #move towards object location
                            self.too_stuck = 0
                            print(self.target_location)
                            action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
                            self.action_index = self.State.move_and_pickup.value
                            
                            self.message_text += self.MessagePattern.sensing_help(str(list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.target_object_idx)]))
                            
                            if not action and isinstance(action, list): #If already next to object, try to pick it up
                            
                                wait_for_others,_ = self.wait_for_others_func(occMap, info, robotState, nearby_other_agents,[], ego_location)
                                self.being_helped_locations = []
                                
                                if not wait_for_others:
                                    action = self.pick_up(occMap, self.target_location, ego_location)
                                    #ego_location = np.where(occMap == 5)
                                    if action < 0:
                                        action = LLMControl.position_to_action([ego_location[0][0],ego_location[1][0]],self.past_location,False) 
                                    #self.past_location = [ego_location[0][0],ego_location[1][0]]
                                    self.action_index = self.State.pickup_and_move_to_goal.value
                                    self.retries = 0
                                else:
                                   action = Action.get_occupancy_map.value 
                    else: 
                        self.past_location = [ego_location[0][0],ego_location[1][0]]
                        
                        
                        
                elif self.action_index == self.State.move_and_pickup.value: #From here on, only lifter behavior
                    self.ignore_object = []
                    action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
                    print(self.target_location)
                    
                    if occMap[self.target_location[0],self.target_location[1]] == 0 or robotState.items[self.target_object_idx]["item_danger_level"] == 1: #The package was taken or something happened
                        print("Something happened at move_and_pickup")
                        message = ""
                        if self.being_helped:
                            message = self.MessagePattern.carry_help_finish()
                        self.cancel_cooperation(message=message)
                 
                    if not action and isinstance(action, list):
                    
                        print("waiting for others!")                    
                        
                            
                        wait_for_others,_ = self.wait_for_others_func(occMap, info, robotState, nearby_other_agents, [], ego_location)
                        
                        if not wait_for_others and not robotState.object_held: #pickup if next to object already
                            action = self.pick_up(occMap, self.target_location, ego_location)
                            if action < 0:
                                action = LLMControl.position_to_action([ego_location[0][0],ego_location[1][0]],self.past_location,False) 
                            
                        else:
                            action = Action.get_occupancy_map.value 
                            self.past_location = [ego_location[0][0],ego_location[1][0]]
                            
                        #self.past_location = [ego_location[0][0],ego_location[1][0]]
                        self.action_index = self.State.pickup_and_move_to_goal.value
                        self.retries = 0
                        self.asked_time = time.time()
                        self.being_helped_locations = []

                    else:
                        self.past_location = [ego_location[0][0],ego_location[1][0]]
                        
                    
                elif self.action_index == self.State.pickup_and_move_to_goal.value:
                    self.ignore_object = []

                    if robotState.object_held:
                        g_coord = []
                        for g_coord in self.goal_coords:
                            if not occMap[g_coord[0],g_coord[1]]:
                                target = g_coord
                                break
                        
                        self.target_location = g_coord
                        
                        action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
                        self.action_index = self.State.drop_object.value
                        
                        self.being_helped_locations = []
                        self.previous_next_loc = []
                        #self.wait_for_others_func(occMap, info, robotState, self.next_loc)

                        self.asked_time = time.time()
                        if not action and isinstance(action, list):
                            #pdb.set_trace()
                            action = self.drop()
                            self.target_location = self.past_location
                            self.action_index = self.State.move_end.value
                        else:
                            action = Action.get_occupancy_map.value #Wait for the next state in order to start moving
                    else:
                        
                        #ego_location = np.where(occMap == 5)
                        action = LLMControl.position_to_action([ego_location[0][0],ego_location[1][0]],self.past_location,False)
                          
                        if action == -1:
                        
                            wait_for_others,combinations_found = self.wait_for_others_func(occMap, info, robotState, nearby_other_agents, [], ego_location)
                            
                            print("Action move and pickup:", action, wait_for_others, self.being_helped_locations) 
                            
                            if not wait_for_others:
                                
                                action = self.pick_up(occMap, self.target_location, ego_location)
                                
                                if self.retries == 3: #If can't pickup object just try with another

                                    self.ignore_object.append(tuple(self.target_location))
                                    
                                    message = ""
                                    if self.being_helped:
                                        message = self.MessagePattern.carry_help_finish()
                                    
                                    action = self.cancel_cooperation(message=message)
                                    
                                self.retries += 1   
                                self.asked_time = time.time()
                                
                                print("Pickup retries:", self.retries)
                            elif not combinations_found: #No way of moving                          
                                action = self.cancel_cooperation(message=self.MessagePattern.carry_help_finish())
                            elif time.time() - self.asked_time > self.help_time_limit2:                           
                                action = self.cancel_cooperation(message=self.MessagePattern.carry_help_complain())
                            else:
                                action = Action.get_occupancy_map.value 
                                
                            
                        

                elif self.action_index == self.State.drop_object.value:
                                
                    if not robotState.object_held:            

                        action = self.cancel_cooperation(message=self.MessagePattern.carry_help_complain())
                        
                    else:
                    
                        for agent_id in self.being_helped: #remove locations with teammates

                            agent_idx = info['robot_key_to_index'][agent_id]
                            other_robot_location = robotState.robots[agent_idx]["neighbor_location"]
                            occMap[other_robot_location[0],other_robot_location[1]] = 3
                                    
                        
                        loop_done = False
                        
                        if not self.previous_next_loc or (self.previous_next_loc and self.previous_next_loc[0].tolist() == [ego_location[0][0],ego_location[1][0]]):
                            action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
                            
                            print("HAPPENING", action, self.next_loc)
                            
                            if not action and isinstance(action, list):
                                loop_done = True
                             
                            if not loop_done and self.next_loc:
                                self.previous_next_loc = [self.next_loc[0]]
                                self.being_helped_locations = []
                            
                                print("PEFIOUVS",self.being_helped_locations, self.next_loc, self.previous_next_loc)
                                
                            
                            
                            
                        
                        if not loop_done:
                            wait_for_others,combinations_found = self.wait_for_others_func(occMap, info, robotState, nearby_other_agents, self.previous_next_loc, ego_location)
                            
                            if not combinations_found: #No way of moving
                                action = self.drop()
                                
                                self.cancel_cooperation(message=self.MessagePattern.carry_help_finish())
                        
                        
                                
                        if loop_done or not wait_for_others: #If carrying heavy objects, wait for others
                            
                            action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
                            
                            if self.next_loc and self.previous_next_loc and not self.previous_next_loc[0].tolist() == self.next_loc[0].tolist(): #location changed
                                self.previous_next_loc = []
                                
                            if occMap[self.target_location[0],self.target_location[1]] == 2: #A package is now there
                                self.action_index = self.State.pickup_and_move_to_goal.value
                                self.being_helped_locations = []

                        
                            if not action and isinstance(action, list): #If already next to drop location
                                action = self.drop()
                                self.target_location = self.past_location
                                self.action_index = self.State.move_end.value
                            else:
                                self.past_location = [ego_location[0][0],ego_location[1][0]]
                                
                            self.asked_time = time.time()
                        elif time.time() - self.asked_time > self.help_time_limit2:
                            action = self.drop()
                            
                            self.cancel_cooperation(message=self.MessagePattern.carry_help_complain())
                        elif action != Action.drop_object.value:
                            action = Action.get_occupancy_map.value
                            print("waiting for others...")
                            
                elif self.action_index == self.State.move_end.value:
                    action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
                
                    if not action and isinstance(action, list):
                        action = Action.get_occupancy_map.value
                        
                        self.action_index = self.State.get_closest_object.value
                        
                        if self.being_helped:
                            self.message_text += self.MessagePattern.carry_help_finish()
                            self.asked_time = time.time()
                        self.being_helped = []
                        self.being_helped_locations = []
                        
                elif self.action_index == self.State.wait_message.value:
                    if time.time() - self.asked_time > self.wait_time_limit:

                        self.asked_help = False                    
                        self.cancel_cooperation(message=self.MessagePattern.carry_help_cancel())
                        self.help_time_limit = random.randrange(self.wait_time_limit,30)
                        print("end of waiting")
                    action = Action.get_occupancy_map.value
                    
                elif self.action_index == self.State.wait_random.value:
                
                    #for rm in received_messages:
                    #    if self.MessagePattern.move_request(self.robot_id) in rm[1]:
                    #        pdb.set_trace() 
                    #        break
                  
                
                    other_robot_location = robotState.robots[self.wait_requester]["neighbor_location"]
                    #if not (self.next_loc and (occMap[self.next_loc[0][0],self.next_loc[0][1]] == 3 or (len(self.next_loc) > 1 and occMap[self.next_loc[1][0],self.next_loc[1][1]] == 3))): #Wait until there is no one in your next location
                    
                    if self.compute_real_distance(other_robot_location,[ego_location[0][0],ego_location[1][0]]) >= self.env.map_config['communication_distance_limit'] or time.time() - self.asked_time > self.help_time_limit: #Until the other robot is out of range we can move
                        self.action_index = self.last_action_index
                    
                    if self.pending_location and self.pending_location != [ego_location[0][0],ego_location[1][0]]:
                        action = LLMControl.position_to_action([ego_location[0][0],ego_location[1][0]],self.pending_location,False)
                    else:
                        action = Action.get_occupancy_map.value    
                        self.pending_location = []
                        
                        
                elif self.action_index == self.State.wait_free.value: 
                    
                    for loc_wait_idx in reversed(range(len(self.wait_locations))): #Wait for robots to move from location
                        loc_wait = self.wait_locations[loc_wait_idx]
                        if occMap[loc_wait[0],loc_wait[1]] == 0:
                            del self.wait_locations[loc_wait_idx]
                    print(time.time() - self.asked_time)
                    if not self.wait_locations or time.time() - self.asked_time > self.wait_time_limit:
                        self.action_index = self.last_action_index
                        self.wait_locations = []
                        print("Last action", self.last_action_index)
                    else:
                        action = Action.get_occupancy_map.value
                            
                    
                    

                    """
                    if self.action_index == self.State.check_neighbors.value:
                        agent_idx = info['robot_key_to_index'][self.helping[0]]
                        action = Action.check_robot.value
                        robot = agent_idx
                        self.action_index += 1
                    """
                elif self.action_index == self.State.follow.value:
                    
                    agent_idx = info['robot_key_to_index'][self.helping[0]]
                    self.target_location = robotState.robots[agent_idx]["neighbor_location"]
                    

                    action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
                    
                    if (not action and isinstance(action, list)) or self.compute_real_distance([self.target_location[0],self.target_location[1]],[ego_location[0][0],ego_location[1][0]]) < self.env.map_config['communication_distance_limit']-1:
                        action = Action.get_occupancy_map.value
                            
                            
                elif self.action_index == self.State.obey.value:
                    print("TARGET LOCATION:", self.target_location)
                    
                    try:
                        agent_idx = info['robot_key_to_index'][self.helping[0]]
                    except:
                        pdb.set_trace()
                    helping_location = robotState.robots[agent_idx]["neighbor_location"]
                    
                    occMap[helping_location[0],helping_location[1]] = 1
                    
                    action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info, ego_location, end=True)
                    if (not action and isinstance(action, list)):
                        action = Action.get_occupancy_map.value
                
                    
                    if action == -1:
                        self.ignore_go_location = []
                        #pdb.set_trace()
                        
                elif self.action_index == self.State.end_meeting.value:
                
                    action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
            
            
                    if [ego_location[0][0],ego_location[1][0]] in self.ending_locations:
                        self.just_started = False
            
                    if not action and isinstance(action, list):
                        
                        if self.role == "scout": #Scout should share information
                            
                            """
                            object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.item_index)]
                            
                            missing_objects = 0
                            while robotState.items[self.item_index]["item_danger_level"] == 0 and self.item_index < len(robotState.items):
                                self.message_text += self.MessagePattern.item(robotState.items,self.item_index,object_id, self.env.convert_to_real_coordinates)
                                
                                if robotState.items[self.item_index]["item_danger_level"] == 0:
                                    missing_objects += 1
                                
                                self.item_index += 1
                                
                                
                            
                            if self.item_index == len(robotState.items):
                                self.item_index = 0
                            """    
                            missing_objects = 0
                            for obj in robotState.items:
                                if obj["item_danger_level"] == 0:
                                    missing_objects += 1
                                    
                            if self.planning != "coordinated":
                                if not missing_objects: #If scout finishes sensing, it can transition to lift                        
                                    self.role = "lifter"
                                
                                self.action_index = self.State.get_closest_object.value
                            else:
                                self.message_text += self.MessagePattern.order_finished()
                                self.action_index = self.State.waiting_order.value
                                
                        elif self.role == "lifter": #If there are objects one can lift
                            if self.planning != "coordinated":
                                
                                for idx in range(len(robotState.items)):
                                    if robotState.items[idx]["item_danger_level"] == 2 and robotState.items[idx]["item_weight"] == 1:
                                        self.action_index = self.State.get_closest_object.value

                                if self.action_index == self.State.end_meeting.value and self.robot_id not in self.finished_robots: #Voluntarily finish
                                    self.message_text += self.MessagePattern.finish()
                                    self.finished_robots.append(self.robot_id)
                            else:
                                self.message_text += self.MessagePattern.order_finished()
                                self.action_index = self.State.waiting_order.value
                                
                        elif self.planning == "coordinated":
                            self.message_text += self.MessagePattern.order_finished()
                            self.action_index = self.State.waiting_order.value
                            
                            
                            

                    
                        action = Action.get_occupancy_map.value
                    print("Finished")
                    
                elif self.action_index == self.State.waiting_order.value:
                    action = Action.get_occupancy_map.value
                    
            
            else:
                action = self.central_planning(robotState, info, occMap, ego_location, nearby_other_agents)
                    
            if num_neighbors: #If there are nearby robots, announce next location and goal
            
            
                if action in [Action.move_up.value, Action.move_down.value, Action.move_left.value, Action.move_right.value, Action.move_up_right.value, Action.move_up_left.value, Action.move_down_right.value, Action.move_down_left]: #If it is going to move
                
                    if not self.next_loc:
                        self.next_loc = [np.array([ego_location[0][0],ego_location[1][0]])]
                    
                    if len(self.next_loc) < 2:
                        self.next_loc.append(self.next_loc[0])
                        
                        
                    if not self.target_location:
                        target_loc = [ego_location[0][0],ego_location[1][0]]
                    else:
                        target_loc = self.target_location
                        
                else: #It stays in a place
                    self.next_loc = [np.array([ego_location[0][0],ego_location[1][0]]), np.array([ego_location[0][0],ego_location[1][0]])]
                    target_loc = [ego_location[0][0],ego_location[1][0]]
                    
                try:
                
                    if self.being_helped:
                        helping = [self.robot_id]
                    else:
                        helping = self.helping
                
                    self.message_text +=  self.MessagePattern.location(target_loc[0],target_loc[1],self.next_loc[0][0],self.next_loc[0][1], self.env.convert_to_real_coordinates, [ego_location[0][0],ego_location[1][0]], robotState.object_held, helping)
                except:
                    pdb.set_trace()
            
            
            if self.message_text: #Send message first before doing action
                

                if re.search(self.MessagePattern.location_regex(),self.message_text):
                    self.message_send_time = info['time']
                    rematch = re.search(self.MessagePattern.location_regex(),self.message_text)
                    target_goal = eval(rematch.group(1))
                    target_loc = eval(rematch.group(2))
                    
                    #pdb.set_trace()
                    if target_goal != target_loc and not (self.previous_message and self.previous_message[0] == target_goal and self.previous_message[1] == target_loc): #Only if there was a change of location do we prioritize this message

                        self.previous_message = [target_goal,target_loc]

                        
                        action,message = self.send_message(self.message_text)
                        self.message_text = ""

                        print("SENDING MESSAGE", info['time'], message)
                        

                
            
        else:
        
            
        
            if re.search(self.MessagePattern.location_regex(),self.message_text):
                self.message_send_time = info['time']
        
            action,message = self.send_message(self.message_text)
            self.message_text = ""
            print("SENDING MESSAGE2", info['time'], message)
            

        """        
        next_state = torch.tensor(np.concatenate((robotState.latest_map.ravel(),np.array([robotState.object_held]))), dtype=torch.float32).unsqueeze(0)
        
        
        if self.last_action >= 0 and action != Action.get_occupancy_map.value:
            self.memory_replay.push(self.state, self.last_action, next_state, reward)
            if step_count % 100:
                self.memory_replay.save_to_disk("memory.json")
        
        self.state = next_state
        
        self.last_action = action
        """
        
        
        if self.action_index != self.State.end_meeting.value and self.robot_id in self.finished_robots: #Voluntarily finish
            self.message_text += self.MessagePattern.finish_reject()
            self.finished_robots.remove(self.robot_id)
        
        if action == -1 or action == "":
            
            action = Action.get_occupancy_map.value
            print("STUCK")
            
            
        
        print("action index:",self.State(self.action_index), "action:", Action(action), ego_location)
                
        if done: # or step_count == self.num_steps:
            action = -1
            
        if not action and isinstance(action, list):
            pdb.set_trace()



        
        return action,item,message,robot,len(self.finished_robots) == self.env.action_space["robot"].n
        
        
        
    
