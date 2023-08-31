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

    def __init__(self, goal_coords, num_steps, robot_id, env):
        self.goal_coords = goal_coords

        self.memory_replay = ReplayMemory(10000)
        
        self.num_steps = num_steps
        
        self.robot_id = robot_id
        
        self.env = env
        self.help_time_limit = 20
        
    class State(Enum):
        get_closest_object = 0
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
        
    class Other_Agent:
        
        def __init__(self):
            self.current_location = []
            self.next_location = []
    
        
        
    class MessagePattern:
        @staticmethod
        def location(goal_x,goal_y,next_loc_x,next_loc_y,convert_to_real_coordinates, current_location, carrying):
            real_goal = convert_to_real_coordinates([goal_x,goal_y])
            real_next_location = convert_to_real_coordinates([next_loc_x,next_loc_y])
            real_current_location = convert_to_real_coordinates(current_location)
            output_string = "My goal is (" + str(real_goal[0]) + "," + str(real_goal[1]) + "), I'm moving towards (" + str(real_next_location[0]) + "," + str(real_next_location[1]) + "). My current location is (" + str(real_current_location[0]) + "," + str(real_current_location[1]) + "). "
            
            if carrying:
                output_string += "Carrying object. "
                
            return output_string
            
        @staticmethod
        def location_regex():
            return "My goal is (\(-?\d+\.\d+,-?\d+\.\d+\)), I'm moving towards (\(-?\d+\.\d+,-?\d+\.\d+\)). My current location is (\(-?\d+\.\d+,-?\d+\.\d+\)). (Carrying object)?"
            
        @staticmethod
        def item(items,item_idx,object_id, convert_to_real_coordinates):
    
            """ Example
            Object 1 (weight: 1) Last seen in (5.5,5.5) at 00:57 Status Danger: benign, Prob. Correct: 88.1%
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
            return "Object (\d+) \(weight: (\d+)\) Last seen in (\(-?\d+\.\d+,-?\d+\.\d+\)) at (\d+:\d+)\. Status Danger: (\w+), Prob. Correct: (\d+\.\d+)%"
        
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
        self.helping = []
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
        self.pending_action = -1
        
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
        
    def go_to_location(self,x,y,occMap,robotState,info):
    
        occMap_clean = np.copy(occMap)
        locations_to_test = [[1,0],[0,1],[1,1],[-1,0],[0,-1],[-1,-1],[-1,1],[1,-1]]
        ego_location = np.where(occMap == 5)
        all_movements = not robotState.object_held
        
        try:
            path_to_follow = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),occMap,ignore=self.ignore_go_location, all_movements=all_movements)
        except:
            pdb.set_trace()
        
        
        
        if x == ego_location[0][0] and y == ego_location[1][0]: #In case we are already in the destination
            action = []
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
                
                if path_to_follow and occMap[path_to_follow[0][0],path_to_follow[0][1]] != 0: #Removing robots does work, make other robots move
                    robot_combinations = self.get_combinations(nearby_robots)
                    allowed_robots_blocking = (-1,0)
                    for rc_idx,rc in enumerate(robot_combinations):
                        t_occMap = np.copy(occMap_clean)
                        for robot_loc in rc:
                            t_occMap[robot_loc[0],robot_loc[1]] = 1
                            
                        temp_path_to_follow = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),t_occMap,ignore=self.ignore_go_location, all_movements=all_movements)
                        
                        if temp_path_to_follow and len(rc) >= allowed_robots_blocking[1]:
                            allowed_robots_blocking = (rc_idx,len(rc))
                    
                    order_robots = []
                            
                            
                    robot_index_to_key = list(info['robot_key_to_index'].keys())
                    for rb_idx in self.ignore_robots:
                        rb = robotState.robots[rb_idx]["neighbor_location"]
                        if rb not in robot_combinations[allowed_robots_blocking[0]]:
                            #order_robots.append(rb)
                            for nrobot_idx in range(len(robotState.robots)):
                                if robotState.robots[nrobot_idx]["neighbor_location"] == rb:
                                    robot_id = robot_index_to_key[list(info['robot_key_to_index'].values()).index(nrobot_idx)]
                                    break
                                    
 
                                
                            self.message_text += self.MessagePattern.move_request(robot_id)
                            if not self.action_index == self.State.wait_free.value:
                                self.last_action_index = self.action_index
                            self.action_index = self.State.wait_free.value
                            self.wait_locations.append(rb)
                            self.asked_time = time.time()
                            
                        
                    action = -1
                    
                elif path_to_follow and occMap[path_to_follow[0][0],path_to_follow[0][1]] == 0: #If the next step has no robot move until you are next to a blocking robot
                    action = LLMControl.position_to_action([ego_location[0][0],ego_location[1][0]],path_to_follow[0],False)
                    
                else: #We need to wait
                    action = -1
                    #pdb.set_trace()
                
            else:
        
                action = -1
                print("Couldn't go to", x,y)
            
                self.stuck_retries += 1
            
                if self.stuck_retries >= random.randrange(5,20):
                    self.ignore_go_location = []
                    self.stuck_retries = 0
                
        elif x == path_to_follow[0][0] and y == path_to_follow[0][1] and occMap[x,y]: #Next location is our destination. Actually we never arrive to the destination if there is already something there, we just stay one cell before.
            action = []
            self.stuck_retries = 0
        #elif self.helping and (x == path_to_follow[1][0] and y == path_to_follow[1][1] and occMap[x,y]):
        #    action = []
        else:
            self.stuck_retries = 0
            current_location = [ego_location[0][0],ego_location[1][0]]
            
            if self.previous_go_location and path_to_follow[0][0] == self.previous_go_location[0] and path_to_follow[0][1] == self.previous_go_location[1]: #If it gets stuck at location
                if self.go_retries == 2:

                    self.ignore_go_location.append(path_to_follow[0])
                    path_to_follow = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),occMap, ignore=self.ignore_go_location, all_movements=all_movements)
                    print(path_to_follow, self.ignore_go_location)
                    if not path_to_follow: #stuck
                        action = -1
                        
                    self.go_retries = 0
                else:
                    self.go_retries += 1
            else:
                self.go_retries = 0
                self.ignore_go_location = []
            
            if path_to_follow:
                self.previous_go_location = [path_to_follow[0][0],path_to_follow[0][1]]
                action = LLMControl.position_to_action(current_location,path_to_follow[0],False)
                
            
            
        return action,path_to_follow
        
    def drop(self):
        return Action.drop_object.value
        
    def pick_up(self, occMap, item_location):
        
        ego_location = np.where(occMap == 5)
    
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
        
    
        
    def wait_for_others_func(self,occMap, info, robotState):
    
        wait_for_others = False    
                                
        if self.being_helped:
            ego_location = np.where(occMap == 5)
            neighbors_location = np.where(occMap == 3)
            wait_for_others = True
            number_not_wait = 0

            for rb in self.being_helped:
                agent_idx = info['robot_key_to_index'][rb]
                other_robot_location = robotState.robots[agent_idx]["neighbor_location"]
                if np.linalg.norm(np.array([other_robot_location[0],other_robot_location[1]]) - np.array([ego_location[0][0],ego_location[1][0]]))*self.env.map_config['cell_size'] < self.env.map_config['strength_distance_limit']:
                    number_not_wait += 1


            if number_not_wait >= len(self.being_helped):
                wait_for_others = False
                
        return wait_for_others
        
    def wait_movement(self, agent_idx, agent):
    
        self.message_text = self.MessagePattern.wait(agent) #What happens when someone is carrying object
        if not self.action_index == self.State.wait_random.value:
            self.last_action_index = self.action_index
        self.action_index = self.State.wait_random.value
        self.wait_requester = agent_idx
        self.asked_time = time.time()


    def message_processing(self,received_messages, robotState, info):
    
        action = -1
        last_move_request = ""
    
        for rm in received_messages:
            
            print("Received message:", rm)


            if self.MessagePattern.carry_help_accept(self.robot_id) in rm[1]:
                self.asked_time = time.time()
                
                if self.asked_help:
                    
                    self.being_helped.append(rm[0])
                    
                    if len(self.being_helped)+1 >= self.heavy_objects["weight"][self.chosen_heavy_object]:
                        self.asked_help = False
                        
                        self.target_location = robotState.items[self.heavy_objects["index"][self.chosen_heavy_object]]['item_location']
                        
                        self.action_index = self.State.move_and_pickup.value
                        
                    self.message_text += self.MessagePattern.follow(rm[0],len(self.being_helped)-1)
                else:
                    self.message_text += self.MessagePattern.carry_help_reject(rm[0])
            if re.search(self.MessagePattern.carry_help_regex(),rm[1]): # "I need help" in rm[1]:
                rematch = re.search(self.MessagePattern.carry_help_regex(),rm[1])
                
                if not robotState.object_held and not self.helping and not self.being_helped: # and not self.asked_help:
                    self.message_text += self.MessagePattern.carry_help_accept(rm[0])
                    #self.helping = rm[0]
                    #self.action_index = self.State.check_neighbors.value
                    
                else:
                    self.message_text += self.MessagePattern.carry_help_participant_reject(rm[0])
                    print(not robotState.object_held, not self.helping, not self.being_helped, not self.asked_help)
                    
            if re.search(self.MessagePattern.follow_regex(),rm[1]):
            
                rematch = re.search(self.MessagePattern.follow_regex(),rm[1])
            
                if rematch.group(1) == self.robot_id:
            
                    teammate_number = int(rematch.group(2))
                    
                    self.helping = [rm[0],teammate_number]
                    
                    self.action_index = self.State.follow.value
                
            if rm[1] == self.MessagePattern.carry_help_cancel() or self.MessagePattern.carry_help_reject(self.robot_id) in rm[1] or self.MessagePattern.carry_help_finish() in rm[1] or self.MessagePattern.carry_help_complain() in rm[1]:
            
                if self.helping:
                    self.action_index = self.State.get_closest_object.value
                self.helping = []
                
                
            if self.MessagePattern.carry_help_participant_reject(self.robot_id) in rm[1]:
                #self.asked_help = False
                self.asked_time = time.time()
                
            if re.search(self.MessagePattern.sensing_help_regex(),rm[1]): #"What do you know about object " in rm[1]:
                rematch = re.search(self.MessagePattern.sensing_help_regex(),rm[1])
                object_id = rematch.group(1) #rm[1].strip().split()[-1] 
                object_idx = info['object_key_to_index'][object_id]
                
                self.message_text += self.MessagePattern.item(robotState.items,object_idx,object_id, self.env.convert_to_real_coordinates)
                
                if not self.message_text:
                     self.message_text += self.MessagePattern.sensing_help_negative_response(object_id)
            if re.search(self.MessagePattern.item_regex_partial(),rm[1]):
            
                rematch = re.search(self.MessagePattern.item_regex_full(),rm[1])
                
                full_match = False
                
                if not rematch:
                    rematch = re.search(self.MessagePattern.item_regex_partial(),rm[1])
                else:
                    full_match = True
            
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
                
                
                if full_match:
                    
                    if "benign" in rematch.group(5):
                        danger_level = 1
                    else:
                        danger_level = 2
                
                    item["item_danger_level"] = danger_level
                    item["item_danger_confidence"] = [float(rematch.group(6))/100]
                    print("update estimates", danger_level,item["item_danger_confidence"])
                    
                robotState.update_items(item,object_idx)

                    
            if re.search(self.MessagePattern.location_regex(),rm[1]): #"Going towards location" in rm[1]:
                match_pattern = re.search(self.MessagePattern.location_regex(),rm[1])

                #pdb.set_trace()
                other_target_location = self.env.convert_to_grid_coordinates(eval(match_pattern.group(1)))
                other_next_step = self.env.convert_to_grid_coordinates(eval(match_pattern.group(2)))
                agent_idx = info['robot_key_to_index'][rm[0]]
                
                
                curr_loc = tuple(self.env.convert_to_grid_coordinates(eval(match_pattern.group(3))))
                
                if curr_loc not in self.occupied_locations:
                    self.occupied_locations.append(curr_loc)
                
                if other_next_step == other_target_location: #robot stays there
                    
                    if agent_idx not in self.ignore_robots:
                        self.ignore_robots.append(agent_idx)
                        
                    if self.target_location == other_target_location: #Change destination
                        self.action_index = self.State.get_closest_object.value
                else:
                    if self.target_location == other_target_location and not robotState.object_held and not rm[0] in self.being_helped:
                    
                     
                    
                        if rm[2] <= self.message_send_time: #Message arrive at the same time or previous than this robot sent its message.  
                        
                            if rm[2] == self.message_send_time: #rules to disambiguate are based on alphabetic order
                                
                                if ord(rm[0]) < ord(self.robot_id): #If sender's id appears first than receiver in alphabetic order
                                    self.ignore_object.append(other_target_location)
                                    self.action_index = self.State.get_closest_object.value
                                    
                               
                            else:
                                self.ignore_object.append(other_target_location)
                                self.action_index = self.State.get_closest_object.value
                            
                                
                                
                            
                            if re.search(self.MessagePattern.location_regex(),self.message_text):
                                self.message_text = ""
                                print("changing going location!!!")
                     
                    else: #If we are not going to same destination, just ignore temporarily the other location
                        self.ignore_object.append(other_target_location)
                        
                        
                
                
                    if self.next_loc:

                    
                        try:
                            len(self.next_loc)
                        except:
                            pdb.set_trace()
                
                        if (other_next_step == self.next_loc[0].tolist() or (len(self.next_loc) > 1 and other_next_step == self.next_loc[1].tolist())) and not rm[0] in self.being_helped:
                        
                            if (rm[2] <= self.message_send_time and not match_pattern.group(3) and not robotState.object_held) or (rm[2] <= self.message_send_time and match_pattern.group(3) and robotState.object_held) or (self.helping and rm[0] == self.helping[0]): #Message arrive at the same time or previous than this robot sent its message. Priority given to robots carrying objects, and to robots being helped
                            
                                if rm[2] == self.message_send_time and not (self.helping and rm[0] == self.helping[0]): #If helping the one who send the message, automatically wait
                              
                                    if ord(rm[0]) < ord(self.robot_id): #If sender's id appears first than receiver in alphabetic order
                                        #Move to free location
                                        print(rm[2],self.message_send_time)
                                        self.wait_movement(agent_idx,rm[0])
                                else:
                                    print(rm[2],self.message_send_time)
                                    self.wait_movement(agent_idx,rm[0])

                            elif robotState.object_held and not match_pattern.group(3):
                                pass #Nothing happens here, our robot has priority as it is carrying an object
                            elif match_pattern.group(3) and not robotState.object_held: #we are not carrying an object
                                print(rm[2],self.message_send_time)
                                self.wait_movement(agent_idx,rm[0])
                    
            if self.MessagePattern.explanation_question(self.robot_id) in rm[1]:
                self.message_text += self.MessagePattern.explanation_response(self.action_index)
                
            if self.MessagePattern.wait(self.robot_id) in rm[1]:
                agent_idx = info['robot_key_to_index'][rm[0]]
                #other_robot_location = robotState.robots[agent_idx]["neighbor_location"]
                #self.ignore_robots.append(other_robot_location)
                if agent_idx not in self.ignore_robots:
                    self.ignore_robots.append(agent_idx)
            
            if self.MessagePattern.move_request(self.robot_id) in rm[1]: # and not (last_move_request and last_move_request == rm[0]):
            
                
            
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
                    action = LLMControl.position_to_action([ego_location[0][0],ego_location[1][0]],ego_location2,False) 
                    if not self.action_index == self.State.wait_random.value:
                        self.last_action_index = self.action_index
                    self.action_index = self.State.wait_random.value
                    self.wait_requester = agent_idx
                    self.asked_time = time.time()
                elif maximum_distance_with_robot:
                    #object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.heavy_objects['index'][ho])]
                    robot_index_to_key = list(info['robot_key_to_index'].keys())
                    self.message_text += self.MessagePattern.move_request(robot_index_to_key[list(info['robot_key_to_index'].values()).index(agent_idx)])
                    if not self.action_index == self.State.wait_free.value:
                        self.last_action_index = self.action_index
                    self.action_index = self.State.wait_free.value
                    self.asked_time = time.time()
                    self.wait_locations.append(next_location_with_robot)
                
                
                    
        return action  

    def planner_sensing(self, robotState, reward, step_count, done, next_observation, info, received_messages):
    
        occMap = np.copy(robotState.latest_map)
        
        action = ""
        item = 0
        message = ''
        robot = 0
        num_neighbors = 0
        
        ego_location = np.where(occMap == 5)
        neighbors_location = np.where(occMap == 3)
        

        if robotState.object_held: #Eliminate all carried objects from the occupancy map if robot is carrying object
            carried_objects = np.where(occMap == 4)
            for c_idx in range(len(carried_objects[0])):
                occMap[carried_objects[0][c_idx],carried_objects[1][c_idx]] = 0
            

        
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
        
        
        #Get number of neighboring robots at communication range
        for nl_idx in range(len(neighbors_location[0])):

            if self.compute_real_distance([neighbors_location[0][nl_idx],neighbors_location[1][nl_idx]],[ego_location[0][0],ego_location[1][0]]) < self.env.map_config['communication_distance_limit']:
            
                num_neighbors += 1
        
        
        
        if not self.helping: #not helping another robot

            
            for ho_idx in reversed(range(len(self.heavy_objects["index"]))): #Eliminate heavy objects if they have already been taken care of
                if occMap[robotState.items[self.heavy_objects["index"][ho_idx]]['item_location'][0],robotState.items[self.heavy_objects["index"][ho_idx]]['item_location'][1]] == 0:
                    del self.heavy_objects["index"][ho_idx]
                    del self.heavy_objects["weight"][ho_idx]
                                
                        
            if not self.message_text and self.heavy_objects["index"] and not robotState.object_held and not self.being_helped and not self.asked_help and time.time() - self.asked_time > self.help_time_limit: #Ask for help to move heavy objects 
                
                
                        
                order_heavy_objects_ind = np.argsort(self.heavy_objects['weight'])[::-1] #Depening on the number of neighbors, ask for help for a specific object
                        
                for ho in order_heavy_objects_ind:
                        
                    if self.heavy_objects['weight'][ho] <= num_neighbors+1:
                        object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.heavy_objects['index'][ho])]
                        self.message_text += self.MessagePattern.carry_help(object_id,self.heavy_objects['weight'][ho]-1)
                        self.asked_help = True
                        self.asked_time = time.time()
                        self.action_index = self.State.wait_message.value
                        self.chosen_heavy_object = ho
                        break


        
        if received_messages: #Process received messages
            self.pending_action = self.message_processing(received_messages, robotState, info)
            
            
            #for rm in received_messages:
            #    if self.MessagePattern.move_request(self.robot_id) in rm[1]:
            #        pdb.set_trace() 
            #        break
                                

            
            
        if not self.message_text: #if going to send message, skip normal execution of actions
        
        
            if self.action_index == self.State.get_closest_object.value:
                print("New sequence")
                item_locations = np.where(occMap == 2)
                
                ego_location = np.where(occMap == 5)
                
                
                min_possible_path = float('inf')
                item_location_idx = -1
                min_path = []
                
                heavy_objects_location = [tuple(robotState.items[idx]['item_location']) for idx in self.heavy_objects["index"]]
                
                for it_idx in range(len(item_locations[0])):
                    loc = (item_locations[0][it_idx],item_locations[1][it_idx])
                    
                    if loc in self.goal_coords or loc in self.ignore_object or loc in self.not_dangerous_objects or loc in heavy_objects_location: 
                        continue
                        
                    try:
                        possible_path = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([loc[0],loc[1]]),occMap,all_movements=(not robotState.object_held))
                    except:
                        pdb.set_trace()
                    
                    if possible_path:
                        possible_path_len = len(possible_path)
                    
                        if possible_path_len < min_possible_path:
                            min_possible_path = possible_path_len
                            min_path = possible_path
                            item_location_idx = it_idx
                        
                if item_location_idx >= 0: #If there is an object to go to
                    self.target_location = [item_locations[0][item_location_idx],item_locations[1][item_location_idx]]
                    
                    
                    
                    action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info)
                    
                    print(action)
                    self.action_index = self.State.sense_area.value
                    already_scanned = False
                    
                    
                    for obj_idx,obj in enumerate(robotState.items): #If we have already scanned it, no need to do it again
                        if obj_idx in self.sensed_items and obj["item_location"][0] == self.target_location[0] and obj["item_location"][1] == self.target_location[1]:
                            self.action_index = self.State.move_and_pickup.value
                            already_scanned = True
                            
                            print("Scanned", robotState.items[obj_idx],heavy_objects_location, self.heavy_objects)
                            if tuple(robotState.items[obj_idx]["item_location"]) in heavy_objects_location:
                                pdb.set_trace()
                            #self.message_text = "What do you know about object " + str(list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(obj_idx)])
                            
                            if not action and isinstance(action, list): #If already near target

                                wait_for_others = self.wait_for_others_func(occMap, info, robotState)
                                if not wait_for_others:
                                    action = self.pick_up(occMap, self.target_location)
                                    self.action_index = self.State.pickup_and_move_to_goal.value
                                    if action < 0:
                                        action = Action.get_occupancy_map.value
          
                                else:
                                    action = Action.get_occupancy_map.value
                                    
                                self.asked_time = time.time()
                                ego_location = np.where(occMap == 5)
                                #self.past_location = [ego_location[0][0],ego_location[1][0]]
                                self.retries = 0
                            break
                    
                    
                    self.past_location = [ego_location[0][0],ego_location[1][0]]   
                    if not already_scanned and not action and isinstance(action, list): #If already near target, start sensing
                        action = Action.danger_sensing.value
                        self.target_location = []
                        self.action_index = self.State.init_check_items.value   
                    
                       
    
                else:  
                    action = Action.get_occupancy_map.value
                    print("Finished")
                       
            elif self.action_index == self.State.sense_area.value:
            
                self.ignore_object = []
                action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info)
            
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
                            self.not_dangerous_objects.append(tuple(robotState.items[self.item_index-1]['item_location']))
                    elif robotState.items[self.item_index-1]['item_weight'] == 1: #If dangerous and their weight is 1
                        self.target_location = robotState.items[self.item_index-1]['item_location']
                        self.target_idx = self.item_index-1
                    else: #If dangerous and heavy
                        if self.item_index-1 not in self.sensed_items:
                            #self.heavy_objects["location"].append(tuple(robotState.items[self.item_index-1]['item_location']))
                            self.heavy_objects["index"].append(self.item_index-1)
                            self.heavy_objects["weight"].append(robotState.items[self.item_index-1]['item_weight'])

                    if self.item_index-1 not in self.sensed_items: #Create a list of sensed objects
                        self.sensed_items.append(self.item_index-1)

                        object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.item_index-1)]
                        
                        self.message_text += self.MessagePattern.item(robotState.items,self.item_index-1,object_id, self.env.convert_to_real_coordinates)
                        if not self.message_text:
                            pdb.set_trace()

            
                action,item = self.process_sensor(robotState, next_observation)
                if action < 0: #finished processing sensor measurements
                    if not self.target_location: #in case there is no object sensed
                        self.action_index = self.State.get_closest_object.value
                        action = Action.get_occupancy_map.value

                    else: #move towards object location
                        print(self.target_location)
                        action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info)
                        self.action_index = self.State.move_and_pickup.value
                        
                        self.message_text += self.MessagePattern.sensing_help(str(list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.target_idx)]))
                        
                        if not action and isinstance(action, list): #If already next to object, try to pick it up
                        
                            wait_for_others = self.wait_for_others_func(occMap, info, robotState)
                        
                            
                            if not wait_for_others:
                                action = self.pick_up(occMap, self.target_location)
                                ego_location = np.where(occMap == 5)
                                if action < 0:
                                    action = LLMControl.position_to_action([ego_location[0][0],ego_location[1][0]],self.past_location,False) 
                                #self.past_location = [ego_location[0][0],ego_location[1][0]]
                                self.action_index = self.State.pickup_and_move_to_goal.value
                                self.retries = 0
                            else:
                               action = Action.get_occupancy_map.value 
                else: 
                    self.past_location = [ego_location[0][0],ego_location[1][0]]
            elif self.action_index == self.State.move_and_pickup.value:
                self.ignore_object = []
                action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info)
                print(self.target_location)
                if occMap[self.target_location[0],self.target_location[1]] == 0: #The package was taken or something happened
                    self.action_index = self.State.get_closest_object.value
                    if self.being_helped:
                        self.message_text += self.MessagePattern.carry_help_finish()
                    self.being_helped = []

             
                if not action and isinstance(action, list):
                
                    
                    
                    wait_for_others = self.wait_for_others_func(occMap, info, robotState)
                    
                    if not wait_for_others and not robotState.object_held: #pickup if next to object already
                        action = self.pick_up(occMap, self.target_location)
                        if action < 0:
                            action = LLMControl.position_to_action([ego_location[0][0],ego_location[1][0]],self.past_location,False) 
                          
                    else:
                        action = Action.get_occupancy_map.value 
                        self.past_location = [ego_location[0][0],ego_location[1][0]]
                        
                    #self.past_location = [ego_location[0][0],ego_location[1][0]]
                    self.action_index = self.State.pickup_and_move_to_goal.value
                    self.retries = 0
                    self.asked_time = time.time()
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
                    action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info)
                    self.action_index = self.State.drop_object.value

                    self.asked_time = time.time()
                    if not action and isinstance(action, list):
                        #pdb.set_trace()
                        action = self.drop()
                        self.target_location = self.past_location
                        self.action_index = self.State.move_end.value
                else:
                    
                    ego_location = np.where(occMap == 5)
                    action = LLMControl.position_to_action([ego_location[0][0],ego_location[1][0]],self.past_location,False)   
                    if action == -1:
                    
                        wait_for_others = self.wait_for_others_func(occMap, info, robotState)
                    
                        
                        if not wait_for_others:
                            
                            action = self.pick_up(occMap, self.target_location)
                            self.retries += 1
                            if self.retries == 3: #If can't pickup object just try with another
                                self.action_index = self.State.get_closest_object.value
                                self.ignore_object.append(tuple(self.target_location))
                                
                                if self.being_helped:
                                    self.message_text += self.MessagePattern.carry_help_finish()
                                self.being_helped = []
                            self.asked_time = time.time()
                        elif time.time() - self.asked_time > self.help_time_limit:
                            self.action_index = self.State.get_closest_object.value
                            self.message_text += self.MessagePattern.carry_help_complain() #"Thanks for nothing. I don't need your help."
                            action = Action.get_occupancy_map.value
                            self.being_helped = []
                        else:
                            action = Action.get_occupancy_map.value 
                            

                    

            elif self.action_index == self.State.drop_object.value:
                            
                wait_for_others = self.wait_for_others_func(occMap, info, robotState)
                
                
                        
                if not wait_for_others: #If carrying heavy objects, wait for others
                    action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info)

                    if occMap[self.target_location[0],self.target_location[1]] == 2: #A package is now there
                        self.action_index = self.State.pickup_and_move_to_goal.value

                
                    if not action and isinstance(action, list): #If already next to drop location
                        action = self.drop()
                        self.target_location = self.past_location
                        self.action_index = self.State.move_end.value
                    else:
                        ego_location = np.where(occMap == 5)
                        self.past_location = [ego_location[0][0],ego_location[1][0]]
                        
                    self.asked_time = time.time()
                elif time.time() - self.asked_time > self.help_time_limit:
                    action = self.drop()
                    self.action_index = self.State.get_closest_object.value
                    self.message_text += self.MessagePattern.carry_help_complain() #"Thanks for nothing. I don't need your help."
                    self.being_helped = []
                else:
                    action = Action.get_occupancy_map.value
                    print("waiting for others...")
                    
            elif self.action_index == self.State.move_end.value:
                action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info)
            
                if not action and isinstance(action, list):
                    action = Action.get_occupancy_map.value
                    
                    self.action_index = self.State.get_closest_object.value
                    
                    if self.being_helped:
                        self.message_text += self.MessagePattern.carry_help_finish()
                    self.being_helped = []
                    
            elif self.action_index == self.State.wait_message.value:
                if time.time() - self.asked_time > self.help_time_limit:
                    self.action_index = self.State.get_closest_object.value
                    self.asked_time = time.time()
                    self.asked_help = False
                    self.message_text += self.MessagePattern.carry_help_cancel()
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
                if self.pending_action == -1:
                    action = Action.get_occupancy_map.value
                else:
                    action = self.pending_action
                    self.pending_action = -1
                    
            elif self.action_index == self.State.wait_free.value: 
                
                for loc_wait_idx in reversed(range(len(self.wait_locations))):
                    loc_wait = self.wait_locations[loc_wait_idx]
                    if occMap[loc_wait[0],loc_wait[1]] == 0:
                        del self.wait_locations[loc_wait_idx]
                print(time.time() - self.asked_time)
                if not self.wait_locations or time.time() - self.asked_time > self.help_time_limit:
                    self.action_index = self.last_action_index
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
                
                
                map_side = np.array(self.target_location)-occMap.shape[0]/2
                
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
                found_free_location = False
                
                for p_idx,p in enumerate(possible_locations):
                    next_location[0] = self.target_location[0] + p[0]
                    next_location[1] = self.target_location[1] + p[1]
                    if occMap[next_location[0],next_location[1]] != 0 and not occMap[next_location[0],next_location[1]] == 3:
                        offset_location += 1
                    elif p_idx == self.helping[1] + offset_location:
                        self.target_location = next_location
                        found_free_location = True
                        break
                
                action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info)
                if (not action and isinstance(action, list)) or not found_free_location:
                    self.action_index = self.State.follow.value
                    action = Action.get_occupancy_map.value
                        
                        
            
            
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
                    self.message_text +=  self.MessagePattern.location(target_loc[0],target_loc[1],self.next_loc[0][0],self.next_loc[0][1], self.env.convert_to_real_coordinates, [ego_location[0][0],ego_location[1][0]], robotState.object_held)
                except:
                    pdb.set_trace()
            
            
            if self.message_text: #Send message first before doing action
                

                if re.search(self.MessagePattern.location_regex(),self.message_text):
                    self.message_send_time = info['time']
                    rematch = re.search(self.MessagePattern.location_regex(),self.message_text)
                    target_goal = eval(rematch.group(1))
                    target_loc = eval(rematch.group(2))
                    
                    #pdb.set_trace()
                    if target_goal != target_loc and not (self.previous_message and self.previous_message[0] == target_goal and self.previous_message[1] == target_loc):

                        self.previous_message = [target_goal,target_loc]

                        
                        action,message = self.send_message(self.message_text)
                        self.message_text = ""
                        print("SENDING MESSAGE", message)

                
            
        else:
        
            
        
            if re.search(self.MessagePattern.location_regex(),self.message_text):
                self.message_send_time = info['time']
        
            action,message = self.send_message(self.message_text)
            self.message_text = ""
            print("SENDING MESSAGE", message)
            

        """        
        next_state = torch.tensor(np.concatenate((robotState.latest_map.ravel(),np.array([robotState.object_held]))), dtype=torch.float32).unsqueeze(0)
        
        
        if self.last_action >= 0 and action != Action.get_occupancy_map.value:
            self.memory_replay.push(self.state, self.last_action, next_state, reward)
            if step_count % 100:
                self.memory_replay.save_to_disk("memory.json")
        
        self.state = next_state
        
        self.last_action = action
        """
        
        if action == -1 or action == "":
            action = Action.get_occupancy_map.value
            print("STUCK")
            #pdb.set_trace()
        
        print("action index:",self.State(self.action_index), "action:", Action(action), ego_location)
                
        if done or step_count == self.num_steps:
            action = -1
            
        if not action and isinstance(action, list):
            pdb.set_trace()



        
        return action,item,message,robot
        
        
        
    def planner(self, robotState, reward, step_count, done):
    
        occMap = robotState.latest_map
        
        action = ""
        
        if self.action_index == 0:
            print("New sequence")
            item_locations = np.where(occMap == 2)
            
            ego_location = np.where(occMap == 5)
            
            
            min_possible_path = float('inf')
            item_location_idx = -1
            min_path = []
            
            for it_idx in range(len(item_locations[0])):
                loc = (item_locations[0][it_idx],item_locations[1][it_idx])
                
                if loc in self.goal_coords or loc in self.ignore_object: 
                    continue
                    

                possible_path = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([loc[0],loc[1]]),occMap)
                
                if possible_path:
                    possible_path_len = len(possible_path)
                
                    if possible_path_len < min_possible_path:
                        min_possible_path = possible_path_len
                        min_path = possible_path
                        item_location_idx = it_idx
                    
            self.target_location = [item_locations[0][item_location_idx],item_locations[1][item_location_idx]]
            
            action = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info)
            self.ignore_object = []
            print(action)
            self.action_index += 1
            
            if not action and isinstance(action, list):
                action = self.pick_up(occMap, self.target_location)
                ego_location = np.where(occMap == 5)
                self.past_location = [ego_location[0][0],ego_location[1][0]]
                self.action_index += 1
                self.retries = 0
        
        elif self.action_index == 1:
            action = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info)
        
            if occMap[self.target_location[0],self.target_location[1]] == 0: #The package was taken or something happened
                self.action_index = 0

         
            if not action and isinstance(action, list):
                action = self.pick_up(occMap, self.target_location)
                ego_location = np.where(occMap == 5)
                self.past_location = [ego_location[0][0],ego_location[1][0]]
                self.action_index += 1
                self.retries = 0
                
                
        elif self.action_index == 2:
            if robotState.object_held:
                g_coord = []
                for g_coord in self.goal_coords:
                    if not occMap[g_coord[0],g_coord[1]]:
                        target = g_coord
                        break
                
                self.target_location = g_coord
                action = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info)
                self.action_index += 1
                if not action and isinstance(action, list):
                    #pdb.set_trace()
                    action = self.drop()
                    self.target_location = self.past_location
                    self.action_index += 1
            else:
                
                ego_location = np.where(occMap == 5)
                action = LLMControl.position_to_action([ego_location[0][0],ego_location[1][0]],self.past_location,False)   
                if action == -1:
                    action = self.pick_up(occMap, self.target_location)
                    self.retries += 1
                    if self.retries == 3: #If can't pickup object just try with another
                        self.action_index = 0
                        self.ignore_object.append(tuple(self.target_location))

                

        elif self.action_index == 3:
                        
            action = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info)

            if occMap[self.target_location[0],self.target_location[1]] == 2: #A package is now there
                self.action_index = 2

        
            if not action and isinstance(action, list):
                action = self.drop()
                self.target_location = self.past_location
                self.action_index += 1
            else:
                ego_location = np.where(occMap == 5)
                self.past_location = [ego_location[0][0],ego_location[1][0]]
                
        elif self.action_index == 4:
            action = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info)
        
            if not action and isinstance(action, list):
                action = Action.get_occupancy_map.value
                
                self.action_index = 0
                
                
        next_state = torch.tensor(np.concatenate((robotState.latest_map.ravel(),np.array([robotState.object_held]))), dtype=torch.float32).unsqueeze(0)
        
        if self.last_action >= 0 and action != Action.get_occupancy_map.value:
            self.memory_replay.push(self.state, self.last_action, next_state, reward)
            if step_count % 100:
                self.memory_replay.save_to_disk("memory.json")
        
        self.state = next_state
        
        self.last_action = action
        
        if done or step_count == self.num_steps:
            return -1
            
        if not action and isinstance(action, list):
            pdb.set_trace()
        
        print(self.action_index)
        
        return action
            
