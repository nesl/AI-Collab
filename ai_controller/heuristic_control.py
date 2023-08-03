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
        
    def start(self):
    
        self.action_index = 0
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
        self.being_helped = 0
        self.helping = ""
        self.target_location = []
        self.chosen_heavy_object = -1
        self.stuck_retries = 0
        self.message_send_time = float('inf')
        

        
    def go_to_location(self,x,y,occMap):
    
        locations_to_test = [[1,0],[0,1],[1,1],[-1,0],[0,-1],[-1,-1],[-1,1],[1,-1]]
        ego_location = np.where(occMap == 5)
        
        path_to_follow = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),occMap,ignore=self.ignore_go_location)
        
        
        
        if x == ego_location[0][0] and y == ego_location[1][0]:
            action = []
            self.stuck_retries = 0
        elif not path_to_follow:
            action = -1
            print("Couldn't go to", x,y)
            
            self.stuck_retries += 1
            
            if self.stuck_retries >= random.randrange(5,20):
                self.ignore_go_location = []
                self.stuck_retries = 0
                
        elif x == path_to_follow[0][0] and y == path_to_follow[0][1] and occMap[x,y]:
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
                    path_to_follow = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),occMap, ignore=self.ignore_go_location)
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
                
            
            
        return action
        
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
        
    def prepare_item_message(self,items,item_idx,object_id):
    
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
            
            real_location = self.env.convert_to_real_coordinates(item_loc)
        
            message_text = "Object " + str(object_id) + " (weight: " +  str(items[item_idx]["item_weight"]) + ") Last seen in (" + str(real_location[0]) + "," + str(real_location[1]) + ") at " + time_formatted
                                
            if items[item_idx]['item_danger_level'] > 0:
                                
                message_text +=  " Status Danger: "
                if items[item_idx]['item_danger_level'] == 1:
                    message_text += "benign, "
                else:
                    message_text += "dangerous, "
                    
                message_text += "Prob. Correct: " + str(round(items[item_idx]["item_danger_confidence"][0]*100,1)) + "%"
        

        return message_text
        
    def wait_for_others_func(self,occMap):
    
        wait_for_others = False    
                                
        if self.being_helped:
            ego_location = np.where(occMap == 5)
            neighbors_location = np.where(occMap == 3)
            wait_for_others = True
            number_not_wait = 0

            for nl_idx in range(len(neighbors_location[0])):
                if np.linalg.norm(np.array([neighbors_location[0][nl_idx],neighbors_location[1][nl_idx]]) - np.array([ego_location[0][0],ego_location[1][0]]))*self.env.map_config['cell_size'] < self.env.map_config['strength_distance_limit']-1:
                    number_not_wait += 1

            if number_not_wait >= self.being_helped:
                wait_for_others = False
                
        return wait_for_others



    def planner_sensing(self, robotState, reward, step_count, done, next_observation, info, received_messages):
    
        occMap = robotState.latest_map
        
        action = ""
        item = 0
        message = ''
        robot = 0
        
        ego_location = np.where(occMap == 5)
        
        if not self.helping: #not helping another robot
        
            ego_location = np.where(occMap == 5)
            neighbors_location = np.where(occMap == 3)
            
            for ho_idx in reversed(range(len(self.heavy_objects["index"]))): #Eliminate heavy objects if they have already been taken care of
                if occMap[robotState.items[self.heavy_objects["index"][ho_idx]]['item_location'][0],robotState.items[self.heavy_objects["index"][ho_idx]]['item_location'][1]] == 0:
                    del self.heavy_objects["index"][ho_idx]
                    del self.heavy_objects["weight"][ho_idx]
                        
            
            if not self.message_text and self.heavy_objects["index"] and not robotState.object_held and not self.being_helped and not self.asked_help and time.time() - self.asked_time > self.help_time_limit: #Ask for help to move heavy objects 
                num_neighbors = 0
                for nl_idx in range(len(neighbors_location[0])):

                    if np.linalg.norm(np.array([neighbors_location[0][nl_idx],neighbors_location[1][nl_idx]]) - np.array([ego_location[0][0],ego_location[1][0]]))*self.env.map_config['cell_size'] < self.env.map_config['communication_distance_limit']:
                    
                        num_neighbors += 1
                        
                order_heavy_objects_ind = np.argsort(self.heavy_objects['weight'])[::-1] #Depening on the number of neighbors, ask for help for a specific object
                        
                for ho in order_heavy_objects_ind:
                        
                    if self.heavy_objects['weight'][ho] <= num_neighbors+1:
                        self.message_text = "I need help to carry an object with weight " + str(self.heavy_objects["weight"][ho])
                        self.asked_help = True
                        self.asked_time = time.time()
                        self.action_index = self.State.wait_message.value
                        self.chosen_heavy_object = ho
                        break

        
        
        if received_messages: #Process received messages
            for rm in received_messages:
            
                print("Received message:", rm)

                if "I can help you " + str(self.robot_id) in rm[1]:
                    self.asked_time = time.time()
                    
                    if self.asked_help:
                        
                        self.being_helped += 1
                        
                        if self.being_helped+1 >= self.heavy_objects["weight"][self.chosen_heavy_object]:
                            self.asked_help = False
                            
                            self.target_location = robotState.items[self.heavy_objects["index"][self.chosen_heavy_object]]['item_location']
                            
                            self.action_index = self.State.move_and_pickup.value
                            
                        self.message_text = "Thanks, follow me " + str(rm[0])
                    else:
                        self.message_text = "Nevermind " + str(rm[0])
                elif "I need help" in rm[1]:
                    if not robotState.object_held and not self.helping and not self.being_helped: # and not self.asked_help:
                        self.message_text = "I can help you " + str(rm[0])
                        #self.helping = rm[0]
                        #self.action_index = self.State.check_neighbors.value
                        
                    else:
                        self.message_text = "I cannot help you"
                        print(not robotState.object_held, not self.helping, not self.being_helped, not self.asked_help)
                elif "Thanks, follow me " + str(self.robot_id) in rm[1]:
                    self.helping = rm[0]
                    self.action_index = self.State.check_neighbors.value
                elif rm[1] == "Nevermind" or "Nevermind " + str(self.robot_id) in rm[1] or "No need for more help" in rm[1] or "Thanks for nothing" in rm[1]:
                
                    if self.helping:
                        self.action_index = self.State.get_closest_object.value
                    self.helping = ""
                    
                    
                elif "I cannot help you" in rm[1]:
                    #self.asked_help = False
                    self.asked_time = time.time()
                    
                elif "What do you know about object " in rm[1]:
                    object_id = rm[1].strip().split()[-1] 
                    object_idx = info['object_key_to_index'][object_id]
                    
                    self.message_text = self.prepare_item_message(robotState.items,object_idx,object_id)
                    
                    if not self.message_text:
                         self.message_text = "I don't know anything"
                elif re.search("Object \d+ \(weight: \d+\) Last seen in .*",rm[1]):
                    object_id = re.search("Object (\d+)",rm[1]).group(1)
                    object_idx = info['object_key_to_index'][object_id]
                    last_seen_match = re.search("Last seen in (\(\d+\.\d+,\d+\.\d+\))",rm[1])
                    last_time_match = re.search("at (\d+:\d+)",rm[1])
                    
                    item = {}
                    
                    if last_seen_match and last_time_match:
                        last_seen = list(eval(last_seen_match.group(1)))
                        item["item_location"] = self.env.convert_to_grid_coordinates(last_seen)
                        last_time = last_time_match.group(1).split(":")
                        item["item_time"] = [int(last_time[1]) + int(last_time[0])*60]
                        item["item_weight"] = int(re.search("\(weight: (\d+)\)",rm[1]).group(1))
                        item["item_danger_level"] = 0
                        item["item_danger_confidence"] = []
                        danger_level_match = re.search("Status Danger: (\w+)",rm[1])
                        
                        
                        if danger_level_match:
                            
                            if "benign" in danger_level_match.group(1):
                                danger_level = 1
                            else:
                                danger_level = 2
                        
                            item["item_danger_level"] = danger_level
                            item["item_danger_confidence"] = [float(re.search("Prob. Correct: (\d+\.\d+)",rm[1]).group(1))/100]
                            print("update estimates", danger_level,item["item_danger_confidence"])
                            
                        robotState.update_items(item,object_idx)
                        
                elif "Going towards location" in rm[1]:
                    match_pattern = re.search("(\(\d+,\d+\))",rm[1])

                    if match_pattern:
                        other_target_location = eval(match_pattern.group(1))
                        
                        if tuple(self.target_location) == other_target_location:
                        
                            if rm[2] <= self.message_send_time: #Message arrive at the same time or previous than this robot sent its message
                            
                                if rm[2] == self.message_send_time:
                                    
                                    if not self.action_index == self.State.wait_random.value:
                                        self.previous_action_index = self.State.get_closest_object.value
                                        self.target_timesteps = random.randrange(1,10)
                                        self.wait_timesteps = 0
                                        self.action_index = self.State.wait_random.value
                                        self.target_location = []
                                        self.message_send_time = float('inf')
                                else:
                                    self.ignore_object.append(other_target_location)
                                    self.action_index = self.State.get_closest_object.value
                                
                                    
                                    
                                print("changing going location!!!")
                                if "Going towards location" in self.message_text:
                                    self.message_text = ""
                            else:
                                self.message_text = "I'm already going to location (" + str(self.target_location[0]) + ',' + str(self.target_location[1]) + ')'
                        else:
                            self.ignore_object.append(other_target_location)
                                
                elif "I'm already going to location" in rm[1]:
                    match_pattern = re.search("(\(\d+,\d+\))",rm[1])

                    if match_pattern:
                        other_target_location = eval(match_pattern.group(1))
                        
                        if tuple(self.target_location) == other_target_location:
                            self.action_index = self.State.get_closest_object.value
                        
                        self.ignore_object.append(other_target_location)
                            
                    
        
        if not self.message_text: #if going to send message, skip normal execution of actions
            if not self.helping: #normal sequence of actions
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
                            

                        possible_path = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([loc[0],loc[1]]),occMap)
                        
                        if possible_path:
                            possible_path_len = len(possible_path)
                        
                            if possible_path_len < min_possible_path:
                                min_possible_path = possible_path_len
                                min_path = possible_path
                                item_location_idx = it_idx
                            
                    if item_location_idx >= 0:
                        self.target_location = [item_locations[0][item_location_idx],item_locations[1][item_location_idx]]
                        
                        
                        
                        action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)
                        
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

                                    wait_for_others = self.wait_for_others_func(occMap)
                                    if not wait_for_others:
                                        action = self.pick_up(occMap, self.target_location)
                                        self.action_index = self.State.pickup_and_move_to_goal.value
                                        if action < 0:
                                            action = Action.get_occupancy_map.value
                                        self.message_text = "Going towards location (" + str(self.target_location[0]) + ',' + str(self.target_location[1]) + ')'
                                    else:
                                        action = Action.get_occupancy_map.value
                                        
                                    self.asked_time = time.time()
                                    ego_location = np.where(occMap == 5)
                                    #self.past_location = [ego_location[0][0],ego_location[1][0]]
                                    self.retries = 0
                                break
                        
                        
                        self.past_location = [ego_location[0][0],ego_location[1][0]]   
                        if not already_scanned and not action and isinstance(action, list): #If already near target
                            action = Action.danger_sensing.value
                            self.target_location = []
                            self.action_index = self.State.init_check_items.value   
                        
                           
                        if not self.message_text and self.target_location:
                            self.message_text = "Going towards location (" + str(self.target_location[0]) + ',' + str(self.target_location[1]) + ')'      
                        else:
                            print("Already have message", self.message_text) 
                    else:  
                        action = Action.get_occupancy_map.value
                        print("Finished")
                           
                elif self.action_index == self.State.sense_area.value:
                
                    self.ignore_object = []
                    action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)
                
                    if occMap[self.target_location[0],self.target_location[1]] == 0: #The package was taken or something happened
                        self.action_index = self.State.get_closest_object.value

                 
                    if not action and isinstance(action, list):
                        action = Action.danger_sensing.value
                        self.target_location = []
                        self.action_index = self.State.init_check_items.value

                        self.message_text = "Sensing area"
                       
                        
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
                            
                            self.message_text = self.prepare_item_message(robotState.items,self.item_index-1,object_id)
                            if not self.message_text:
                                pdb.set_trace()

                
                    action,item = self.process_sensor(robotState, next_observation)
                    if action < 0: #finished processing sensor measurements
                        if not self.target_location: #in case there is no object sensed
                            self.action_index = self.State.get_closest_object.value
                            action = Action.get_occupancy_map.value

                        else: #move towards object location
                            print(self.target_location)
                            action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)
                            self.action_index = self.State.move_and_pickup.value
                            
                            self.message_text = "What do you know about object " + str(list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.target_idx)])
                            
                            if not action and isinstance(action, list): #If already next to object, try to pick it up
                            
                                wait_for_others = self.wait_for_others_func(occMap)
                            
                                
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
                    action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)
                    print(self.target_location)
                    if occMap[self.target_location[0],self.target_location[1]] == 0: #The package was taken or something happened
                        self.action_index = self.State.get_closest_object.value
                        if self.being_helped:
                            self.message_text = "No need for more help"
                        self.being_helped = 0

                 
                    if not action and isinstance(action, list):
                    
                        
                        
                        wait_for_others = self.wait_for_others_func(occMap)
                        
                        if not wait_for_others and not robotState.object_held: #pickup if next to object already
                            action = self.pick_up(occMap, self.target_location)
                            if action < 0:
                                action = LLMControl.position_to_action([ego_location[0][0],ego_location[1][0]],self.past_location,False) 
                            self.message_text = "Going towards location (" + str(self.target_location[0]) + ',' + str(self.target_location[1]) + ')'   
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
                        action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)
                        self.action_index = self.State.drop_object.value
                        self.message_text = "Returning to the middle of the room"
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
                        
                            wait_for_others = self.wait_for_others_func(occMap)
                        
                            
                            if not wait_for_others:
                                self.message_text = "Going towards location (" + str(self.target_location[0]) + ',' + str(self.target_location[1]) + ')'
                                action = self.pick_up(occMap, self.target_location)
                                self.retries += 1
                                if self.retries == 3: #If can't pickup object just try with another
                                    self.action_index = self.State.get_closest_object.value
                                    self.ignore_object.append(tuple(self.target_location))
                                    
                                    if self.being_helped:
                                        self.message_text = "No need for more help"
                                    self.being_helped = 0
                                self.asked_time = time.time()
                            elif time.time() - self.asked_time > self.help_time_limit:
                                self.action_index = self.State.get_closest_object.value
                                self.message_text = "Thanks for nothing. I don't need your help."
                                action = Action.get_occupancy_map.value
                                self.being_helped = 0
                            else:
                                action = Action.get_occupancy_map.value 
                                

                        

                elif self.action_index == self.State.drop_object.value:
                                
                    wait_for_others = self.wait_for_others_func(occMap)
                    
                    
                            
                    if not wait_for_others: #If carrying heavy objects, wait for others
                        action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)

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
                        self.message_text = "Thanks for nothing. I don't need your help."
                        self.being_helped = 0
                    else:
                        action = Action.get_occupancy_map.value
                        print("waiting for others...")
                        
                elif self.action_index == self.State.move_end.value:
                    action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)
                
                    if not action and isinstance(action, list):
                        action = Action.get_occupancy_map.value
                        
                        self.action_index = self.State.get_closest_object.value
                        
                        if self.being_helped:
                            self.message_text = "No need for more help"
                        self.being_helped = 0
                        
                elif self.action_index == self.State.wait_message.value:
                    if time.time() - self.asked_time > self.help_time_limit:
                        self.action_index = self.State.get_closest_object.value
                        self.asked_time = time.time()
                        self.asked_help = False
                        self.message_text = "Nevermind"
                    action = Action.get_occupancy_map.value
                    
                elif self.action_index == self.State.wait_random.value:
                    if self.wait_timesteps >= self.target_timesteps:
                        self.action_index = self.previous_action_index
                    action = Action.get_occupancy_map.value
                    self.wait_timesteps += 1
                    
                    
            else: #Follow behavior
                
                if self.action_index == self.State.check_neighbors.value:
                    agent_idx = info['robot_key_to_index'][self.helping]
                    action = Action.check_robot.value
                    robot = agent_idx
                    self.action_index += 1
                elif self.action_index == self.State.follow.value:

                    agent_idx = info['robot_key_to_index'][self.helping]
                    self.target_location = robotState.robots[agent_idx]["neighbor_location"]
                    action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)
                    if not action and isinstance(action, list):
                        self.action_index = self.State.check_neighbors.value
                        action = Action.get_occupancy_map.value
        else:
        
            if "Going towards location" in self.message_text:
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
            
            action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)
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
            action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)
        
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
                action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)
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
                        
            action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)

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
            action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)
        
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
            
