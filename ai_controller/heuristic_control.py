from llm_control import LLMControl
from gym_collab.envs.action import Action
import numpy as np
from deepq_control import ReplayMemory
import torch
import pdb

class HeuristicControl:

    def __init__(self, goal_coords, num_steps):
        self.goal_coords = goal_coords

        self.memory_replay = ReplayMemory(10000)
        
        self.num_steps = num_steps
        
        
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
        self.heavy_objects = []
        
        
    def go_to_location(self,x,y,occMap):
    
        ego_location = np.where(occMap == 5)
        
        path_to_follow = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),occMap,ignore=self.ignore_go_location)
        
        if not path_to_follow or (x == path_to_follow[0][0] and y == path_to_follow[0][1] and occMap[x,y]):
            action = []
        else:
            current_location = [ego_location[0][0],ego_location[1][0]]
            
            if self.previous_go_location and path_to_follow[0][0] == self.previous_go_location[0] and path_to_follow[0][1] == self.previous_go_location[1]: #If it gets stuck at location
                if self.go_retries == 2:
                    path_to_follow = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),occMap, ignore=self.ignore_go_location)
                    if not path_to_follow:
                        action = []
                    self.go_retries = 0
                else:
                    self.go_retries += 1
            else:
                self.go_retries = 0
                self.ignore_go_location = []
            
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
        
    def planner_sensing(self, robotState, reward, step_count, done, next_observation, info):
    
        occMap = robotState.latest_map
        
        action = ""
        item = ""
        message = ''
        
        if not self.message_text:
            if self.action_index == 0:
                print("New sequence")
                item_locations = np.where(occMap == 2)
                
                ego_location = np.where(occMap == 5)
                
                
                min_possible_path = float('inf')
                item_location_idx = -1
                min_path = []
                
                for it_idx in range(len(item_locations[0])):
                    loc = (item_locations[0][it_idx],item_locations[1][it_idx])
                    
                    if loc in self.goal_coords or loc in self.ignore_object or loc in self.not_dangerous_objects or loc in self.heavy_objects: 
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
                already_scanned = False
                
                
                for obj_idx,obj in enumerate(robotState.items): #If we have already scanned it, no need to do it again
                    if obj_idx in self.sensed_items and obj["item_location"][0] == self.target_location[0] and obj["item_location"][1] == self.target_location[1]:
                        self.action_index = 4
                        already_scanned = True
                        
                        if not action and isinstance(action, list): #If already near target
                            action = self.pick_up(occMap, self.target_location)
                            ego_location = np.where(occMap == 5)
                            self.past_location = [ego_location[0][0],ego_location[1][0]]
                            self.action_index += 1
                            self.retries = 0
                            self.message_text = "Picking up object at location (" + str(self.target_location[0]) + ',' + str(self.target_location[1]) + ')'
                        break
                
                
                    
                if not already_scanned and not action and isinstance(action, list): #If already near target
                    action = Action.danger_sensing.value
                    self.action_index += 1    
                
                   
                if not self.message_text:
                    self.message_text = "Going towards location (" + str(self.target_location[0]) + ',' + str(self.target_location[1]) + ')'         
                
                       
            elif self.action_index == 1:
                action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)
            
                if occMap[self.target_location[0],self.target_location[1]] == 0: #The package was taken or something happened
                    self.action_index = 0

             
                if not action and isinstance(action, list):
                    action = Action.danger_sensing.value
                    self.action_index += 1

                    self.message_text = "Sensing area"
                   
                    
            elif self.action_index == 2:
                self.item_index = 0
                self.target_location = []
                action,item = self.process_sensor(robotState, next_observation)
                if action < 0: #No new sensing measurements

                    self.action_index = 0
                    action = Action.get_occupancy_map.value
                else:
                    self.action_index += 1
            elif self.action_index == 3:
            
                
                if robotState.items[self.item_index-1]['item_danger_level'] > 0:
                    print(robotState.items[self.item_index-1])
                    
                    if robotState.items[self.item_index-1]['item_danger_level'] == 1:
                        if self.item_index-1 not in self.sensed_items:
                            self.not_dangerous_objects.append(tuple(robotState.items[self.item_index-1]['item_location']))
                    elif robotState.items[self.item_index-1]['item_weight'] == 1:
                        self.target_location = robotState.items[self.item_index-1]['item_location']
                    else:
                        if self.item_index-1 not in self.sensed_items:
                            self.heavy_objects.append(tuple(robotState.items[self.item_index-1]['item_location']))
                        
                    if self.item_index-1 not in self.sensed_items:
                        self.sensed_items.append(self.item_index-1)

                        object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.item_index-1)]
                        
                        self.message_text = "Object " + str(object_id) + " with weight " + str(robotState.items[self.item_index-1]["item_weight"]) + " is "
                        
                        if robotState.items[self.item_index-1]['item_danger_level'] == 1:
                            self.message_text += "benign "
                        else:
                            self.message_text += "dangerous "
                            
                        self.message_text += "with confidence of " + str(robotState.items[self.item_index-1]["item_danger_confidence"][0]) + "%"
                        
                        #print(self.sensed_items)

            
                action,item = self.process_sensor(robotState, next_observation)
                if action < 0:
                    if not self.target_location: #in case there is no object sensed
                        self.action_index = 0
                        action = Action.get_occupancy_map.value

                    else:
                        print(self.target_location)
                        action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)
                        self.action_index += 1
                        
                        if not action and isinstance(action, list):
                            action = self.pick_up(occMap, self.target_location)
                            ego_location = np.where(occMap == 5)
                            self.past_location = [ego_location[0][0],ego_location[1][0]]
                            self.action_index += 1
                            self.retries = 0
                
            elif self.action_index == 4:
                action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)
                print(self.target_location)
                if occMap[self.target_location[0],self.target_location[1]] == 0: #The package was taken or something happened
                    self.action_index = 0

             
                if not action and isinstance(action, list):
                    action = self.pick_up(occMap, self.target_location)
                    ego_location = np.where(occMap == 5)
                    self.past_location = [ego_location[0][0],ego_location[1][0]]
                    self.action_index += 1
                    self.retries = 0
                    
                    self.message_text = "Picking up object at location (" + str(self.target_location[0]) + ',' + str(self.target_location[1]) + ')'   
                    
                    
            elif self.action_index == 5:
                if robotState.object_held:
                    g_coord = []
                    for g_coord in self.goal_coords:
                        if not occMap[g_coord[0],g_coord[1]]:
                            target = g_coord
                            break
                    
                    self.target_location = g_coord
                    action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)
                    self.action_index += 1
                    self.message_text = "Returning to the middle of the room"
                    if not action and isinstance(action, list):
                        #pdb.set_trace()
                        action = self.drop()
                        self.target_location = self.past_location
                        self.action_index += 1
                else:
                    
                    ego_location = np.where(occMap == 5)
                    action = LLMControl.position_to_action([ego_location[0][0],ego_location[1][0]],self.past_location,False)   
                    if action == -1:
                        self.message_text = "Picking up object at location (" + str(self.target_location[0]) + ',' + str(self.target_location[1]) + ')'
                        action = self.pick_up(occMap, self.target_location)
                        self.retries += 1
                        if self.retries == 3: #If can't pickup object just try with another
                            self.action_index = 0
                            self.ignore_object.append(tuple(self.target_location))

                    

            elif self.action_index == 6:
                            
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
                    
            elif self.action_index == 7:
                action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)
            
                if not action and isinstance(action, list):
                    action = Action.get_occupancy_map.value
                    
                    self.action_index = 0
        else:
            action,message = self.send_message(self.message_text)
            self.message_text = ""
            print("SENDING MESSAGE", message)
                
        next_state = torch.tensor(np.concatenate((robotState.latest_map.ravel(),np.array([robotState.object_held]))), dtype=torch.float32).unsqueeze(0)
        
        if self.last_action >= 0 and action != Action.get_occupancy_map.value:
            self.memory_replay.push(self.state, self.last_action, next_state, reward)
            if step_count % 100:
                self.memory_replay.save_to_disk("memory.json")
        
        self.state = next_state
        
        self.last_action = action
        
        if done or step_count == self.num_steps:
            action = -1
            
        if not action and isinstance(action, list):
            pdb.set_trace()
        
        print("action index:",self.action_index)
        
        return action,item,message
        
        
        
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
            
