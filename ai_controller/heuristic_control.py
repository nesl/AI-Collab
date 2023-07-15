from llm_control import LLMControl
from gym_collab.envs.action import Action
import numpy as np
import pdb

class HeuristicControl:

    def __init__(self, goal_coords):
        self.goal_coords = goal_coords
        
        self.action_index = 0
        
    def go_to_location(self,x,y,occMap):
    
        ego_location = np.where(occMap == 5)
        
        path_to_follow = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),occMap)
        
        if not path_to_follow or (x == path_to_follow[0][0] and y == path_to_follow[0][1] and occMap[x,y]):
            action = []
        else:
            next_location = [ego_location[0][0],ego_location[1][0]]
            action = LLMControl.position_to_action(next_location,path_to_follow[0],False)
            
        return action
        
    def drop(self):
        return Action.drop_object.value
        
    def pick_up(self, occMap, item_location):
        
        ego_location = np.where(occMap == 5)
    
        action = LLMControl.position_to_action([ego_location[0][0],ego_location[1][0]],item_location,True)
        
        return action
        
        
    def planner(self, robotState):
    
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
                
                if loc in self.goal_coords: 
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
            
            print(action)
            self.action_index += 1
        
        elif self.action_index == 1:
            action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)
        
            if not action:
                action = self.pick_up(occMap, self.target_location)
                
                self.action_index += 1
                
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
            else:
                action = self.pick_up(occMap, self.target_location)
                
        elif self.action_index == 3:
            action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)

        
            if not action:
                action = self.drop()
                self.target_location = self.past_location
                self.action_index += 1
            else:
                ego_location = np.where(occMap == 5)
                self.past_location = [ego_location[0][0],ego_location[1][0]]
                
        elif self.action_index == 4:
            action = self.go_to_location(self.target_location[0],self.target_location[1],occMap)
        
            if not action:
                action = Action.get_occupancy_map.value
                
                self.action_index = 0
                
        return action
            
