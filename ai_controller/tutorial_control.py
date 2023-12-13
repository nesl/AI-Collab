from gym_collab.envs.action import Action
import numpy as np
import pdb
from enum import Enum
import re
from cnl import MessagePattern
from movement import Movement



class TutorialControl:

    def __init__(self, num_steps, robot_id, env):

        
        self.num_steps = num_steps
        
        self.robot_id = robot_id
        
        self.env = env
        
            
            
        self.movement = Movement(env)
        
    class State(Enum):
        wait_message = 0
        follow = 1

        
        
    def start(self, robotState):
    
        self.action_index = self.State.wait_message
        
        self.message_text = ""
        
        
        num_humans = sum([1 if not robot["neighbor_type"] else 0 for robot in robotState.robots])
        
        self.helping = chr(ord(self.robot_id) - num_humans)
        
        self.answered_first = False

        
        

        
        
    def send_message(self,message):
        
        action = Action.send_message.value
        
        
        return action,message
        
    
    
    
    
    def message_processing(self,received_messages, robotState, info):
    
        pattern1 = "Hey, what results did you get for object (\d+)"
        pattern2 = "Yes, can you help me carry that object?"
    
        for rm in received_messages:
            
            print("Received message:", rm)


            if re.search(pattern1,rm[1]):
                rematch = re.search(pattern1,rm[1])
                object_id = rematch.group(1)
                self.message_text = "Object " + object_id + " (weight: 2) Last seen in (5.5,5.5) at 00:01. Status Danger: dangerous, Prob. Correct: 99.1%. What do you have?"
            
                            
            
            if not self.answered_first and (re.search(MessagePattern.item_regex_full(),rm[1]) or re.search(MessagePattern.item_regex_full_alt(),rm[1])):
                self.message_text = "I think it's dangerous!"
                self.answered_first = True

                
                
            if pattern2 in rm[1]:
                self.message_text = "Sure, I'll follow you"
                self.action_index = self.State.follow     
                                              
     
            
    
    
    def planner_sensing(self, robotState, reward, step_count, done, next_observation, info, received_messages):
    
        occMap = np.copy(robotState.latest_map)
        
        action = ""
        item = 0
        message = ''
        robot = 0
        
        ego_location = np.where(occMap == 5)
        
        
        if received_messages: #Process received messages
            self.message_processing(received_messages, robotState, info)
        
        if not self.message_text: #if going to send message, skip normal execution of actions
                if self.action_index == self.State.wait_message:
                
                    action = Action.get_occupancy_map.value
                    
                elif self.action_index == self.State.follow: 

                    agent_idx = info['robot_key_to_index'][self.helping]
                    
                    target_location = robotState.robots[agent_idx]["neighbor_location"]
                    

                    action,_,_,_ = self.movement.go_to_location(target_location[0],target_location[1],occMap,robotState,info,ego_location,-1)
                    
                    
                    real_distance = self.env.compute_real_distance([target_location[0],target_location[1]],[ego_location[0][0],ego_location[1][0]])
                    

                    distance_limit = self.env.map_config["strength_distance_limit"]-1
                    
                    if (not action and isinstance(action, list)) or real_distance < distance_limit:
                        action = Action.get_occupancy_map.value
                
        
        else:
        
        
            action,message = self.send_message(self.message_text)
            self.message_text = ""
            print("SENDING MESSAGE2", info['time'], message)
            


        
        if action == -1 or action == "":
            
            action = Action.get_occupancy_map.value
            print("STUCK")
            
            
        
        print("action index:",self.action_index, "action:", Action(action), ego_location)
                
        if done: # or step_count == self.num_steps:
            action = -1



        
        return action,message,False
        
        
        
    
