import numpy as np
from gym_collab.envs.action import Action
import os
import re
import json
import pdb
from cnl import MessagePattern
from movement import Movement
from enum import Enum
import time
from collections import deque
from itertools import combinations
import pyAgrum as gum
import math
import random
from process_text import Human2AIText

#TODO respond with object location when order is carrying? Make sure task is not ordered again, check cost for carrying. Agree to end between leaders. Distance 0 allow

class DecisionControl:

    def __init__(self, env, robotState, team_structure):
        self.action_retry = 0
        self.next_loc = []
        self.item_list = []
        self.item_list_dup = []
        self.action_sequence = 0
        self.top_action_sequence = 0
        self.held_objects = []
        self.env = env
        self.sample_action_space = env.action_space.sample()
        self.action_function = ""

        self.explore_location = []
        self.previous_message = []
        self.message_text = ""
        self.message_info = [False,"",0]
        
        if not all(robot[1] for robot in env.neighbors_info): #Check if there are human peers
            self.ask_info_time_limit = 28
            self.non_request = 60
        else:
            self.ask_info_time_limit = 10
            self.non_request = 10

        self.other_agents = [self.Other_Agent() for r in range(len(self.env.map_config['all_robots']))]
        
        print(self.other_agents)
        
        self.message_text = ""
        self.ai_message_text = ""
        self.chosen_object_idx = -1
        self.target_location = []
        self.movement = Movement(env)
        self.action_index = self.State.decision_state
        self.previous_next_loc = []
        self.nearby_other_agents = []
        self.disabled_agents = []
        self.help_requests = []
        self.object_of_interest = ""
        self.help_time_limit2 = 30
        #self.sensing_ask_time = 0
        #self.sensing_ask_time_limit = 10
        #self.sense_request_data = []
        #self.sense_request = self.State_Sense_Request.no_request
        #self.sense_request_time = {}
        self.help_request_time = {}
        self.past_decision = ""
        self.helping_type = self.HelpType.carrying
        self.carried_objects = {}
        self.action_history = deque(maxlen=5)
        self.message_history = deque(maxlen=100)
        self.team_structure = team_structure
        self.order_status = self.OrderStatus.finished
        self.order_status_info = []
        self.leaders = []
        self.finished = False
        self.told_to_finish = False
        self.collect_attempts = {}
        self.agent_requesting_order = False
        
        self.leader_id = ""
        
        for tm in self.team_structure["hierarchy"].keys():
            if self.team_structure["hierarchy"][tm] == "order":
                self.leaders.append(tm)
        
        self.extended_goal_coords = env.goal_coords.copy()
        self.extended_goal_coords.extend([(g[0]+op[0],g[1]+op[1]) for g in env.goal_coords for op in [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]] if [g[0]+op[0],g[1]+op[1]] not in env.goal_coords])
        
        self.ending_locations = [[x,y] for x in range(8,13) for y in range(15,19)] #ending locations
        self.ending_locations.remove([12,18]) #To ensure all locations are within communication range
        self.ending_locations.remove([8,18])
        
        self.human_to_ai_text = []
        if not all(robot[1] for robot in env.neighbors_info): #Check if there are human peers    
            self.human_to_ai_text = Human2AIText(env, robotState)
       
    class Other_Agent:
        
        def __init__(self):
            self.my_location = {"ego_location": [], "goal_location": [], "next_location": []} #Location of agent being controlled here
            self.my_carrying = False
            self.my_team = ""
            
            self.other_location = {"ego_location": [], "goal_location": [], "next_location": []} 
            self.team = ""
            self.carrying = False
            self.items = {}
            self.items_info_provided = []
            self.assignment = ""
            self.previous_assignment = ""
            self.observations = deque(maxlen=5)
            self.finished = False
            
    class State(Enum):
        decision_state = 0
        drop_object = 1
        
       
    class HelpType(Enum):
        carrying = 1
        sensing = 2
        
    class OrderStatus(Enum):
        finished = 1
        ongoing = 2
        reporting_output = 3
        reporting_availability = 4
       
    def message_processing(self,received_messages, robotState, info):
    
        time_log = info["time"]
        
        objects_str = {}
    
        tmp_message_history = []
        translated_messages_index = -1
    
        for rm_idx,rm in enumerate(received_messages):
            
            print("Received message:", rm)
            template_match = False
            tmp_message = rm[1]
            
            agent_idx = info['robot_key_to_index'][rm[0]]
            
            if re.search(MessagePattern.carry_help_accept_regex(),rm[1]):
            
                if robotState.get("agents", "type", info['robot_key_to_index'][rm[0]]):
                    template_match = True
                
                rematch = re.search(MessagePattern.carry_help_accept_regex(),rm[1])
                

                
                if rematch.group(1) == str(self.env.robot_id) and self.movement.help_status == self.movement.HelpState.asking or self.movement.help_status == self.movement.HelpState.being_helped:
                
                    template_match = True
                
                    object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.chosen_object_idx)]
                    
                    obs_string = "Offered me help to carry object " + str(object_id)
                    
                    self.other_agents[agent_idx].observations.append(obs_string)
                    
                    following = False
                    if self.helping_type == self.HelpType.carrying:
                        num_agents_needed = robotState.get("objects", "weight", self.chosen_object_idx)
                    elif self.helping_type == self.HelpType.sensing:
                        num_agents_needed = 2
                        following = True
                        
                    return_value,self.message_text,_ = self.movement.message_processing_carry_help_accept(rm, {"weight": num_agents_needed}, self.message_text, following)
                    
                    
                    if return_value == 1:

                        object_location = robotState.get("objects", "last_seen_location", self.chosen_object_idx)
                        if not (object_location[0] == -1 and object_location[1] == -1):
                            self.target_location = object_location
                            self.object_of_interest = object_id
                            #self.target_object_idx = self.heavy_objects["index"][self.chosen_heavy_object]
                            
                            self.action_index = self.State.decision_state
                            
                        else: #Somehow we end here
                            self.message_text += MessagePattern.carry_help_reject(rm[0])
                            
                        



            if re.search(MessagePattern.carry_help_regex(),rm[1]):
            
                rematch = re.search(MessagePattern.carry_help_regex(),rm[1])
                
                template_match = True
                
                #Calculate utility. When to collaborate? -> When to accept or offer collaboration? -> Collaboration score of others and expectation of such collaboration score
                #Finish other strategies
                
        
                              
                self.other_agents[agent_idx].observations.append("Asked me to help carry object " + rematch.group(2))
                
                if self.team_structure["role"][self.env.robot_id] != "sensing": 
                    self.message_text,self.action_index,_ = self.movement.message_processing_carry_help(rm, robotState, self.action_index, self.message_text)
                    
                    if MessagePattern.carry_help_accept(rm[0]) in self.message_text:
                        self.helping_type = self.HelpType.carrying

                    #Make sure carrying action is robust to failure here
                    """                  
                    if re.search(MessagePattern.carry_help_regex(),message_text): #This means the robot is preparing to ask for help and reject the help request, we shouldn't allow this
                        message_text = message_text.replace(re.search(MessagePattern.carry_help_regex(),message_text).group(), "")
                        self.movement.asked_help = False
                        self.movement.asked_time = time.time()
                        action_index = self.last_action_index
                    """
                    """
                    if not robotState.object_held and self.movement.help_status == self.movement.HelpState.no_request and self.sense_request == self.State_Sense_Request.no_request: # accept help request
                        #message_text += MessagePattern.carry_help_accept(rm[0])
                        #self.movement.accepted_help = rm[0]
                        self.help_requests.append(rm[0])

                        #self.helping = rm[0]
                        #self.action_index = self.State.check_neighbors
                        
                    else: #reject help request
                        self.message_text += MessagePattern.carry_help_participant_reject(rm[0])
                        #print("Cannot help", not robotState.object_held, not self.movement.helping, not self.movement.being_helped, not self.movement.accepted_help, not self.movement.asked_help, self.sense_request)
                    """
                
                    """
                    template_match = True
                
                    self.message_text,self.action_index,_ = self.movement.message_processing_carry_help(rm, robotState, self.action_index, self.message_text)
                    """
                
            if re.search(MessagePattern.follow_regex(),rm[1]) or re.search(MessagePattern.following_regex(),rm[1]) or MessagePattern.carry_help_cancel() in rm[1] or MessagePattern.carry_help_reject(self.env.robot_id) in rm[1] or MessagePattern.carry_help_finish() in rm[1] or MessagePattern.carry_help_complain() in rm[1] or re.search(MessagePattern.object_not_found_regex(), rm[1]):
            
                template_match = True
            
                self.action_index,_,message = self.movement.message_processing_help(rm, self.action_index, self.helping_type == self.HelpType.sensing, self.State.decision_state)
                
                self.message_text += message
                
                """   
                if re.search(MessagePattern.follow_regex(),rm[1]):
                    rematch = re.search(MessagePattern.follow_regex(),rm[1])
                
                    if rematch.group(1) == str(self.env.robot_id):
                        self.other_agents[agent_idx].observations.append("Asked me to follow him")
                    else:
                        self.other_agents[agent_idx].observations.append("Asked " + rematch.group(1) + " to follow him")
                """        
                
                follow_match = re.search(MessagePattern.follow_regex(),rm[1])
                following_match = re.search(MessagePattern.following_regex(),rm[1])
                
                if (follow_match and follow_match.group(1) == str(self.env.robot_id)) or (following_match and following_match.group(1) == str(self.env.robot_id)):
                    if self.movement.help_status == self.movement.HelpState.helping:
                        if self.helping_type == self.HelpType.carrying:
                            pass
                                
                        elif self.helping_type == self.HelpType.sensing:
                            
                            if self.movement.help_status == self.movement.HelpState.helping:
                                self.action_sequence = 0
                                self.top_action_sequence = 0
                                
                                object_id = self.movement.help_status_info[4]
                                grid_location = self.movement.help_status_info[5]
                                
                                self.create_action_function("sense_by_request('" + object_id + "','" + rm[0] + "'," + str(grid_location) + ")")
                                
                    elif self.movement.help_status == self.movement.HelpState.no_request:
                        if (follow_match and follow_match.group(1) == str(self.env.robot_id)):
                            self.message_text += "Why do you want me to follow you? "
                        elif (following_match and following_match.group(1) == str(self.env.robot_id)):
                            self.message_text += "Why do you want to follow me? "
                        
                        
                elif MessagePattern.carry_help_cancel() in rm[1]:
                    self.other_agents[agent_idx].observations.append("Cancelled his request for help")
                    
                elif MessagePattern.carry_help_finish() in rm[1]:
                    self.other_agents[agent_idx].observations.append("Finished moving heavy object with help from others")
                
                elif MessagePattern.carry_help_complain() in rm[1]:
                    self.other_agents[agent_idx].observations.append("Dismissed his team for not collaborating effectively")
                    
                
                if MessagePattern.carry_help_cancel() in rm[1] or MessagePattern.carry_help_reject(self.env.robot_id) in rm[1] or MessagePattern.carry_help_finish() in rm[1] or MessagePattern.carry_help_complain() in rm[1]:
                    self.message_text += "Ok " + rm[0] + ". I won't help anymore. " 
                elif not re.search(MessagePattern.follow_regex(),rm[1]) and not re.search(MessagePattern.following_regex(),rm[1]):
                    self.message_text += "Ok " + rm[0] + ". "     
                    
            if re.search(MessagePattern.carry_help_reject_regex(),rm[1]):
                
                template_match = True
                
                rematch = re.search(MessagePattern.carry_help_reject_regex(),rm[1])
                
                if rematch.group(1) == str(self.env.robot_id):
                    self.other_agents[agent_idx].observations.append("Rejected my offer to help him")
                else:
                    self.other_agents[agent_idx].observations.append("Rejected " + rematch.group(1) + "'s offer to help him")    
                      
                      
            if re.search(MessagePattern.carry_help_participant_reject_regex(),rm[1]):
            
                template_match = True
                rematch = re.search(MessagePattern.carry_help_participant_reject_regex(),rm[1])
                
                if rematch.group(1) == str(self.env.robot_id):
                    
                    agent_idx = info['robot_key_to_index'][rm[0]]
                    self.help_request_time[agent_idx] = [time.time(), random.random()*20]
                    
                    self.message_text += "Ok " + rm[0] + ". Don't help me then. " 
                    
            if re.search(MessagePattern.location_regex(),rm[1]):
            
                template_match = True
                
                carrying_variable = self.other_agents[agent_idx].carrying
                team_variable = self.other_agents[agent_idx].team
                
                if not (self.movement.help_status == self.movement.HelpState.being_helped and rm[0] in self.movement.help_status_info[0] and self.action_index == self.State.drop_object):            
                    self.message_text,self.action_index,_ = self.movement.message_processing_location(rm, robotState, info, self.other_agents, self.target_location, self.action_index, self.message_text, self.State.decision_state, self.next_loc)
                    
                
                rematch = re.search(MessagePattern.location_regex(),rm[1])

                #For message history                
                rematch_str = rematch.group()
                if not rematch_str[-1] == ".":
                    rematch_str += "."
                tmp_message = tmp_message.replace(rematch_str, "").strip()
               
                
                obs_string = ""
                
                if not self.other_agents[agent_idx].carrying and self.other_agents[agent_idx].carrying != carrying_variable:
                    obs_string += "Dropped an object"
                    
                    
                
                if not self.other_agents[agent_idx].team and self.other_agents[agent_idx].team != team_variable:
                    
                    if obs_string:
                        obs_string += ", "
                
                    if team_variable != str(self.env.robot_id):
                        obs_string += "Stopped helping agent " + team_variable
                    else:
                        obs_string += "Helped me carry an object"
                            
                
                
                if rematch.group(1) != "location":
                
                    if obs_string:
                        obs_string += ", "
                
                    obs_string += "Announced his current objective is " + rematch.group(1)
                    
                if rematch.group(5):
                    if obs_string:
                        obs_string += ", "
                        
                    obs_string += "Carried object " + rematch.group(6)
                    
                    
                if rematch.group(7):
                    
                        
                    if rematch.group(8) != str(self.env.robot_id):
                        if obs_string:
                            obs_string += ", "
                            
                        obs_string += "Is helping agent " + rematch.group(8)
                        
                
                if self.movement.help_status == self.movement.HelpState.being_helped and self.helping_type == self.HelpType.sensing and self.movement.help_status_info[0][0] == rm[0] and not MessagePattern.carry_help_accept(self.env.robot_id) in rm[1]: #If sensing partner stops helping
                    print("Stops helping")
                    if not rematch.group(7):
                        self.movement.help_status = self.movement.HelpState.no_request
                    elif rematch.group(8) != str(self.env.robot_id):
                        self.movement.help_status = self.movement.HelpState.no_request
                        
                if rematch.group(6):
                    carried_object = rematch.group(6)
                    self.carried_objects[agent_idx] = carried_object
                elif agent_idx in self.carried_objects and self.carried_objects[agent_idx]:
                    self.carried_objects[agent_idx] = ""
                        

                
                if obs_string and obs_string not in self.other_agents[agent_idx].observations:
                    self.other_agents[agent_idx].observations.append(obs_string)
            
            if MessagePattern.wait(self.env.robot_id) in rm[1] or re.search(MessagePattern.move_order_regex(),rm[1]):
                template_match = True
                self.target_location, self.action_index, _ = self.movement.message_processing_wait(rm, info, self.target_location, self.action_index)
                self.object_of_interest = ""
                
            if re.search(MessagePattern.wait_regex(),rm[1]):
                template_match = True
                rematch = re.search(MessagePattern.wait_regex(),rm[1])
                
                if rematch.group(1) == str(self.env.robot_id):
                    self.other_agents[agent_idx].observations.append("Waited for me to pass")
                else:
                    self.other_agents[agent_idx].observations.append("Waited for " + rematch.group(1) + " to pass")
                
            if MessagePattern.move_request(self.env.robot_id) in rm[1]:
                template_match = True
                
                
                self.message_text,self.action_index,_ = self.movement.message_processing_move_request(rm, robotState, info, self.action_index, self.message_text, self.other_agents, self.helping_type == self.HelpType.sensing)
                
            if re.search(MessagePattern.move_request_regex(),rm[1]):
                template_match = True
                rematch = re.search(MessagePattern.move_request_regex(),rm[1])
                
                if rematch.group(1) == str(self.env.robot_id):
                    self.other_agents[agent_idx].observations.append("Asked me to move")
                else:
                    self.other_agents[agent_idx].observations.append("Asked " + rematch.group(1) + " to move")
                    
            if re.search(MessagePattern.sensing_help_regex(),rm[1]): #"What do you know about object " in rm[1]:
                rematch = re.search(MessagePattern.sensing_help_regex(),rm[1])
                
                template_match = True
                
                object_id = rematch.group(1) #rm[1].strip().split()[-1] 
                
                if object_id in info['object_key_to_index']:
                
                    object_idx = info['object_key_to_index'][object_id]
                    
                    self.other_agents[agent_idx].observations.append("Asked me for information about object " + str(object_id))
                    
                    self.message_text += MessagePattern.item(robotState,object_idx,object_id, info, self.env.robot_id, self.env.convert_to_real_coordinates)
                    
                    if not self.message_text:
                         self.message_text += MessagePattern.sensing_help_negative_response(object_id)
                         
                else:
                    self.message_text += MessagePattern.sensing_help_negative_response(object_id)
                    
            if re.search(MessagePattern.item_regex_full(),rm[1]) or re.search(MessagePattern.item_regex_full_alt(),rm[1]):
            
                template_match = True
                
                new_rm = list(rm)
                new_rm[1] += MessagePattern.translate_item_message(new_rm[1],rm[0])
            
                obs_str = "Shared information with me of objects: ["
            
                if rm[1] not in objects_str:
                    objects_str[rm[1]] = []
                
            
                for ridx,rematch in enumerate(re.finditer(MessagePattern.item_regex_full(),new_rm[1])):
                
                    object_id = rematch.group(1)
                
                    if object_id == str(self.message_info[1]):
                        self.message_info[0] = True
                        
                    MessagePattern.parse_sensing_message(rematch, new_rm, robotState, info, self.other_agents, self.env.convert_to_grid_coordinates, self.env.convert_to_real_coordinates)
                    
                    
                        
                    if object_id not in objects_str[rm[1]]:
                        if objects_str[rm[1]]:
                            obs_str += ", "
                        obs_str += object_id
                        objects_str[rm[1]].append(object_id)
                        
                    if self.movement.help_status == self.movement.HelpState.asking:
                        print("kepasa", self.movement.help_status_info[0])    
                        
                    if (self.movement.help_status == self.movement.HelpState.being_helped or self.movement.help_status == self.movement.HelpState.asking) and self.helping_type == self.HelpType.sensing and self.movement.help_status_info[0][0] == rm[0]:
                        chosen_object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.chosen_object_idx)]
                        print("canceling?")
                        if chosen_object_id == object_id:
                            _,_,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text)
                            self.message_text += "Thanks " + rm[0] + ". "
                            print("canceling action")
                        
                            
                    
                obs_str += "]"
                self.other_agents[agent_idx].observations.append(obs_str)
                
                
                    
            if re.search(MessagePattern.sensing_help_negative_response_regex(),rm[1]):
            
                template_match = True
            
                rematch = re.search(MessagePattern.sensing_help_negative_response_regex(),rm[1])
                
                self.other_agents[agent_idx].observations.append("Told me he doesn't have any information about object " + rematch.group(1))
                
                if rematch.group(1) == str(self.message_info[1]):
                    self.message_info[0] = True
                    
                    
            if re.search(MessagePattern.ask_for_agent_regex(),rm[1]):
            
                template_match = True
            
                rematch = re.search(MessagePattern.ask_for_agent_regex(),rm[1])
                
                if not rematch.group(1) == self.env.robot_id and rematch.group(1) in info['robot_key_to_index']:
                
                    robot_idx = info['robot_key_to_index'][rematch.group(1)]
       
                    robo_location = robotState.get("agents", "last_seen_location", robot_idx)             
                    
                    if not (robo_location[0] == -1 and robo_location[1] == -1):
                        self.message_text += MessagePattern.agent(rematch.group(1), robot_idx, robotState, self.env.convert_to_real_coordinates)
                    else:
                        self.message_text += MessagePattern.agent_not_found(rematch.group(1))
                    
            if re.search(MessagePattern.agent_regex(),rm[1]):
            
                template_match = True
            
                rematch = re.search(MessagePattern.agent_regex(),rm[1])
                
                if not rematch.group(1) == self.env.robot_id:
                
                    robot_idx = info['robot_key_to_index'][rematch.group(1)]
                
                
                    robot = {"neighbor_type": -1, "neighbor_disabled": -1}
                    
                    last_seen = list(eval(rematch.group(3)))
                    
                    max_real_coords = self.env.convert_to_real_coordinates((robotState.latest_map.shape[0]-1, robotState.latest_map.shape[1]-1))
            
                    if last_seen[0] > max_real_coords[0] or last_seen[1] > max_real_coords[1]: #last_seen[0] == 99.99 and last_seen[1] == 99.99:
                        robot["neighbor_location"] = [-1,-1]
                    else:
                        robot["neighbor_location"] = self.env.convert_to_grid_coordinates(last_seen)
                    last_time = rematch.group(4).split(":")
                    robot["neighbor_time"] = [int(last_time[1]) + int(last_time[0])*60]
                    
                    robotState.update_robots(robot, robot_idx)
                    
                    if rematch.group(1) == str(self.message_info[1]):
                        self.message_info[0] = True
        
            if re.search(MessagePattern.order_collect_regex(),rm[1]):
            
                template_match = True
            
                if "hierarchy" in self.team_structure and self.team_structure["role"][self.env.robot_id] != "sensing":  
            
                    for rematch in re.finditer(MessagePattern.order_collect_regex(),rm[1]):
                        if rematch.group(1) == self.env.robot_id:
                            if self.team_structure["hierarchy"][self.env.robot_id] == "obey" and self.team_structure["hierarchy"][rm[0]] == "order":
                                if self.order_status == self.OrderStatus.finished:
                                    object_idx = info['object_key_to_index'][rematch.group(2)]
                                    
                                    object_location = robotState.get("objects", "last_seen_location", object_idx)
                                    if (object_location[0] == -1 and object_location[1] == -1):
                                        pdb.set_trace()
                                    
                                    
                                    self.create_action_function("collect_object('" + rematch.group(2) + "')")
                                    
                                    self.order_status = self.OrderStatus.ongoing
                                    
                                    self.message_text += MessagePattern.order_response(rm[0], "collect")
                                    self.leader_id = rm[0]
                                else:
                                    self.message_text += MessagePattern.order_response_negative(rm[0], self.leader_id)
                            else:
                                 self.message_text += MessagePattern.order_not_obey(rm[0])
                                
            if re.search(MessagePattern.order_sense_regex(),rm[1]):
            
                template_match = True
            
                if "hierarchy" in self.team_structure and self.team_structure["role"][self.env.robot_id] != "lifter":  
                    for rematch in re.finditer(MessagePattern.order_sense_regex(),rm[1]):
                        if rematch.group(1) == self.env.robot_id:
                            if self.team_structure["hierarchy"][self.env.robot_id] == "obey" and self.team_structure["hierarchy"][rm[0]] == "order":
                            
                                object_id = rematch.group(2)
                                if object_id in info['object_key_to_index'] and info['object_key_to_index'][object_id] in robotState.get_object_keys() and robotState.get("object_estimates", "danger_status", [info['object_key_to_index'][object_id],robotState.get_num_robots()]): #send info if already have it
                                    self.message_text += MessagePattern.item(robotState,info['object_key_to_index'][object_id],object_id, info, self.env.robot_id, self.env.convert_to_real_coordinates) 
                                else:
                                    if self.order_status == self.OrderStatus.finished:
                                        assigned_target_location = self.env.convert_to_grid_coordinates(eval(rematch.group(3)))
                                        self.create_action_function("sense_object(''," +  str(assigned_target_location) + ")")
                                        
                                        self.order_status = self.OrderStatus.ongoing
                                        self.message_text += MessagePattern.order_response(rm[0], "sense")
                                        self.leader_id = rm[0]
                                    else:
                                        self.message_text += MessagePattern.order_response_negative(rm[0], self.leader_id)
                            else:
                                 self.message_text += MessagePattern.order_not_obey(rm[0])
                        
            if re.search(MessagePattern.order_explore_regex(),rm[1]):
            
                template_match = True
            
                if "hierarchy" in self.team_structure:  
                    for rematch in re.finditer(MessagePattern.order_explore_regex(),rm[1]):
                        if rematch.group(1) == self.env.robot_id:
                            if self.team_structure["hierarchy"][self.env.robot_id] == "obey" and self.team_structure["hierarchy"][rm[0]] == "order":
                                if self.order_status == self.OrderStatus.finished:
                                    assigned_target_location = self.env.convert_to_grid_coordinates(eval(rematch.group(2)))
                                    self.create_action_function("go_to_location(" +  str(assigned_target_location) + ")")
                                    
                                    self.order_status = self.OrderStatus.ongoing
                                    self.message_text += MessagePattern.order_response(rm[0], "explore")
                                    self.leader_id = rm[0]
                                else:
                                    self.message_text += MessagePattern.order_response_negative(rm[0], self.leader_id)
                            else:
                                 self.message_text += MessagePattern.order_not_obey(rm[0])
                        
            if re.search(MessagePattern.order_collect_group_regex(),rm[1]):
            
                template_match = True
                
                if "hierarchy" in self.team_structure and self.team_structure["role"][self.env.robot_id] != "sensing":
            
                    for rematch in re.finditer(MessagePattern.order_collect_group_regex(),rm[1]):
                    
                        teammates = rematch.group(2)[1:].replace("]", "").split(",")
                        
                        if rematch.group(1) == self.env.robot_id or self.env.robot_id in teammates:
                        
                            if self.team_structure["hierarchy"][self.env.robot_id] == "obey" and self.team_structure["hierarchy"][rm[0]] == "order": 
                                if self.order_status == self.OrderStatus.finished:
                                
                                    if rematch.group(1) == self.env.robot_id:
                                        object_idx = info['object_key_to_index'][rematch.group(4)]
                                        
                                        self.movement.help_status = self.movement.HelpState.being_helped
                                        self.movement.help_status_info[2] = []
                                        self.movement.help_status_info[0] = []
                                        self.movement.help_status_info[6] = []
                                        self.movement.help_status_info[1] = time.time()
                                        
                                        self.movement.help_status_info[0].extend(teammates)
                                    
                                        object_location = robotState.get("objects", "last_seen_location", object_idx)
                                        if (object_location[0] == -1 and object_location[1] == -1):
                                            pdb.set_trace()
                                        
                                        self.create_action_function("collect_object('" + rematch.group(4) + "')")
                                        match_pattern = re.search(MessagePattern.location_regex(),self.message_text)
                                        
                                        if match_pattern and not match_pattern.group(7):
                                            self.message_text = self.message_text.replace(match_pattern.group(), match_pattern.group() + " Helping " + self.env.robot_id + ". ")

                                        self.order_status = self.OrderStatus.ongoing
                                        
                                            
                                    elif self.env.robot_id in teammates:

                                        self.movement.help_status = self.movement.HelpState.helping
                                        
                                        self.movement.help_status_info[0] = [rematch.group(1)]
                                        
                                        teammates_sub = teammates.copy()
                                        teammates_sub.remove(self.env.robot_id)
                                        
                                        self.movement.help_status_info[6].extend(teammates_sub)
                                        
                                        self.action_index = self.movement.State.follow
                                        
                                        self.order_status = self.OrderStatus.ongoing
                                    
                                    self.leader_id = rm[0]
                                    self.message_text += MessagePattern.order_response(rm[0], "collect")
                
                                else:
                                    self.message_text += MessagePattern.order_response_negative(rm[0], self.leader_id)
                            else:
                                 self.message_text += MessagePattern.order_not_obey(rm[0])
                        
                        
            if re.search(MessagePattern.order_response_negative_regex(), rm[1]):
                
                template_match = True
            
                rematch = re.search(MessagePattern.order_response_negative_regex(),rm[1])
                
                if "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "order" and self.env.robot_id == rematch.group(1):
                
                    robot_idx = info['robot_key_to_index'][rm[0]]
                
                    wait_time = random.randrange(self.non_request,self.non_request+20)
                    self.help_request_time[robot_idx] = [time.time(), wait_time]
                    
                    team_members = eval(robotState.get("agents", "team", robot_idx))
                    
                    for t in team_members:
                    
                        if t == robotState.get_num_robots():
                            self.action_function = ""
                            self.top_action_sequence = 0
                            robotState.set("agents", "team", robotState.get_num_robots(), "[]", info["time"])
                            
                        elif t != robot_idx:
                        
                            robot_id = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(t)]
                            self.message_text += MessagePattern.order_cancel(robot_id)
                            
                    if self.movement.help_status == self.movement.HelpState.being_helped:
                        _,self.message_text,_ = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_finish())
                    else:
                        self.movement.help_status = self.movement.HelpState.no_request
                    
            if re.search(MessagePattern.order_cancel_regex(), rm[1]):
                template_match = True
            
                rematch = re.search(MessagePattern.order_cancel_regex(),rm[1])
                
                if "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "obey" and self.env.robot_id == rematch.group(1) and self.leader_id == rm[0]:
                
                    self.order_status = self.OrderStatus.finished
                    self.leader_id = ""
                    self.action_function = ""
                    self.top_action_sequence = 0
                    self.message_text += "Ok " + rm[0] + ". "
                    
                        
            if re.search(MessagePattern.surroundings_regex(),rm[1]):
            
                template_match = True
            
                rematch = re.search(MessagePattern.surroundings_regex(),rm[1])
                view_radius = int(rematch.group(2))
                location = list(eval(rematch.group(1)))
                center = self.env.convert_to_grid_coordinates(location)
                    
                ego_location = np.where(robotState.latest_map == 5)    
                
                max_x = center[0] + view_radius
                max_y = center[1] + view_radius
                min_x = max(center[0] - view_radius, 0)
                min_y = max(center[1] - view_radius, 0)
                robotState.latest_map[min_x:max_x+1,min_y:max_y+1] = 0
                
                if rematch.group(3):
                    walls = eval(rematch.group(4))
                    for w in walls:
                        grid_wall = self.env.convert_to_grid_coordinates(w)
                        robotState.latest_map[grid_wall[0],grid_wall[1]] = 1
                
                if rematch.group(6):
                    objects = eval(rematch.group(7))
                    for o_key in objects.keys():
                        grid_object = self.env.convert_to_grid_coordinates(objects[o_key])
                        robotState.latest_map[grid_object[0],grid_object[1]] = 2
                        
                        self.env.update_objects_info(str(o_key), 0, {}, grid_object, 0, False)
                        o_idx = self.env.object_key_to_index[str(o_key)]
                        template_item_info = {'item_weight': 0, 'item_danger_level': 0, 'item_danger_confidence': np.array([0.]), 'item_location': np.array(grid_object, dtype=np.int16), 'item_time': np.array([0], dtype=np.int16)}
                        robotState.update_items(template_item_info, o_key, o_idx, -1)
             
                robotState.latest_map[ego_location[0][0],ego_location[1][0]] = 5
                
            if MessagePattern.order_finished() in rm[1]:  
            
                template_match = True
            
                if "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "order":
                    robot_idx = info['robot_key_to_index'][rm[0]]
                    
                    self.other_agents[robot_idx].previous_assignment  = self.other_agents[robot_idx].assignment
                    self.other_agents[robot_idx].assignment = "" 
                    
                    robotState.set("agents", "team", robot_idx, "[]", info["time"])
                    self.agent_requesting_order = True
                      
            if re.search(MessagePattern.sensing_ask_help_regex(),rm[1]):
            
                template_match = True
            
                rematch = re.search(MessagePattern.sensing_ask_help_regex(),rm[1])
                
                if rematch.group(1) == str(self.env.robot_id) and self.team_structure["role"][self.env.robot_id] != "lifter":
                
                    """
                    if re.search(MessagePattern.sensing_ask_help_regex(),self.message_text): #This means the robot is preparing to ask for help and reject the help request, we shouldn't allow this
                        #self.message_text = self.message_text.replace(re.search(MessagePattern.sensing_ask_help_regex(),self.message_text).group(), "")
                        #self.sense_request = self.State_Sense_Request.no_request
                        self.sense_request_data = []
                    """
                    object_id = rematch.group(2)
                
                    if object_id in info['object_key_to_index'] and info['object_key_to_index'][object_id] in robotState.get_object_keys() and robotState.get("object_estimates", "danger_status", [info['object_key_to_index'][object_id],robotState.get_num_robots()]): #send info if already have it
                        self.message_text += MessagePattern.item(robotState,info['object_key_to_index'][object_id],object_id, info, self.env.robot_id, self.env.convert_to_real_coordinates)
                    else:
                    
                        self.message_text,self.action_index,_ = self.movement.message_processing_carry_help(rm, robotState, self.action_index, self.message_text)
                        
                        if MessagePattern.carry_help_accept(rm[0]) in self.message_text: #not robotState.object_held and self.movement.help_status == self.movement.HelpState.no_request and self.sense_request == self.State_Sense_Request.no_request:
                        
                            #self.sense_request = self.State_Sense_Request.helping
                            
                            self.helping_type = self.HelpType.sensing
                            
                            location = list(eval(rematch.group(3)))
                            
                            max_real_coords = self.env.convert_to_real_coordinates((robotState.latest_map.shape[0]-1, robotState.latest_map.shape[1]-1))
                            
                            if location[0] > max_real_coords[0] or location[1] > max_real_coords[1]:#location[0] == 99.99 and location[1] == 99.99:
                                grid_location = [-1,-1]
                            else:
                                grid_location = self.env.convert_to_grid_coordinates(location)

                            #self.message_text += MessagePattern.sensing_ask_help_confirm(rm[0], object_id)
                            
                            last_time = rematch.group(4).split(":")
                            timer = int(last_time[1]) + int(last_time[0])*60
                            template_item_info = {'item_weight': 0, 'item_danger_level': 0, 'item_danger_confidence': np.array([0.]), 'item_location': np.array([int(grid_location[0]), int(grid_location[1])], dtype=np.int16), 'item_time': np.array([timer], dtype=np.int16)}
                            
                            self.movement.help_status_info[5] = grid_location
                            
                            if object_id in info['object_key_to_index']:
                                ob_key = info['object_key_to_index'][object_id]
                                robotState.update_items(template_item_info, object_id, ob_key, -1)
                            
                            #self.action_sequence = 0
                            #self.top_action_sequence = 0
                        
                            #self.create_action_function("sense_by_request('" + object_id + "','" + rm[0] + "'," + str(grid_location) + ")")
                        #else:
                        #    self.message_text += MessagePattern.sensing_ask_help_reject(rm[0])


            """
            if re.search(MessagePattern.sensing_ask_help_confirm_regex(),rm[1]):
            
                template_match = True
            
                rematch = re.search(MessagePattern.sensing_ask_help_confirm_regex(),rm[1])
                
                if rematch.group(2) == str(self.env.robot_id) and self.sense_request_data and self.sense_request_data[0] == rm[0] and self.sense_request == self.State_Sense_Request.asking:
                    self.sense_request = self.State_Sense_Request.being_helped
                    
            if re.search(MessagePattern.sensing_ask_help_reject_regex(),rm[1]):
            
                template_match = True
            
                rematch = re.search(MessagePattern.sensing_ask_help_reject_regex(),rm[1])
                
                if rematch.group(1) == str(self.env.robot_id) and self.sense_request_data and self.sense_request_data[0] == rm[0] and self.sense_request == self.State_Sense_Request.asking:
                    self.sense_request = self.State_Sense_Request.no_request
                    self.sense_request_data = []
            
            """
            
            if re.search(MessagePattern.finish_regex(),rm[1]):
            
                template_match = True
            
                robot_idx = info['robot_key_to_index'][rm[0]]
                
                self.other_agents[robot_idx].finished = True
                
                ego_location = np.where(robotState.latest_map == 5)
                
                #if not robotState.robots[info['robot_key_to_index'][rm[0]]]["neighbor_type"]:
                
                if "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "obey" and rm[0] in self.leaders:
                    self.finished = True
                        
                    for oa in self.other_agents:
                        oa.finished = True
                        
                    print("FINISHING BY ORDER")
                else:
                    if not self.finished and "go_to_meeting_point" not in self.action_function: 
                        self.message_text += "I haven't finished yet. "
                    elif [ego_location[0][0],ego_location[1][0]] not in self.ending_locations:
                        self.message_text += "Let's finish. Let's go to the final meeting location, come with me. "
                    else:
                        self.message_text += "Let's finish. "
                    
                print("Finished other agent")
                    
                    
            if re.search(MessagePattern.finish_reject_regex(),rm[1]): #MessagePattern.finish_reject() in rm[1]:
            
                template_match = True
            
                robot_idx = info['robot_key_to_index'][rm[0]]
                
                self.other_agents[robot_idx].finished = False


            if re.search(MessagePattern.come_closer_regex(),rm[1]):
                template_match = True
            
                rematch = re.search(MessagePattern.come_closer_regex(),rm[1])
                
                if rematch.group(1) == str(self.env.robot_id) and self.action_index == self.movement.State.wait_random:
                    self.action_index = self.movement.last_action_index
                    self.pending_location = []
            
            if re.search(MessagePattern.sensing_ask_help_incorrect_regex(),rm[1]):
                template_match = True    
            
            #template_match = True #CNL only
            
            if not template_match and translated_messages_index >= 0 and translated_messages_index >= rm_idx: #This means the translated message doesn't make sense
                print("understand here 1")
                self.message_text += "I didn't understand you " + rm[0] + ". "
                continue
            
            print(not template_match, not robotState.get("agents", "type", info['robot_key_to_index'][rm[0]]), rm[1])
            if not template_match and not robotState.get("agents", "type", info['robot_key_to_index'][rm[0]]): #Human sent a message, we need to translate it. We put this condition at the end so that humans can also send messages that conform to the templates
                translated_message,message_to_user = self.human_to_ai_text.convert_to_ai(rm[0], rm[1], info, robotState, self.message_history, True)
                
                if translated_message:
                    if translated_message == self.human_to_ai_text.noop:
                        pass
                    elif message_to_user:
                        self.message_text += translated_message
                    else:
                        if translated_messages_index == -1:
                            translated_messages_index = len(received_messages)
                        received_messages.append((rm[0], translated_message, rm[2])) #Add it to the list of received messages
                else:
                    print("understand here 2")
                    self.message_text += "I didn't understand you " + rm[0] + ". "
   
   
            if tmp_message:
                tmp_message_history.append({"Sender": rm[0], "Message": tmp_message})
        
                
        if tmp_message_history:
            self.message_history.extend(tmp_message_history)
    
    def get_neighboring_agents(self, robotState, ego_location):
    
        nearby_other_agents = []
        disabled_agents = []
        #Get number of neighboring robots at communication range
        for n_idx in range(robotState.get_num_robots()):
        
            robo_location = robotState.get("agents", "last_seen_location", n_idx)
            robo_disabled = robotState.get("agents", "disabled", n_idx)
        
            if robo_disabled == 0 and not (robo_location[0] == -1 and robo_location[1] == -1) and self.env.compute_real_distance([robo_location[0],robo_location[1]],[ego_location[0][0],ego_location[1][0]]) < self.env.map_config['communication_distance_limit']:
                nearby_other_agents.append(n_idx)
            elif robo_disabled == 1:
                disabled_agents.append(n_idx)
                self.other_agents[n_idx].assignment = ""
                self.other_agents[n_idx].previous_assignment = ""
                self.other_agents[n_idx].finished = True
                print("Agent nearby finished")
                
        return nearby_other_agents,disabled_agents    
        
    def modify_occMap(self,robotState, occMap, ego_location, info):
    
        self.movement.modify_occMap(robotState, occMap, ego_location, info, self.next_loc)
        
        if self.action_index != self.State.drop_object and robotState.object_held:
            if self.movement.help_status == self.movement.HelpState.being_helped:
                for agent_id in self.movement.help_status_info[0]: #self.movement.being_helped: #if you are being helped, ignore locations of your teammates

                    agent_idx = info['robot_key_to_index'][agent_id]
                    other_robot_location = robotState.get("agents", "last_seen_location", agent_idx)
                    
                    if not (other_robot_location[0] == -1 and other_robot_location[1] == -1) and occMap[other_robot_location[0],other_robot_location[1]] != 5:
                        occMap[other_robot_location[0],other_robot_location[1]] = 3
                    
    
    def create_action_function(self, function_str):
    
        #self.action_function = input("Next action > ").strip()
        #self.action_function = "scan_area()"
        self.action_function = "self." + function_str[:-1]
                
        #if not ("drop" in self.action_function or "activate_sensor" in self.action_function or "scan_area" in self.action_function):
        if not ("explore" in self.action_function or "wait" in self.action_function):
            self.action_function += ","
                        
        self.action_function += "robotState, next_observation, info)"
        

    def control(self,messages, robotState, info, next_observation):
        #print("Messages", messages)
        
        terminated = False
        message_order = ""
        
        self.occMap = np.copy(robotState.latest_map)
        
        ego_location = np.where(self.occMap == 5)
        
        self.modify_occMap(robotState, self.occMap, ego_location, info)
        
        self.nearby_other_agents, self.disabled_agents = self.get_neighboring_agents(robotState, ego_location)
        
        
        if messages: #Process received messages
            self.message_processing(messages, robotState, info)
            
        
        if self.movement.help_status == self.movement.HelpState.accepted and time.time() - self.movement.help_status_info[1] > self.help_time_limit2: #reset help state machine after a while if accepted a request
            self.movement.help_status = self.movement.HelpState.no_request
            self.movement.help_status_info[0] = []
        
        #self.message_text += MessagePattern.exchange_sensing_info(robotState, info, self.nearby_other_agents, self.other_agents, self.env.robot_id, self.env.convert_to_real_coordinates) #Exchange info about objects sensing measurements
        
        if not self.message_text:
        
            self.action_history.append(self.action_index)
            
            if len(self.action_history) > 2 and self.action_history[-1] == self.movement.State.wait_free and self.action_history[-2] != self.movement.State.wait_free and self.action_history[-3] == self.movement.State.wait_free and not robotState.object_held:
            
                if not self.movement.help_status == self.movement.HelpState.helping: #(self.movement.help_status == self.movement.HelpState.helping and self.other_agents[info["robot_key_to_index"][self.movement.help_status_info[0][0]]].carrying):
                
            
                    self.action_index = self.State.decision_state
                    self.action_function = ""
                    print("action_function -1", self.agent_requesting_order)
                    
                    print("Cancel help")
                    
                    if self.movement.help_status == self.movement.HelpState.being_helped:
                        _,self.message_text,_ = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_finish())
                    else:
                        self.movement.help_status = self.movement.HelpState.no_request
                 
                    
                    robotState.set("agents", "team", robotState.get_num_robots(), "[]", info["time"])
                
        
            if self.action_index == self.State.decision_state or self.action_index == self.State.drop_object:
                if not self.action_function or self.help_requests:
                    self.action_sequence = 0
                    self.top_action_sequence = 0
                    
                    
                    if "hierarchy" in self.team_structure:
                        if self.team_structure["hierarchy"][self.env.robot_id] == "obey":
                            function_str = self.decision_obey(messages, robotState, info, [], self.nearby_other_agents, next_observation)
                        #elif self.team_structure["hierarchy"][self.env.robot_id] == "order":
                        #    function_str = self.decision_order(messages, robotState, info, [], self.nearby_other_agents)
                        else:
                            function_str = self.decision(messages, robotState, info, [], self.nearby_other_agents, self.help_requests)
                            message_order = self.message_text
                    else:
                        function_str = self.decision(messages, robotState, info, [], self.nearby_other_agents, self.help_requests)
                    
                    if function_str:
                    
                        print("Starting...")
                    
                        self.create_action_function(function_str)
                        
                    else:
                        self.action_function = ""
                        print("action_function 0", self.agent_requesting_order)
                        
                try:
                    action, action_finished,function_output = eval(self.action_function)
                except:
                    pdb.set_trace()
                
                if message_order not in self.message_text:
                    self.message_text = message_order + self.message_text
                    message_order = ""
                #except:
                #    pdb.set_trace()

                
                if action_finished:
                    self.action_sequence = 0
                    self.top_action_sequence = 0
                    
                    
                    if "hierarchy" in self.team_structure:
                        if self.team_structure["hierarchy"][self.env.robot_id] == "obey":
                            function_str = self.decision_obey(messages, robotState, info, function_output, self.nearby_other_agents, next_observation)
                        #elif self.team_structure["hierarchy"][self.env.robot_id] == "order":
                        #    function_str = self.decision_order(messages, robotState, info, function_output, self.nearby_other_agents)
                        else:
                            robotState.set("agents", "team", robotState.get_num_robots(), "[]", info["time"])
                            function_str = self.decision(messages, robotState, info, function_output, self.nearby_other_agents, self.help_requests)
                    else:
                        function_str = self.decision(messages, robotState, info, function_output, self.nearby_other_agents, self.help_requests)
                    
                    if function_str:
                        self.create_action_function(function_str)
                    else: #No function selected
                        self.action_function = ""
                        print("action_function 1", self.agent_requesting_order)
                        
                        
                if self.agent_requesting_order:
                    
                    function_str = self.decision(messages, robotState, info, function_output, self.nearby_other_agents, self.help_requests)
                    
                    if function_str:
                        self.create_action_function(function_str)
                        print("Created function")
                        self.action_sequence = 0
                        self.top_action_sequence = 0


            else:
                action = self.sample_action_space
                action["action"] = -1
                action["num_cells_move"] = 1
            
                previous_action_index = self.action_index
                
                self.message_text,self.action_index,self.target_location,self.next_loc, low_action = self.movement.movement_state_machine(self.occMap, info, robotState, self.action_index, self.message_text, self.target_location,self.State.decision_state, self.next_loc, ego_location, -1)
                self.object_of_interest = ""
                
                if previous_action_index == self.movement.State.wait_message and not self.movement.help_status == self.movement.HelpState.asking: #self.movement.asked_help:
                    self.action_function = ""
                    robotState.set("agents", "team", robotState.get_num_robots(), "[]", info["time"])
                    print("action_function 2", self.agent_requesting_order)
                
                action["action"] = low_action
                
            #print("Locationss", self.next_loc, self.target_location, ego_location)  
            if self.nearby_other_agents: #If there are nearby robots, announce next location and goal
                self.message_text, self.next_loc = self.movement.send_state_info(action["action"], self.next_loc, self.target_location, self.message_text, self.other_agents, self.nearby_other_agents, ego_location, robotState, self.object_of_interest, self.held_objects)  
                #pdb.set_trace()
               
                
            if self.message_text: #Send message first before doing action
                

                if re.search(MessagePattern.location_regex(),self.message_text):
                    rematch = re.search(MessagePattern.location_regex(),self.message_text)
                    target_goal = eval(rematch.group(2))
                    target_loc = eval(rematch.group(3))
                    
                    if target_goal != target_loc and not (self.previous_message and self.previous_message[0] == target_goal and self.previous_message[1] == target_loc): #Only if there was a change of location do we prioritize this message

                        self.previous_message = [target_goal,target_loc]

                        
                        action,_,_ = self.send_message(self.message_text, robotState, next_observation, info)
                        print("SENDING MESSAGE", info['time'], self.message_text)
                        self.message_text = ""

                        
                
        else:
        
            action,_,_ = self.send_message(self.message_text, robotState, next_observation, info)
            print("SENDING MESSAGE2", info['time'], self.message_text)
            self.message_text = ""
            

        

        if action["action"] == -1 or action["action"] == "":
            
            action["action"] = Action.get_occupancy_map.value
            print("STUCK")


        print("action index:",self.action_index, "action:", Action(action["action"]), ego_location, self.action_function, self.movement.help_status, self.top_action_sequence, self.order_status, info['time'])
        
        if "end_participation" in self.action_function:
            print("end participation")
            terminated = True
        
        #print("Finished?", [p.finished for p in self.other_agents], self.finished)
        
        if all(p.finished for p in self.other_agents) and self.finished:
            print("finished participation")
            terminated = True

        return action,terminated
        
    def sense_by_request(self, object_id, agent_id, grid_location, robotState, next_observation, info):
    
        finished = False
        action = self.sample_action_space
        action["robot"] = 0
        action["action"] = -1
        output = []
        ego_location = np.where(robotState.latest_map == 5)
        
        if self.top_action_sequence < 2:
            action, tmp_finished, output = self.sense_object(object_id, grid_location, robotState, next_observation, info)
            
            if tmp_finished:
                self.top_action_sequence = 2
         
        elif self.top_action_sequence == 2:
               
            chosen_location = robotState.get("agents", "last_seen_location", info['robot_key_to_index'][str(agent_id)]) #robotState.robots[info['robot_key_to_index'][str(agent_id)]]["neighbor_location"]
            
            self.object_of_interest = ""
            
            if (chosen_location[0] == -1 and chosen_location[1] == -1): #if there is no robot in the correct place
            
                if self.nearby_other_agents:
                    action,temp_finished,_ = self.ask_info(agent_id, MessagePattern.ask_for_agent(agent_id), robotState, next_observation, info)
                    if temp_finished:
                        self.action_sequence = 0
                        chosen_location = robotState.get("agents", "last_seen_location", info['robot_key_to_index'][str(agent_id)]) #robotState.robots[info['robot_key_to_index'][str(agent_id)]]["neighbor_location"]
                        if (chosen_location[0] == -1 and chosen_location[1] == -1):
                            self.movement.help_status = self.movement.HelpState.no_request
                            self.message_text += MessagePattern.object_not_found(agent_id, object_id)
                            print("AAAAAAAAAA")
                            #pdb.set_trace()
                            finished = True
                else:
                    finished = True
                    self.action_sequence = 0
            
            action, temp_finished, output = self.go_to_location(agent_id, robotState, next_observation, info)
            
            real_distance = self.env.compute_real_distance([chosen_location[0],chosen_location[1]],[ego_location[0][0],ego_location[1][0]])
                
            distance_limit = self.env.map_config['communication_distance_limit']-1
            
            if real_distance < distance_limit:
                self.top_action_sequence += 1
                
            print(self.next_loc)
        elif self.top_action_sequence == 3:
            
            if object_id in info['object_key_to_index']:
                ob_key = info['object_key_to_index'][object_id]
                self.message_text += MessagePattern.item(robotState,ob_key,object_id, info, self.env.robot_id, self.env.convert_to_real_coordinates)
            else:
                self.message_text += MessagePattern.object_not_found(agent_id, object_id)
                
            self.movement.help_status = self.movement.HelpState.no_request
            
            finished = True
        
        return action, finished, output
        
    def sense_object(self, object_id, grid_location, robotState, next_observation, info):
        
        finished = False
        
        if self.top_action_sequence == 0:
        
            if grid_location and not (grid_location[0] == -1 and grid_location[0] == -1):
                chosen_location = grid_location
                object_id = grid_location
            else:
                chosen_location = robotState.get("objects", "last_seen_location", info['object_key_to_index'][str(object_id)]) #robotState.items[info['object_key_to_index'][str(object_id)]]["item_location"]
                self.object_of_interest = object_id
        
            if (chosen_location[0] == -1 and chosen_location[1] == -1):# or self.occMap[chosen_location[0],chosen_location[1]] != 2: #if there is no object in the correct place
                print("object not found for sensing")

                finished = True
        
            action, temp_finished, output = self.go_to_location(object_id, robotState, next_observation, info)
            if temp_finished:
                self.top_action_sequence += 1
        elif self.top_action_sequence == 1:
            action, finished, output = self.activate_sensor(robotState, next_observation, info)
            
    
        return action, finished, output
        
        
    def collect_object(self, object_id, robotState, next_observation, info):
    
        finished = False
        output = []
        
        action = self.sample_action_space
        
        ego_location = np.where(robotState.latest_map == 5)
        
        self.chosen_object_idx = info['object_key_to_index'][str(object_id)]
        
        if self.top_action_sequence == 0:
        
            self.object_of_interest = object_id

            chosen_location = robotState.get("objects", "last_seen_location", info['object_key_to_index'][str(object_id)]) #robotState.items[info['object_key_to_index'][str(object_id)]]["item_location"]
            
            if (chosen_location[0] == -1 and chosen_location[1] == -1) or tuple(chosen_location) in self.extended_goal_coords: #or self.occMap[chosen_location[0],chosen_location[1]] != 2 if there is no object in the correct place
                finished = True
                action["action"],self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_complain())
            else:
            
                wait = False
                for agent_id in self.movement.help_status_info[0]:
                    agent_idx = info['robot_key_to_index'][agent_id]
                    other_robot_location = robotState.get("agents", "last_seen_location", agent_idx) #robotState.robots[agent_idx]["neighbor_location"]
                    if (other_robot_location[0] == -1 and other_robot_location[1] == -1):
                        wait = True
                    else:
                        real_distance = self.env.compute_real_distance([other_robot_location[0],other_robot_location[1]],[ego_location[0][0],ego_location[1][0]])
                
                        distance_limit = self.env.map_config['communication_distance_limit']-1
            
                        if real_distance >= distance_limit:
                            wait = True
                            if not robotState.get("agents", "type", agent_idx): #robotState.robots[agent_idx]["neighbor_type"]:
                                self.message_text += MessagePattern.come_closer(agent_id)
                if wait:          
                    if time.time() - self.movement.help_status_info[1] > self.help_time_limit2: #time.time() - self.movement.asked_time > self.help_time_limit2:                           
                        action["action"],self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_complain())
                        finished = True
                    else:
                        action["action"] = Action.get_occupancy_map.value
                else:            
                    action, temp_finished, output = self.go_to_location(object_id, robotState, next_observation, info)
                    self.movement.help_status_info[1] = time.time()
                    if temp_finished:
                        self.top_action_sequence += 1
                        #self.movement.asked_time = time.time()
                        #self.movement.being_helped_locations = []
                        self.movement.help_status_info[2] = []
                        self.movement.help_status_info[7] = []
                        
        elif self.top_action_sequence == 1:
        
            wait_for_others,combinations_found,self.message_text = self.movement.wait_for_others_func(self.occMap, info, robotState, self.nearby_other_agents, [], ego_location,self.message_text)
        
            if not wait_for_others:        
                action, temp_finished, output = self.pick_up(object_id, robotState, next_observation, info)
                self.held_object = object_id
                self.movement.help_status_info[1] = time.time()
                if temp_finished:
                    self.top_action_sequence += 1
                    #self.movement.being_helped_locations = []
                    self.movement.help_status_info[2] = []
                    self.movement.help_status_info[7] = []
                    self.previous_next_loc = []
   
   
                    g_coord = []
                    for g_coord in self.env.goal_coords:
                        if not self.occMap[g_coord[0],g_coord[1]]:
                            self.target_location = g_coord
                            self.object_of_interest = "goal"
                            break
                    
                    
                    
            elif not combinations_found: #No way of moving                          
                action["action"],self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_finish())
                finished = True
            elif time.time() - self.movement.help_status_info[1] > self.help_time_limit2: #time.time() - self.movement.asked_time > self.help_time_limit2:                           
                action["action"],self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_complain())
                finished = True
            else:
                action["action"] = Action.get_occupancy_map.value 
                
        elif self.top_action_sequence == 2:

            self.action_index = self.State.drop_object
            
            if not robotState.object_held:            
                action["action"],self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_complain())
                self.top_action_sequence += 1
            else:
            
                if self.movement.help_status == self.movement.HelpState.being_helped:
                    for agent_id in self.movement.help_status_info[0]:

                        agent_idx = info['robot_key_to_index'][agent_id]
                        other_robot_location = robotState.get("agents", "last_seen_location", agent_idx) #robotState.robots[agent_idx]["neighbor_location"]
                        
                        if not (other_robot_location[0] == -1 and other_robot_location[1] == -1):
                            self.occMap[other_robot_location[0],other_robot_location[1]] = 3
                            
                
                loop_done = False
                
                if not self.previous_next_loc or (self.previous_next_loc and self.previous_next_loc[0].tolist() == [ego_location[0][0],ego_location[1][0]]):
                    action["action"], self.next_loc, self.message_text, self.action_index = self.movement.go_to_location(self.target_location[0],self.target_location[1], self.occMap, robotState, info, ego_location, self.action_index, help_sensing=self.helping_type == self.HelpType.sensing)
                    
                    
                    print("HAPPENING", action["action"], self.next_loc)
                    
                    if not action["action"] and isinstance(action["action"], list):
                        loop_done = True
                     
                    if not loop_done and self.next_loc:
                        self.previous_next_loc = [self.next_loc[0]]
                        #self.movement.being_helped_locations = []
                        self.movement.help_status_info[2] = []
                        self.movement.help_status_info[7] = []
                    
                        #print("PEFIOUVS",self.movement.being_helped_locations, self.next_loc, self.previous_next_loc)
                        
                    
                    
                    
                
                if not loop_done:
                    wait_for_others,combinations_found,self.message_text = self.movement.wait_for_others_func(self.occMap, info, robotState, self.nearby_other_agents, self.previous_next_loc, ego_location,self.message_text)
                    
                    if not combinations_found: #No way of moving
                        self.top_action_sequence += 1
                        action["action"] = Action.get_occupancy_map.value
                        _,self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_finish())
                
                
                        
                if loop_done or not wait_for_others: #If carrying heavy objects, wait for others
                    
                    action["action"], self.next_loc, self.message_text, self.action_index = self.movement.go_to_location(self.target_location[0],self.target_location[1], self.occMap, robotState, info, ego_location, self.action_index, help_sensing=self.helping_type == self.HelpType.sensing)
                    
                    if self.next_loc and self.previous_next_loc and not self.previous_next_loc[0].tolist() == self.next_loc[0].tolist(): #location changed
                        self.previous_next_loc = []
                        
                        
                    """    
                    if self.occMap[self.target_location[0],self.target_location[1]] == 2: #A package is now there
                        self.action_index = self.State.pickup_and_move_to_goal
                        self.movement.being_helped_locations = []
                    """
                
                    if not action["action"] and isinstance(action["action"], list): #If already next to drop location
                        action["action"] = Action.get_occupancy_map.value
                        self.top_action_sequence += 1
                        self.target_location = self.past_location
                        self.object_of_interest = ""

                    else:
                        self.past_location = [ego_location[0][0],ego_location[1][0]]
                        
                    #self.movement.asked_time = time.time()
                    self.movement.help_status_info[1] = time.time()
                    
                elif time.time() - self.movement.help_status_info[1] > self.help_time_limit2: #time.time() - self.movement.asked_time > self.help_time_limit2:
                    self.top_action_sequence += 1
                    action["action"] = Action.get_occupancy_map.value
                    _,self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_complain())
                    
                elif action["action"] != Action.drop_object.value:
                    action["action"] = Action.get_occupancy_map.value
                    print("waiting for others...")
            
            
            """
            action, temp_finished, output = self.go_to_location(-1, robotState, next_observation, info)
            if temp_finished:
                self.top_action_sequence += 1
                self.action_index = self.State.decision_state
            """
        elif self.top_action_sequence == 3:
            action, finished, output = self.drop(robotState, next_observation, info)
            
            self.action_index = self.State.decision_state
        
            #if self.movement.being_helped:
            if self.movement.help_status == self.movement.HelpState.being_helped:
                self.message_text += MessagePattern.carry_help_finish()
                #self.movement.asked_time = time.time()
                self.movement.help_status_info[1] = time.time()
                
            #self.movement.being_helped = []
            #self.movement.being_helped_locations = []
            self.movement.help_status = self.movement.HelpState.no_request
            self.movement.help_status_info[2] = []
            self.movement.help_status_info[7] = []
            self.movement.help_status_info[3] = []
            self.object_of_interest = ""
        
    
        return action, finished, output
        
    def explore(self, robotState, next_observation, info):
    
        
    
        action, finished, output = self.go_to_location(-2, robotState, next_observation, info)
        
        if action["action"] == -1:
            action["action"] = Action.get_occupancy_map.value
            self.action_index = self.State.decision_state
            finished = True
        
        if finished:
            self.explore_location = []
        
        return action, finished, output
        
        
    def ask_for_help(self, object_id, robot_id, robotState, next_observation ,info):
    
        action = self.sample_action_space
        action["robot"] = 0
        finished = False
        output = []
        
        ego_location = np.where(robotState.latest_map == 5)
        
        self.chosen_object_idx = info['object_key_to_index'][str(object_id)]
        
        
        if self.top_action_sequence == 0:

            chosen_location = robotState.get("agents", "last_seen_location", info['robot_key_to_index'][str(robot_id)]) #robotState.robots[info['robot_key_to_index'][str(robot_id)]]["neighbor_location"]
            
            if (chosen_location[0] == -1 and chosen_location[1] == -1): #if there is no agent in the correct place
                print("no more helping")
                if self.nearby_other_agents:
                    action,temp_finished,_ = self.ask_info(robot_id, MessagePattern.ask_for_agent(robot_id), robotState, next_observation, info)
                    if temp_finished:
                        self.action_sequence = 0
                        chosen_location = robotState.get("agents", "last_seen_location", info['robot_key_to_index'][str(robot_id)]) #robotState.robots[info['robot_key_to_index'][str(robot_id)]]["neighbor_location"]
                        if (chosen_location[0] == -1 and chosen_location[1] == -1):
                            finished = True
                else:
                    finished = True
                    self.action_sequence = 0
            
            action, temp_finished, output = self.go_to_location(robot_id, robotState, next_observation, info)
            
            real_distance = self.env.compute_real_distance([chosen_location[0],chosen_location[1]],[ego_location[0][0],ego_location[1][0]])
                
            distance_limit = self.env.map_config['communication_distance_limit']-1
            
            if real_distance < distance_limit:
                self.top_action_sequence += 1
                #self.movement.asked_time = time.time()
                #self.movement.being_helped_locations = []
                self.movement.help_status_info[1] = time.time()
                self.movement.help_status_info[2] = []
                
        elif self.top_action_sequence == 1:
            object_idx =info['object_key_to_index'][str(object_id)]
            self.helping_type = self.HelpType.carrying
            
            robots_already_helping = 0
            if self.movement.help_status == self.movement.HelpState.being_helped:
                robots_already_helping = len(self.movement.help_status_info[0])
                action["action"] = Action.get_occupancy_map.value
            else:
                self.movement.help_status = self.movement.HelpState.asking
                self.movement.help_status_info[0] = []
                if not self.action_index == self.movement.State.wait_free and not self.action_index == self.movement.State.wait_random:
                    self.movement.last_action_index = self.action_index
                self.action_index = self.movement.State.wait_message
                
            
            agent_idx = info['robot_key_to_index'][robot_id]
            nearby_other_agents_ids = []
            for noa in self.nearby_other_agents:
                nearby_other_agents_ids.append(list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(noa)])
                
            for helping_agents in self.movement.help_status_info[0]:
                if helping_agents in nearby_other_agents_ids:
                    nearby_other_agents_ids.remove(helping_agents)
            
            if not robotState.get("agents", "type", agent_idx) and nearby_other_agents_ids and False:
                message_to_personalize = MessagePattern.carry_help(str(object_id),robotState.get("objects", "weight", object_idx)-1-robots_already_helping)
                message_to_send = self.human_to_ai_text.personalize_message(message_to_personalize, nearby_other_agents_ids, self.message_history)
                
                if not message_to_send:
                    message_to_send = message_to_personalize
            else:
                message_to_send = MessagePattern.carry_help(str(object_id),robotState.get("objects", "weight", object_idx)-1-robots_already_helping)
            
            self.message_text += message_to_send
         
            #self.movement.asked_help = True
            #self.movement.asked_time = time.time()
            
            self.movement.help_status_info[1] = time.time()
            
            
            self.chosen_object_idx = object_idx
            print("ASKING HELP")
            self.top_action_sequence += 1
            
        elif self.top_action_sequence == 2:
            object_idx =info['object_key_to_index'][str(object_id)]
            if (self.movement.help_status != self.movement.HelpState.asking and len(self.movement.help_status_info[0])+1 >= robotState.get("objects", "weight", object_idx)) or time.time() - self.movement.help_status_info[1] > self.movement.wait_time_limit:
                finished = True
            action["action"] = Action.get_occupancy_map.value
        
        
        
        return action,finished,output
        
    
    def approach(self, robot_id, robotState, next_observation, info):
    
        action = self.sample_action_space
        action["robot"] = 0
        action["action"] = -1
        finished = False
        output = []
        
        ego_location = np.where(robotState.latest_map == 5)
    
        action, temp_finished, output = self.go_to_location(robot_id, robotState, next_observation, info)
        
        robot_idx = info['robot_key_to_index'][str(robot_id)]
            
        chosen_location = robotState.get("agents", "last_seen_location", robot_idx) #robotState.robots[robot_idx]["neighbor_location"]
        
        if (chosen_location[0] == -1 and chosen_location[1] == -1): #if there is no agent in the correct place
            if self.nearby_other_agents and self.top_action_sequence == 0: #problem here, infinite loop
                action,temp_finished,_ = self.ask_info(str(robot_id), MessagePattern.ask_for_agent(str(robot_id)), robotState, next_observation, info)
                self.top_action_sequence += 1
                
            else:
                true_ending_locations = [loc for loc in self.ending_locations if self.occMap[loc[0],loc[1]] == 0 or self.occMap[loc[0],loc[1]] == -2]
                
                target_location = random.choice(true_ending_locations)  
                if [ego_location[0][0],ego_location[1][0]] in self.ending_locations: #If we are already in the ending locations just stay there
                    target_location = [ego_location[0][0],ego_location[1][0]]
                    
                  
                action, temp_finished, output = self.go_to_location(target_location, robotState, next_observation, info)
        
        else:    
            real_distance = self.env.compute_real_distance([chosen_location[0],chosen_location[1]],[ego_location[0][0],ego_location[1][0]])
                
            distance_limit = self.env.map_config['communication_distance_limit']-1
            
            if real_distance < distance_limit:
                finished = True
    
        return action,finished,output
        
    def ask_for_sensing(self, object_id, robot_id, robotState, next_observation ,info):
    
        action = self.sample_action_space
        action["robot"] = 0
        action["action"] = -1
        finished = False
        output = []
        
        ego_location = np.where(robotState.latest_map == 5)
        
        self.chosen_object_idx = info['object_key_to_index'][str(object_id)]
        
        if self.top_action_sequence == 0:

            try:
                chosen_location = robotState.get("agents", "last_seen_location", info['robot_key_to_index'][str(robot_id)]) #robotState.robots[info['robot_key_to_index'][str(robot_id)]]["neighbor_location"]
            except:
                pdb.set_trace()
            
            if (chosen_location[0] == -1 and chosen_location[1] == -1): #if there is no agent in the correct place
                if self.nearby_other_agents:
                    action,temp_finished,_ = self.ask_info(robot_id, MessagePattern.ask_for_agent(robot_id), robotState, next_observation, info)
                    if temp_finished:
                        self.action_sequence = 0
                        chosen_location = robotState.get("agents", "last_seen_location", info['robot_key_to_index'][str(robot_id)]) #robotState.robots[info['robot_key_to_index'][str(robot_id)]]["neighbor_location"]
                        if (chosen_location[0] == -1 and chosen_location[1] == -1):
                            finished = True
                else:
                    finished = True
                    self.action_sequence = 0
            
            action, temp_finished, output = self.go_to_location(robot_id, robotState, next_observation, info)
            
            real_distance = self.env.compute_real_distance([chosen_location[0],chosen_location[1]],[ego_location[0][0],ego_location[1][0]])
                
            distance_limit = self.env.map_config['communication_distance_limit']-1
            
            if real_distance < distance_limit:
                self.top_action_sequence += 1
                
        elif self.top_action_sequence == 1:
        
            item_idx = info['object_key_to_index'][str(object_id)]
            
            self.helping_type = self.HelpType.sensing
            
            agent_idx = info['robot_key_to_index'][robot_id]
            if not robotState.get("agents", "type", agent_idx) and False:
                message_to_personalize = MessagePattern.sensing_ask_help(robotState, item_idx, object_id, robot_id, self.env.convert_to_real_coordinates)
                message_to_send = self.human_to_ai_text.personalize_message(message_to_personalize, [robot_id], self.message_history)
                
                if not message_to_send:
                    message_to_send = message_to_personalize
            else:
                message_to_send = MessagePattern.sensing_ask_help(robotState, item_idx, object_id, robot_id, self.env.convert_to_real_coordinates)
            
            self.message_text += message_to_send
            
            self.movement.help_status = self.movement.HelpState.asking
            self.movement.help_status_info[0] = [robot_id]
            self.movement.help_status_info[1] = time.time()
            if not self.action_index == self.movement.State.wait_free and not self.action_index == self.movement.State.wait_random:
                self.movement.last_action_index = self.action_index
            self.action_index = self.movement.State.wait_message
            
            self.top_action_sequence += 1
            
        elif self.top_action_sequence == 2:
            
            #print("Sense request", self.sense_request)
            action["action"] = Action.get_occupancy_map.value
            
            if self.movement.help_status == self.movement.HelpState.being_helped:
            
            
                agent_idx = info['robot_key_to_index'][robot_id]
                
                target_location = robotState.get("agents", "last_seen_location", agent_idx) #robotState.robots[agent_idx]["neighbor_location"]
                ego_location = np.where(robotState.latest_map == 5)

                action,temp_finished,output = self.go_to_location(robot_id, robotState, next_observation, info)
                
                
                real_distance = self.env.compute_real_distance([target_location[0],target_location[1]],[ego_location[0][0],ego_location[1][0]])
                
                distance_limit = self.env.map_config['communication_distance_limit']-1
                
                
                if real_distance < distance_limit:
                    action["action"] = Action.get_occupancy_map.value
                    
                if robotState.get("agents", "disabled", agent_idx) == 1:
                    self.movement.help_status = self.movement.HelpState.no_request
                    print("disabled sensing")
                    finished = True
                
            elif self.movement.help_status == self.movement.HelpState.no_request:
                finished = True
            

                
        
        
        return action,finished,output
    
    def closest_distance_explore(self, robotState, exclude):
    
        ego_location = np.where(robotState.latest_map == 5)
    
        if exclude: #If points get excluded
            modified_map = np.copy(robotState.latest_map)
            agent_view_radius = int(self.env.view_radius)
            for ex in exclude:
            
                x_inf = ex[0] - agent_view_radius
                x_sup = ex[0] + agent_view_radius
                y_inf = ex[1] - agent_view_radius
                y_sup = ex[1] + agent_view_radius
            
                if x_inf < 0:
                    x_inf = 0
                if x_sup > robotState.latest_map.shape[0]:
                    x_sup = robotState.latest_map.shape[0]
                if y_inf < 0:
                    y_inf = 0
                if y_sup > robotState.latest_map.shape[1]:
                    y_sup = robotState.latest_map.shape[1]
                    
               
                modified_map[x_inf:x_sup,y_inf:y_sup] = 0
            
            still_to_explore = np.where(modified_map == -2)
            
            if not len(still_to_explore[0]):
                modified_map = np.copy(robotState.latest_map)
                
                for ex in exclude:
                    modified_map[ex[0],ex[1]] = 0
                
                still_to_explore = np.where(modified_map == -2)
                
        else:
            still_to_explore = np.where(robotState.latest_map == -2)
                
        closest_dist = float('inf')
        closest_idx = -1
    

        for se_idx in range(len(still_to_explore[0])):
            unknown_loc = [still_to_explore[0][se_idx],still_to_explore[1][se_idx]]
            
            unknown_dist = self.env.compute_real_distance(unknown_loc,[ego_location[0][0],ego_location[1][0]])
            
            if unknown_dist < closest_dist:
                closest_dist = unknown_dist
                closest_idx = se_idx
                
        x = still_to_explore[0][closest_idx]
        y = still_to_explore[1][closest_idx]
        
        return x,y
    
    def go_to_location(self, object_id, robotState, next_observation, info):
                
                
        ego_location = np.where(robotState.latest_map == 5)
        
        finished = False
        action = self.sample_action_space
        action["action"] = -1
        action["num_cells_move"] = 1
        
        output = []
        
        """
        if action_sequence == 0:
            action_sequence += 1
            action = Action.get_occupancy_map.value
        """
        
        if object_id == -1: #Return to middle of the room
            x = 10
            y = 10
        elif object_id == -2: #Explore
        
            if not self.explore_location:
            
                x,y = self.closest_distance_explore(robotState, [])
                
                self.explore_location = [x,y]
            else:
                x = self.explore_location[0]
                y = self.explore_location[1]
            
  
        elif str(object_id).isalpha(): #Agent
            
            robot_idx = info['robot_key_to_index'][str(object_id)]
            
            robo_location = robotState.get("agents", "last_seen_location", robot_idx)
            
            if (robo_location[0] == -1 and robo_location[1] == -1):
                action["action"] = Action.get_occupancy_map.value
                return action,True,output
            
            x,y = robo_location
            
        elif isinstance(object_id, list):    
        
            x = object_id[0]
            y = object_id[1]
        else:
            try:
            
                item_idx = info['object_key_to_index'][str(object_id)]
            
                object_location = robotState.get("objects", "last_seen_location", item_idx)
                if (object_location[0] == -1 and object_location[1] == -1):
                    action["action"] = Action.get_occupancy_map.value
                    return action,True,output
            
                x,y = object_location
                
                
            except:
                pdb.set_trace()
        

        try:
            
            low_action, self.next_loc, self.message_text, self.action_index = self.movement.go_to_location(x, y, self.occMap, robotState, info, ego_location, self.action_index, help_sensing=self.helping_type == self.HelpType.sensing)
        except:
            pdb.set_trace()

        if "approach" in self.action_function:
            print(self.next_loc)

        """
        self.path_to_follow = self.movement.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),robotState.latest_map)
        
        if not self.path_to_follow or x == self.path_to_follow[0][0] and y == self.path_to_follow[0][1]:
            action["action"] = Action.get_occupancy_map.value
            finished = True
        else:
        
            next_location = [ego_location[0][0],ego_location[1][0]]
            action["action"] = self.movement.position_to_action(next_location,self.path_to_follow[0],False)
        
            previous_action = ""
            repetition = 1
            action["num_cells_move"] = repetition 
                
        """
        
        if not low_action and isinstance(low_action, list):
            action["action"] = Action.get_occupancy_map.value
            output = [ego_location[0][0],ego_location[1][0]]
            finished = True
        elif low_action < 0:
            possible_path = self.movement.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),robotState.latest_map)
            action["action"] = low_action
            if not possible_path:
                finished = True
                print("TRAPPED")
                #pdb.set_trace()
        else:
            action["action"] = low_action
            
        
                    
        return action,finished,output
        
    def activate_sensor(self,robotState, next_observation, info):

        action = self.sample_action_space
        action["action"] = -1
        finished = False
        output = []


        if self.action_sequence == 0:
            self.action_sequence += 1
            action["action"] = Action.danger_sensing.value
            
        elif self.action_sequence == 1:
            self.item_list = info["last_sensed"]
            #print(item_list)
            self.item_list_dup = self.item_list.copy()
            
            if not self.item_list: #No items scanned
                action["action"] = Action.get_occupancy_map.value
                finished = True
            else:
            
                object_key = self.item_list.pop(0)
                
                action["action"] = Action.check_item.value    
                action["item"] = info["object_key_to_index"][object_key]
                
                if not self.item_list: #item list finished
                    self.action_sequence += 2
                else:
                    self.action_sequence += 1
            
        elif self.action_sequence == 2:
            object_key = self.item_list.pop(0)
            action["action"] = Action.check_item.value    
            action["item"] = info["object_key_to_index"][object_key]
            
          
            if not self.item_list:
                self.action_sequence += 1
           
                
        elif self.action_sequence == 3:
            #[“object id”, “object x,y location”, “weight”, “benign or dangerous”, “confidence percentage”
            for key in self.item_list_dup:
            
                ob_idx = info["object_key_to_index"][key]
            
                if robotState.get("objects", "danger_status", ob_idx) == 1:
                    danger_level = "benign"
                else:
                    danger_level = "dangerous"
                    
                object_location = robotState.get("objects", "last_seen_location", ob_idx)
                output.append([str(key),str(int(object_location[0]))+","+str(int(object_location[1])),str(robotState.get("objects", "weight", ob_idx)),danger_level,str(robotState.get("objects", "estimate_correct_percentage", ob_idx))])
            
            action["action"] = Action.get_occupancy_map.value
        
            finished = True
                
            
            
        """
        elif action_sequence == 1:
            if next_observation['action_status'][2]:
                action["action"] = Action.check_item.value
                action["item"] = item_number
                
                if item_number < len(robotState.items):
                    item_number += 1
                else:
                    action_sequence += 1
                    
        elif action_sequence == 2:
            most_recent = 0
            items_keys = []
            for ri_key in robotState.items:
                robotState.items[ri_key]["item_time"]
        """
        
        return action,finished,output
        
    def send_message(self,message, robotState, next_observation, info):

        action = self.sample_action_space
        action["action"] = -1
        action["robot"] = 0
        finished = True

        output = []
        action["action"] =  Action.send_message.value
        
        
        message_ai = ""
        if re.search(MessagePattern.location_regex(),message): #only if there are humans apply this special scheme
            rematch_str = re.search(MessagePattern.location_regex(),message).group()
            
            if not rematch_str[-1] == ".":
                rematch_str += "."
            
            tmp_message = message
            tmp_message = tmp_message.replace(rematch_str, "")
            
            if tmp_message and not tmp_message.isspace():
                self.message_history.append({"Sender": self.env.robot_id, "Message": tmp_message})
            else:
                tmp_message = ""
            
            if any(not robotState.get("agents", "type", r_idx) for r_idx in self.nearby_other_agents):
                message_ai = rematch_str
                message = tmp_message

        else:
            self.message_history.append({"Sender": self.env.robot_id, "Message": message})

        action["message"] = message
        action["message_ai"] = message_ai

        return action,finished,output
        
    def pick_up(self,object_id,robotState, next_observation, info):
        
        
        action = self.sample_action_space
        action["action"] = -1
        
        ego_location = np.where(robotState.latest_map == 5)

        output = []
        
        finished = False
        
        
        
        if self.action_sequence == 0:
        
            if not robotState.object_held:
            
                self.action_retry = 0
            
                self.action_sequence += 1    

                ob_idx = info["object_key_to_index"][str(object_id)]
             

                if not robotState.get("objects", "weight", ob_idx):
                    output = -2
                    finished = True
                    action["action"] = Action.get_occupancy_map.value
                else:
                    location = robotState.get("objects", "last_seen_location", ob_idx) #robotState.items[ob_idx]["item_location"]
                    action["action"] = self.movement.position_to_action([ego_location[0][0],ego_location[1][0]],location,True)
                    if action["action"] == -1 or (location[0] == -1 and location[1] == -1):
                        action["action"] = Action.get_occupancy_map.value
                        finished = True
                        output = -1
                
            else:
                action["action"] = Action.get_occupancy_map.value
                
                finished = True
                output = -3
            
        elif self.action_sequence == 1:
            if robotState.object_held or self.action_retry == 2:
                action["action"] = Action.get_occupancy_map.value
        
                finished = True
                
                if self.action_retry == 2 and not robotState.object_held:
                    output = -1
                else:
                    self.held_objects = str(object_id)
            else:
                ob_idx = info["object_key_to_index"][str(object_id)]
                location = robotState.get("objects", "last_seen_location", ob_idx) #robotState.items[ob_idx]["item_location"]
                action["action"] = self.movement.position_to_action([ego_location[0][0],ego_location[1][0]],location,True)
                self.action_retry += 1
                
                if action["action"] == -1 or (location[0] == -1 and location[1] == -1):
                    action["action"] = Action.get_occupancy_map.value
                    finished = True
                    output = -1
    
        
            

        return action,finished,output
        
    def report_heavy_dangerous_objects(self, robotState, info):
    
        for ob_key in range(robotState.get_num_objects()): #include heavy dangerous objects to report
            if robotState.get("objects", "weight", ob_key) >= len(self.env.map_config['all_robots'])+2 and robotState.get("objects", "danger_status", ob_key) == 2:
                object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(ob_key)]
                self.message_text += MessagePattern.item(robotState,ob_key,object_id, info, self.env.robot_id, self.env.convert_to_real_coordinates)    
        
    def go_to_meeting_point(self, target_location, robotState, next_observation, info):
    
        action = self.sample_action_space
        action["action"] = -1
        action["num_cells_move"] = 1
        finished = False
        
        ego_location = np.where(robotState.latest_map == 5)

        output = "go_to_meeting_point"
        
        if self.occMap[target_location[0],target_location[1]] == 3 or self.occMap[target_location[0],target_location[1]] == 2 or self.occMap[target_location[0],target_location[1]] == 1:
            finished = True
            action["action"] = Action.get_occupancy_map.value
        elif self.occMap[target_location[0],target_location[1]] == 5:
        
            robot_disabled = robotState.get("agents", "disabled", -1)
        
            print([r == 1 for r in robot_disabled], self.nearby_other_agents)
            if len(self.nearby_other_agents) == robotState.get_num_robots()-sum(r == 1 for r in robot_disabled):
            
                if "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "order":
                    if self.finished: #not any(eval(robotState.get("agents", "team", r)) for r in range(robotState.get_num_robots()) if r not in robot_disabled):
                        self.report_heavy_dangerous_objects(robotState, info)
                        #self.finished = True
                        
                        for oa in self.other_agents:
                            oa.finished = True
                        print([eval(robotState.get("agents", "team", r)) for r in range(robotState.get_num_robots())])
                        self.message_text += MessagePattern.finish()
                        
                        action,_,_ = self.send_message(self.message_text, robotState, next_observation, info)
                        print("SENDING MESSAGE go_meeting_point", info['time'], self.message_text)
                        self.message_text = ""
                        #pdb.set_trace()
                        
                elif "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "obey":
                    pass
                else:
                    if not self.finished or not self.told_to_finish:
                        self.message_text += MessagePattern.finish()
                        
                        self.report_heavy_dangerous_objects(robotState, info)
                                
                        
                        self.finished = True
                        self.told_to_finish = True
            else: #Only when all agents are next to each other finish
                self.told_to_finish = False
                
            finished = True
            
            if action["action"] < 0:
                action["action"] = Action.get_occupancy_map.value
        else:
            low_action, self.next_loc, self.message_text, self.action_index = self.movement.go_to_location(target_location[0], target_location[1], self.occMap, robotState, info, ego_location, self.action_index, help_sensing=self.helping_type == self.HelpType.sensing)

            
            if not low_action and isinstance(low_action, list):
                action["action"] = Action.get_occupancy_map.value
                finished = True
            else:
                action["action"] = low_action
                
             
        return action,finished,output
        
        
    def drop(self,robotState, next_observation, info):

        action = self.sample_action_space
        action["action"] = -1
        finished = True

        output = self.held_objects
        
        action["action"] = Action.drop_object.value

        return action,finished,output
        
    def end_participation(self,robotState, next_observation, info):

        action = self.sample_action_space
        action["action"] = -1
        finished = True

        output = []

        return action,finished,output
        
    def wait(self,robotState, next_observation, info):
        action = self.sample_action_space

        finished = True

        output = []
        
        action["action"] = Action.get_occupancy_map.value

        return action,finished,output
        
    def help(self, agent_id, robotState, next_observation, info):
    
        action = self.sample_action_space
        
        finished = False
        
        output = []
        
        if self.action_sequence == 0:
            self.message_text += MessagePattern.carry_help_accept(agent_id)
            #self.movement.accepted_help = agent_id
            self.movement.help_status_info[0] = [agent_id]
            
            self.action_index = self.movement.State.wait_follow
            self.action_sequence += 1 
            
        elif self.action_sequence == 1:
            finished = True
        
        action["action"] = Action.get_occupancy_map.value
        
        return action,finished,output
        
    def ask_info(self, thing_id, message, robotState, next_observation, info):
    
        action = self.sample_action_space
        
        finished = False
        
        output = []
        
        if self.action_sequence == 0:
            self.message_text += message
            self.message_info[0] = False
            self.message_info[1] = thing_id
            self.message_info[2] = time.time()
            
            self.action_sequence += 1 
            
        elif (self.action_sequence == 1 and self.message_info[0]) or time.time() - self.message_info[2] > self.ask_info_time_limit:
            finished = True
        
        action["action"] = Action.get_occupancy_map.value
        action["robot"] = 0
        
        return action,finished,output
        
        

        
        
    def calculate_neighbor_distance(self, robotState, info, ego_location, excluded):
    
        agents_distance = []
        
        being_helped_indices = []
        if self.movement.help_status == self.movement.HelpState.being_helped:
            being_helped_indices = [info['robot_key_to_index'][agent_id] for agent_id in self.movement.help_status_info[0]]
        
        for r_idx in range(robotState.get_num_robots()):
        
            robo_location = robotState.get("agents", "last_seen_location", r_idx)
        
            if robotState.get("agents", "disabled", r_idx) == 1:
                continue
            elif (robo_location[0] == -1 and robo_location[1] == -1) or r_idx in being_helped_indices or r_idx in excluded:
                robot_distance = float("inf")
            else:
            
                _, next_locs, _, _ = self.movement.go_to_location(robo_location[0], robo_location[1], self.occMap, robotState, info, ego_location, self.action_index, checking=True, help_sensing=self.helping_type == self.HelpType.sensing)
                robot_distance = len(next_locs)
                
            agents_distance.append(robot_distance)
            
        return agents_distance
    
    
    def get_agents_distance(self, pickup_excluded, robotState, info):
    
        ego_location = np.where(robotState.latest_map == 5)
        initial_agents_distance = self.calculate_neighbor_distance(robotState, info, ego_location, pickup_excluded) #Calculate the distances between yourself and the other robots but exclude some of the robots according to roles
        agents_distance = initial_agents_distance.copy()
        cost_agents = []
        
        
        all_robot_combinations = []
        robot_range = []
        
        for i in range(robotState.get_num_robots()):
            if robotState.get("agents", "disabled", i) != 1: #Only take into account those robots that are not disabled
                robot_range.append(i)
        
        for i in range(len(robot_range)): #Compute all possible combinations of robot groups
            all_robot_combinations.extend(combinations([*robot_range, robotState.get_num_robots()], i+1))
        
        '''
        pickup_not_available = []
        for robo_idx in robot_range:
            if (robo_idx in self.help_request_time.keys() and time.time() - self.help_request_time[robo_idx][0] < self.help_request_time[robo_idx][1]): #If an agent has previously been requested help, wait some time until we take it into consideration again.
                pickup_not_available.append(robo_idx)
        '''
        
        distance_excluded = pickup_excluded.copy()
        
        for ag in range(len(robot_range)): #For all robots that can help and are closeby, compute the distance it would take to reach one of them and then the others in sequence
                    
            closest_idx = np.argmin(agents_distance)
                        
            if agents_distance[closest_idx] == float("inf"):
                break
            else:
                cost_agents.append((closest_idx, agents_distance[closest_idx])) #Cost to reach one of the agents in sequence
                
                robo_location = robotState.get("agents", "last_seen_location", robot_range[closest_idx])
                
                next_location = np.array([[robo_location[0]],[robo_location[1]]])
                
                distance_excluded.append(closest_idx)
                agents_distance = self.calculate_neighbor_distance(robotState, info, next_location, distance_excluded)
                
        return initial_agents_distance,all_robot_combinations,robot_range,cost_agents
    
    def cost_carry(self, ob_idx, number_helping, pickup_excluded, robotState, info):
    
        ob_location = robotState.get("objects", "last_seen_location", ob_idx)
        
        if (ob_location[0] == -1 and ob_location[1] == -1) or tuple(ob_location) in self.extended_goal_coords: #If the location of this objects is not known, continue
            return 0,-1
        
        if "interdependency" in self.team_structure:
            if self.team_structure["interdependency"][self.env.robot_id] == "follower":
                followed_id = self.get_closest_robot("interdependency", "followed", robotState, info)
                agent_location = robotState.get("agents", "last_seen_location", info['robot_key_to_index'][str(followed_id)])
                
                if not (agent_location[0] == -1 and agent_location[1] == -1):
                    distance_agent_object = np.linalg.norm(np.array(self.env.convert_to_real_coordinates(ob_location)) - np.array(self.env.convert_to_real_coordinates(agent_location)))
                    
                    if distance_agent_object > 10:
                        return 0,-1
                
            elif self.team_structure["interdependency"][self.env.robot_id] == "followed":
                followed_ids = [tm for tm in self.team_structure["interdependency"].keys() if self.team_structure["interdependency"][tm] == "followed" and tm != self.env.robot_id]
                
                for f in followed_ids:
                
                    agent_location = robotState.get("agents", "last_seen_location", info['robot_key_to_index'][str(f)])
                    if not (agent_location[0] == -1 and agent_location[1] == -1):
                    
                        distance_agent_object = np.linalg.norm(np.array(self.env.convert_to_real_coordinates(ob_location)) - np.array(self.env.convert_to_real_coordinates(agent_location)))
             
                
                        if distance_agent_object <= 10:
                            return 0,-1
                    
            
        
        ob_weight = robotState.get("objects", "weight", ob_idx)
        pickup_args = -1
        request_cost = 0
        
        goal_x = int(self.occMap.shape[0] / 2)
        goal_y = int(self.occMap.shape[1] / 2)
        
        ego_location = np.where(robotState.latest_map == 5)
        
        if ob_weight > 1: #If the object requires greater strength
                        
            _,_,robot_range,cost_agents = self.get_agents_distance(pickup_excluded, robotState, info)
                        
            if len(cost_agents) < ob_weight -1 -number_helping: #If we don't have enough agents to help carry the object, return
                return 0,-1
                        
            if ob_weight-1 <= number_helping: #If we have enough helping agents to carry
                        
                _, next_locs, _, _ = self.movement.go_to_location(ob_location[0], ob_location[1], self.occMap, robotState, info, ego_location, self.action_index, checking=True, help_sensing=self.helping_type == self.HelpType.sensing)
                distance = len(next_locs)
                            
            else: #Otherwise we also have to consider the cost of requesting help
                        
                for ag_idx in range(ob_weight-1 -number_helping): #Add the cost of requesting help from the needed amount of agents
                    request_cost += cost_agents[ag_idx][1]
                    
                    if pickup_args == -1:
                        if "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "order": #we consider the first other leader there is
                            robot_id = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(cost_agents[0][0])]
                            print("This robot_id:",robot_id)
                            if self.team_structure["hierarchy"][robot_id] == "order":
                                pickup_args = cost_agents[0][0]
                            
                        else:
                            pickup_args = cost_agents[0][0] #agent ID of the closest agent       
                
                          
                robo_idx = cost_agents[ag_idx][0] #last agent to ask
                robo_location = robotState.get("agents", "last_seen_location", robot_range[robo_idx])
                next_ego_location = np.array([[robo_location[0]],[robo_location[1]]])
                _, next_locs, _, _ = self.movement.go_to_location(ob_location[0], ob_location[1], self.occMap, robotState, info, next_ego_location, self.action_index, checking=True, help_sensing=self.helping_type == self.HelpType.sensing)
                distance = len(next_locs)
                        
                        
                #print("pickup args", possible_actions[ob_idx]["pickup_args"])
                        
        else: #If we can carry the object alone
            _, next_locs, _, _ = self.movement.go_to_location(ob_location[0], ob_location[1], self.occMap, robotState, info, ego_location, self.action_index, checking=True, help_sensing=self.helping_type == self.HelpType.sensing)
            distance = len(next_locs)
                    
        #if not distance: #If the distance is 0, ignore
        #    print("DISTANCE 0")
        #    return 0,-1
                        
                            
        goal_location = np.array([[goal_x],[goal_y]])
        _, next_locs, _, _ = self.movement.go_to_location(ob_location[0], ob_location[1], self.occMap, robotState, info, goal_location, self.action_index, checking=True, help_sensing=self.helping_type == self.HelpType.sensing)
        return_distance = len(next_locs) #calculate distance to return to goal from object
                   
        pickup_cost = distance + return_distance + request_cost #pickup cost is equal to adding the distance to get to the object + the distances between the agents to whom request help + the distance to move the object to the goal area
    
        return pickup_cost,pickup_args    
        
        
    def cost_sensing(self, ob_idx, sensing_excluded, robotState, info):
    
        ask_sensing_cost = {}
        distance_object_agent = {} 
        not_available = []
        
        ego_location = np.where(robotState.latest_map == 5)
        ob_location = robotState.get("objects", "last_seen_location", ob_idx)
        
        if (ob_location[0] == -1 and ob_location[1] == -1) or tuple(ob_location) in self.extended_goal_coords: #If the location of this objects is not known, continue
            return {}
            
        if "interdependency" in self.team_structure:
            if self.team_structure["interdependency"][self.env.robot_id] == "follower":
                followed_id = self.get_closest_robot("interdependency", "followed", robotState, info)
                agent_location = robotState.get("agents", "last_seen_location", info['robot_key_to_index'][str(followed_id)])
                
                if not (agent_location[0] == -1 and agent_location[1] == -1):
                    distance_agent_object = np.linalg.norm(np.array(self.env.convert_to_real_coordinates(ob_location)) - np.array(self.env.convert_to_real_coordinates(agent_location)))
                    
                    if distance_agent_object > 10:
                        return {}
                
            elif self.team_structure["interdependency"][self.env.robot_id] == "followed":
                followed_ids = [tm for tm in self.team_structure["interdependency"].keys() if self.team_structure["interdependency"][tm] == "followed" and tm != self.env.robot_id]
                
                for f in followed_ids:
                
                    agent_location = robotState.get("agents", "last_seen_location", info['robot_key_to_index'][str(f)])
                    
                    if not (agent_location[0] == -1 and agent_location[1] == -1):
                        distance_agent_object = np.linalg.norm(np.array(self.env.convert_to_real_coordinates(ob_location)) - np.array(self.env.convert_to_real_coordinates(agent_location)))
                    
                        if distance_agent_object <= 10:
                            return {}
            
        initial_agents_distance,all_robot_combinations,robot_range,_ = self.get_agents_distance(sensing_excluded, robotState, info)
        
        for ia_idx,ia in enumerate(initial_agents_distance):
        
            if ia < float("inf"): #If agent is reacheable calculate distance to it
            
                robo_location = robotState.get("agents", "last_seen_location", robot_range[ia_idx])
                next_ego_location = np.array([[robo_location[0]],[robo_location[1]]])
                _, next_locs, _, _ = self.movement.go_to_location(ob_location[0], ob_location[1], self.occMap, robotState, info, next_ego_location, self.action_index, checking=True, help_sensing=self.helping_type == self.HelpType.sensing)
                ask_sense_distance = len(next_locs)
                
                if not ask_sense_distance: #If distance is zero, ignore
                    not_available.append(robot_range[ia_idx])
                    continue
            
                tmp_cost = ia + ask_sense_distance #sensing cost is the addition of going towards the agent and then to the object we are requesting to be sensed # + return_distance
                ask_sensing_cost[robot_range[ia_idx]] = tmp_cost
                distance_object_agent[robot_range[ia_idx]] = ask_sense_distance
            else: #if agent is not reachable
                not_available.append(robot_range[ia_idx])
    
        
        for robo_idx in range(robotState.get_num_estimates(ob_idx)): #Check if we already have an estimate for that object from specific agents
            if robo_idx not in not_available and (robotState.get("object_estimates", "danger_status", [ob_idx,robo_idx]) != 0): # or robo_idx in pickup_not_available): # or (robo_idx in self.sense_request_time.keys() and time.time() - self.sense_request_time[robo_idx][0] < self.sense_request_time[robo_idx][1]):
                not_available.append(robo_idx)
                    
        if robotState.get("object_estimates", "danger_status", [ob_idx,robotState.get_num_estimates(ob_idx)-1]) != 0: #Check also if we already sensed the objects ourselve
            not_available.append(robotState.get_num_robots())
        
        sensing_combs = {}
        
        actual_combinations = []
        
        #print("Not available", not_available)
        for comb in all_robot_combinations: #For all possible sequences where we move sequentially to a subset of the agents
            comb_cost = 0
            
            if not any(el in not_available or el in sensing_excluded for el in comb): #If no agent in that subset has not been excluded
                actual_combinations.append(comb)
                if not (len(comb) == 1 and robotState.get_num_robots() in comb): #If it's not us only the ones in that combination
                    
                    for elem_idx,elem in enumerate(comb):
                    
                        if elem == robotState.get_num_robots():
                            continue
                            
                        if not elem_idx: 
                            try:
                                comb_cost += ask_sensing_cost[elem] #Add the costs of going towards that robot and requesting help
                            except:
                                pdb.set_trace()
                        else:
                            comb_cost += distance_object_agent[elem]*2 #Add the cost of reaching towards that agent
                
                else:
                    _, next_locs, _, _ = self.movement.go_to_location(ob_location[0], ob_location[1], self.occMap, robotState, info, ego_location, self.action_index, checking=True, help_sensing=self.helping_type == self.HelpType.sensing)
                    distance = len(next_locs)
                    comb_cost = distance    
                #comb_cost += pickup_cost #Add the cost of actually picking the object
                sensing_combs[comb] = comb_cost
                
        return sensing_combs
    
    def utility_calculation_carry(self, ob_idx, collab_score, cost, robotState):
    
        ob_danger = robotState.get("objects", "danger_status", ob_idx)
        utility = 0
        
        if ob_danger:#If we already have an estimation of the danger level of the object, create the decision network to compute the costs of picking that object 
                
                
            """
            diag = gum.InfluenceDiagram()

            Estimate=diag.addChanceNode(gum.LabelizedVariable("Estimate","Estimate",2))
            Pickup=diag.addDecisionNode(gum.LabelizedVariable("Pickup","Pickup",2))
            Pickup_Utility=diag.addUtilityNode(gum.LabelizedVariable("Pickup_Utility","Pickup_Utility",1))
            Pickup_Cost=diag.addUtilityNode(gum.LabelizedVariable("Pickup_Cost","Pickup_Cost",1))


            diag.addArc(Estimate,Pickup)
            diag.addArc(Estimate,Pickup_Utility)
            diag.addArc(Pickup,Pickup_Utility)
            diag.addArc(Pickup,Pickup_Cost)
            
            """

            danger_percentage = 0.50
            ob_danger_estimate = robotState.get("objects", "estimate_correct_percentage", ob_idx)

            if ob_danger == 0:
                #diag.cpt(Estimate).fillWith([0.70, 0.30])
                pass
            elif ob_danger == 1:
                confidence = ob_danger_estimate
                #diag.cpt(Estimate).fillWith([confidence, 1-confidence])
                danger_percentage = 1-confidence
            elif ob_danger == 2:
                confidence = ob_danger_estimate
                #diag.cpt(Estimate).fillWith([1-confidence, confidence])
                danger_percentage = confidence
                
            """
            #diag.utility(Pickup_Utility)[{'Pickup':0}] = [[100],[0]]
            diag.utility(Pickup_Utility)[{'Pickup':0}] = [[0],[0]]
            #diag.utility(Pickup_Utility)[{'Pickup':1}] = [[0],[100]]
            diag.utility(Pickup_Utility)[{'Pickup':1}] = [[0],[1]]

            diag.utility(Pickup_Cost)[{'Pickup':0}] = 0
            diag.utility(Pickup_Cost)[{'Pickup':1}] = 0#-possible_actions[ob_idx]["pickup"]

            ie=gum.ShaferShenoyLIMIDInference(diag)
            ie.addEvidence('Pickup',1)
            ie.makeInference() 
            
            """
            
            if danger_percentage >= 0.7:
            
                #utility["pickup_" + str(ob_idx)] = 100 - (1-ie.MEU()["mean"])*possible_actions[ob_idx]["pickup"]
                utility = 100 - ((1-danger_percentage)+(1-collab_score/10))/2*cost
                
        return utility
        
    def utility_calculation_sensing(self, ob_idx, s_comb, cost, robotState):
    
        utility = 0
        
        collab_score = 0
        for ie_idx in s_comb:
            collab_score += robotState.get("agents", "collaborative_score", int(ie_idx))


        collab_score /= len(s_comb)
        """
    
        diag = gum.InfluenceDiagram()

        Estimate_Benign=diag.addChanceNode(gum.LabelizedVariable("Estimate_Benign","Estimate_Benign",2))
        Estimate_Dangerous=diag.addChanceNode(gum.LabelizedVariable("Estimate_Dangerous","Estimate_Dangerous",2))
        Pickup_2=diag.addDecisionNode(gum.LabelizedVariable("Pickup_2","Pickup_2",2))
        Pickup_Utility_2=diag.addUtilityNode(gum.LabelizedVariable("Pickup_Utility_2","Pickup_Utility_2",1))
        Pickup_Cost_2=diag.addUtilityNode(gum.LabelizedVariable("Pickup_Cost_2","Pickup_Cost_2",1))


        diag.addArc(Estimate_Benign,Pickup_2)
        diag.addArc(Estimate_Dangerous,Pickup_2)
        diag.addArc(Estimate_Benign,Pickup_Utility_2)
        diag.addArc(Estimate_Dangerous,Pickup_Utility_2)
        diag.addArc(Pickup_2,Pickup_Utility_2)
        diag.addArc(Pickup_2,Pickup_Cost_2)
        
        """
        
        prior_benign = 0.5
        prior_dangerous = 0.5
        
        if robotState.get("objects", "danger_status", ob_idx) == 1:
            prior_dangerous = 1-robotState.get("objects", "estimate_correct_percentage", ob_idx)
            prior_benign = robotState.get("objects", "estimate_correct_percentage", ob_idx)
        elif robotState.get("objects", "danger_status", ob_idx) == 2:
            prior_dangerous = robotState.get("objects", "estimate_correct_percentage", ob_idx)
            prior_benign = 1-robotState.get("objects", "estimate_correct_percentage", ob_idx)
        
        
        """
        for item_danger_level in [1,2]:
            for ie_idx in s_comb:
                
                
                    
                if ie_idx == len(robotState.robots):
                    if item_danger_level == 2:
                        benign = 1-robotState.sensor_parameters[0]
                        dangerous = robotState.sensor_parameters[1]
                    elif item_danger_level == 1:
                        benign = robotState.sensor_parameters[0]
                        dangerous = 1-robotState.sensor_parameters[1]
                            
                else:
                    if item_danger_level == 2:
                        benign = 1-robotState.neighbors_sensor_parameters[ie_idx][0]
                        dangerous = robotState.neighbors_sensor_parameters[ie_idx][1]
                    elif item_danger_level == 1:
                        benign = robotState.neighbors_sensor_parameters[ie_idx][0]
                        dangerous = 1-robotState.neighbors_sensor_parameters[ie_idx][1]
                
                prob_evidence = (prior_benign*benign + prior_dangerous*dangerous)
                    
                prior_benign_temp = benign*prior_benign/prob_evidence
                prior_dangerous_temp = dangerous*prior_dangerous/prob_evidence
                
                if item_danger_level == 1:
                    prior_benign = prior_benign_temp
                    prior_dangerous = 1-prior_benign
                elif item_danger_level == 2:
                    prior_dangerous = prior_dangerous_temp
                    prior_benign = 1-prior_dangerous
            
            if item_danger_level == 1:        
                benign_est = prior_benign
            elif item_danger_level == 2:
                danger_est = prior_dangerous

        """
        
        def sequence_sensing(s_comb, prior_benign, prior_dangerous):
        
            results = []
        
            for ie_idx in s_comb:
                for item_danger_level in [1,2]:
                    if ie_idx == robotState.get_num_robots():
                        if item_danger_level == 2:
                            benign = 1-robotState.env.sensor_parameters[0]
                            dangerous = robotState.env.sensor_parameters[1]
                        elif item_danger_level == 1:
                            benign = robotState.env.sensor_parameters[0]
                            dangerous = 1-robotState.env.sensor_parameters[1]
                                
                    else:
                        if item_danger_level == 2:
                            benign = 1-robotState.env.neighbors_sensor_parameters[ie_idx][0]
                            dangerous = robotState.env.neighbors_sensor_parameters[ie_idx][1]
                        elif item_danger_level == 1:
                            benign = robotState.env.neighbors_sensor_parameters[ie_idx][0]
                            dangerous = 1-robotState.env.neighbors_sensor_parameters[ie_idx][1]
                            
                    prob_evidence = (prior_benign*benign + prior_dangerous*dangerous)
                    
                    if item_danger_level == 1:
                        prior_benign_tmp = benign*prior_benign/prob_evidence
                        prior_dangerous_tmp = 1-prior_benign_tmp
                    elif item_danger_level == 2:
                        prior_dangerous_tmp = dangerous*prior_dangerous/prob_evidence
                        prior_benign_tmp = 1-prior_dangerous_tmp
                    
                    if len(s_comb) > 1:
                        results.extend(sequence_sensing(s_comb[1:], prior_benign_tmp, prior_dangerous_tmp))
                    else:    
                        results.append([prior_benign_tmp,prior_dangerous_tmp])   
            
                    
            return results             
                        
                
        sensing_results = sequence_sensing(s_comb, prior_benign, prior_dangerous)
        
        danger_est = sensing_results[1][1]
        benign_est = sensing_results[0][0]
        
        sensing_results_np = np.array(sensing_results)
        #print(np.average(sensing_results_np,axis=0),np.var(sensing_results_np,axis=0)) 
        
        #if len(s_comb) > 2:
        #    pdb.set_trace()
        
        """
        #benign_est = robotState.possible_estimates[ob_idx][s_comb[0]][0]
        diag.cpt(Estimate_Benign).fillWith([benign_est, 1-benign_est])
        #danger_est = robotState.possible_estimates[ob_idx][s_comb[0]][1]
        diag.cpt(Estimate_Dangerous).fillWith([1-danger_est, danger_est])

        diag.utility(Pickup_Utility_2)[{'Pickup_2':0, 'Estimate_Benign':0}] = [[100],[0]]
        diag.utility(Pickup_Utility_2)[{'Pickup_2':0, 'Estimate_Benign':1}] = [[0],[0]]
        diag.utility(Pickup_Utility_2)[{'Pickup_2':1, 'Estimate_Benign':0}] = [[0],[0]]
        diag.utility(Pickup_Utility_2)[{'Pickup_2':1, 'Estimate_Benign':1}] = [[0],[100]]

        diag.utility(Pickup_Cost_2)[{'Pickup_2':0}] = 0
        diag.utility(Pickup_Cost_2)[{'Pickup_2':1}] = 0#-possible_actions[ob_idx]["sensing"][s_comb]

        ie=gum.ShaferShenoyLIMIDInference(diag)
        #ie.addEvidence('Pickup_2',1)
        ie.makeInference()
        
        """
        
        #utility["sense_" + str(ob_idx) + "_" + utility_str] = 100 - (1-(abs(benign_est - (1-danger_est)) + (abs(danger_est - (1-danger_est)) + abs(benign_est - (1-benign_est)))/2)/2)*possible_actions[ob_idx]["sensing"][s_comb] #ie.MEU()["mean"]
        utility = 100 - ((1-(min(1,np.average(sensing_results_np,axis=0)[1] + np.std(sensing_results_np,axis=0)[1]))) + (1-collab_score/10))/2*cost
        #utility["sense_" + str(ob_idx) + "_" + utility_str] = 100 - (1-(min(1,prior_dangerous + np.std(sensing_results_np,axis=0)[1])))*possible_actions[ob_idx]["sensing"][s_comb]
        
        #print("Estimations", s_comb, danger_est, benign_est)
        
        #if utility_str == "4":
        #    pdb.set_trace()
        
        return utility
        
     
    def agents_exclusion(self, preexcluded, robotState, info):
    
        #Exclude requests to certain agents according to role
        pickup_excluded = []
        sensing_excluded = []
                  

        already_requested = False        
        for teammate_idx in range(robotState.get_num_robots()): #If an agent has previously been requested help, wait some time until we take it into consideration again
            
            if teammate_idx in self.help_request_time.keys() and time.time() - self.help_request_time[teammate_idx][0] < self.help_request_time[teammate_idx][1]: 
                if teammate_idx not in pickup_excluded:
                    pickup_excluded.append(teammate_idx)
                        
                if teammate_idx not in sensing_excluded:
                    sensing_excluded.append(teammate_idx)
                    
                    
                already_requested = True
                
            if "role" in self.team_structure: #Exclude according to the roles they assume
            
                teammate = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(teammate_idx)]
            
                if self.team_structure["role"][teammate] == "sensing" and teammate_idx not in pickup_excluded: #If you are only sensing, you cannot ask others to pickup objects. 
                    pickup_excluded.append(teammate_idx)
                elif self.team_structure["role"][teammate] == "lifter" and teammate_idx not in sensing_excluded: #If you are only lifting, you cannot ask others to sense objects
                    sensing_excluded.append(teammate_idx)
                    
            if teammate_idx in preexcluded:
                 sensing_excluded.append(teammate_idx)
                 pickup_excluded.append(teammate_idx)
                
        if already_requested: #Include everyone if already requested help
            for teammate_idx in range(robotState.get_num_robots()):
                if teammate_idx not in pickup_excluded:
                    pickup_excluded.append(teammate_idx)
                        
                if teammate_idx not in sensing_excluded:
                    sensing_excluded.append(teammate_idx)
                
                   
                
        if "role" in self.team_structure: 
            if self.team_structure["role"][self.env.robot_id] == "lifter": #If you are only lifting, exclude yourself from sensing
                sensing_excluded.append(robotState.get_num_robots())   
            elif self.team_structure["role"][self.env.robot_id] == "sensing": #If you are only sensing, exclude yourself from lifting
                pickup_excluded.append(robotState.get_num_robots())
            
        
        if robotState.get_num_robots() in preexcluded: #Exclude also yourself
        
            teammate_idx = robotState.get_num_robots()
        
            if teammate_idx not in pickup_excluded:
                pickup_excluded.append(teammate_idx)
                        
            if teammate_idx not in sensing_excluded:
                sensing_excluded.append(teammate_idx)
        
        return pickup_excluded,sensing_excluded
        
    def calculate_overall_utility(self, preexcluded, robotState, info):
    

        if self.movement.help_status == self.movement.HelpState.being_helped: #if we are already being helped, get the number of helper agents
            number_helping = len(self.movement.help_status_info[0])
        else:
            number_helping = 0

        possible_actions = {}
        utility = {}
        
        #initial_agents_distance = self.calculate_neighbor_distance(robotState, info, ego_location, []) #Calculate the distances between yourself and the other robots
        
        pickup_excluded,sensing_excluded = self.agents_exclusion(preexcluded, robotState, info)   
                
                    
        #initial_agents_distance,all_robot_combinations,robot_range,cost_agents = self.get_agents_distance(pickup_excluded, robotState, info)

        
        
        for ob_idx in range(robotState.get_num_objects()): #For all possible objects
        
        
            collab_score = 10
            try:
                object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(ob_idx)]
            except:
                pdb.set_trace()
        
            if object_id in self.carried_objects.values(): #If this object is being carried by someone else continue with the others
                continue
        
            possible_actions[ob_idx] = {}
            
            
            ob_location = robotState.get("objects", "last_seen_location", ob_idx)
            
            
            
            if (ob_location[0] == -1 and ob_location[1] == -1) or tuple(ob_location) in self.extended_goal_coords: #If the location of this objects is not known, continue
                continue
            else:
                
                pickup_cost,pickup_args = self.cost_carry(ob_idx, number_helping, pickup_excluded, robotState, info)
            
                if not pickup_cost:
                    continue
            
                if pickup_args >= 0:
                    possible_actions[ob_idx]["pickup_args"] = pickup_args
                    collab_score = robotState.get("agents", "collaborative_score", int(pickup_args))
                
                if not ("role" in self.team_structure and self.team_structure["role"][self.env.robot_id] == "sensing"): #If you are not sensing, a possible action can be to pickup this object #not ("pickup_args" in possible_actions[ob_idx] and possible_actions[ob_idx]["pickup_args"] in pickup_not_available) and not ("role" in self.team_structure and self.team_structure["role"][self.env.robot_id] == "sensing"):
                    possible_actions[ob_idx]["pickup"] = pickup_cost
                
                
                possible_actions[ob_idx]["sensing"] = self.cost_sensing(ob_idx, sensing_excluded, robotState, info)
                
                if "pickup" in possible_actions[ob_idx]:
                
                    
                    utility_calculation = self.utility_calculation_carry(ob_idx, collab_score, possible_actions[ob_idx]["pickup"], robotState)
                    
                    if utility_calculation:
                        utility["pickup_" + str(ob_idx)] = utility_calculation
                
                
                
                for s_comb in possible_actions[ob_idx]["sensing"].keys(): #Now for sensing the object compute the decision network 
                
                    if s_comb:
                    
                        utility_calculation = self.utility_calculation_sensing(ob_idx, s_comb, possible_actions[ob_idx]["sensing"][s_comb], robotState)
                    
                        if utility_calculation:
                        
                            utility_str = ""
    
                            for s_idx in range(len(s_comb)):
                                if not s_idx:
                                    utility_str += str(s_comb[s_idx])
                                else:
                                    utility_str += "_" + str(s_comb[s_idx])
                        
                            utility["sense_" + str(ob_idx) + "_" + utility_str] = utility_calculation
         
                            
                      
        
        #pdb.set_trace()
        
        if self.team_structure["role"][self.env.robot_id] != "lifter" and not ("interdependency" in self.team_structure and self.team_structure["interdependency"][self.env.robot_id] == "follower"): #exploration
            agent_view_radius = int(self.env.view_radius)
            if "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "order":
                num_explore = 0
                for r in range(robotState.get_num_robots()+1):
                    
                    if r not in preexcluded:
                        if r < robotState.get_num_robots():
                            robot_id = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(r)]
                            if not self.team_structure["hierarchy"][robot_id] == "order":
                                num_explore += 1
                        else:
                            num_explore += 1
                            
            
            else:
                num_explore = 1
            
            for n in range(num_explore):
                unexplored = np.where(robotState.latest_map == -2)
                
                unexplored_size = unexplored[0].size - n*agent_view_radius*2
                explored = round((robotState.latest_map.size-unexplored_size)/robotState.latest_map.size,2) #calculate the utility of exploring the area
                
                if explored < 1:
                    utility["explore_" + str(n)] = math.exp(-2*explored)*100
            
        utility['end'] = ((pow(100,info['time']/self.env.map_config['timer_limit']) - 1)/(100-1))*100
        
        
        return possible_actions,utility
        
    def get_closest_robot(self, team_structure_category, leader_type, robotState, info):
    
        leaders = [tm for tm in self.team_structure[team_structure_category].keys() if self.team_structure[team_structure_category][tm] == leader_type]
    
        ego_location = np.where(robotState.latest_map == 5)
    
        leader_min_distance = [leaders[0],float("inf")]
        leader_id = leader_min_distance[0]
        
        if len(leaders) > 1:
            for leader in leaders: #Choose the closest leader to reportback
            
                chosen_location = robotState.get("agents", "last_seen_location", info['robot_key_to_index'][str(leader)]) #robotState.robots[info['robot_key_to_index'][str(self.leader_id)]]["neighbor_location"] #We are missing leader id
                
                if (chosen_location[0] == -1 and chosen_location[1] == -1):
                    continue
                else:
                    real_distance = self.env.compute_real_distance([chosen_location[0],chosen_location[1]],[ego_location[0][0],ego_location[1][0]])  
                    
                    if real_distance <  leader_min_distance[1]:
                        leader_min_distance = [leader,real_distance]     
            
            leader_id = leader_min_distance[0]
            
        else:
            leader_id = self.leaders[0]
            
        
        return leader_id
            
    def decision_obey(self,messages, robotState, info, output, nearby_other_agents, next_observation):
    
        
    
        if self.order_status == self.OrderStatus.ongoing:
            self.order_status = self.OrderStatus.reporting_output
            
            self.order_status_info = []
            if output:
                self.order_status_info = [self.action_function, output]
            

        ego_location = np.where(robotState.latest_map == 5)
        
        
        if not self.leader_id:
        
        
            self.leader_id = self.get_closest_robot("hierarchy", "order", robotState, info)
            chosen_location = robotState.get("agents", "last_seen_location", info['robot_key_to_index'][str(self.leader_id)])
                
        else:
            chosen_location = robotState.get("agents", "last_seen_location", info['robot_key_to_index'][str(self.leader_id)])
            
        if self.movement.help_status == self.movement.HelpState.being_helped:
            _,self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text, message=MessagePattern.carry_help_finish())

            
        if (chosen_location[0] == -1 and chosen_location[1] == -1): #if there is no agent in the correct place
            if self.nearby_other_agents:
                action,temp_finished,_ = self.ask_info(self.leader_id, MessagePattern.ask_for_agent(self.leader_id), robotState, next_observation, info)
                function_output = "ask_info('" + self.leader_id + "','" + MessagePattern.ask_for_agent(self.leader_id) + "')"
                
            else:
                true_ending_locations = [loc for loc in self.ending_locations if self.occMap[loc[0],loc[1]] == 0 or self.occMap[loc[0],loc[1]] == -2]
                
                target_location = random.choice(true_ending_locations)  
                if [ego_location[0][0],ego_location[1][0]] in self.ending_locations: #If we are already in the ending locations just stay there
                    target_location = [ego_location[0][0],ego_location[1][0]]
                    
                  
                function_output = "go_to_meeting_point(" + str(target_location) + ")"   
                
        else:
            real_distance = self.env.compute_real_distance([chosen_location[0],chosen_location[1]],[ego_location[0][0],ego_location[1][0]])
                
            distance_limit = self.env.map_config['communication_distance_limit']-1
            print(real_distance)
            if real_distance < distance_limit:
                function_output = "wait()"
                
                if self.order_status == self.OrderStatus.reporting_output:
                    if self.order_status_info:
                        if "sense_object" in self.order_status_info[0]:
                        
                            """
                            most_recent = 0
                            for ob in self.order_status_info[1]:
                                object_idx = info["object_key_to_index"][ob[0]]
                                this_time = robotState.items[object_idx]["item_time"][0]
                                if this_time > most_recent:
                                    most_recent = this_time
                            """
                            
                            for ob in self.order_status_info[1]:
                                object_idx = info["object_key_to_index"][ob[0]]
                                #this_time = robotState.items[object_idx]["item_time"][0]
                                #if this_time >= most_recent:
                                
                                if ob[0] not in self.other_agents[info['robot_key_to_index'][str(self.leader_id)]].items_info_provided:
                                    self.message_text += MessagePattern.item(robotState,object_idx,ob[0], info, self.env.robot_id, self.env.convert_to_real_coordinates)
                                    self.other_agents[info['robot_key_to_index'][str(self.leader_id)]].items_info_provided.append(ob[0])
                                    
                        elif "collect_object" in self.order_status_info[0]:
                            try:
                                object_idx = info["object_key_to_index"][self.order_status_info[1]]
                            except:
                                pdb.set_trace()
                            self.message_text += MessagePattern.item(robotState,object_idx,self.order_status_info[1], info, self.env.robot_id, self.env.convert_to_real_coordinates)
                        elif "go_to_location" in self.order_status_info[0]:
                            self.message_text += MessagePattern.surroundings(self.order_status_info[1], int(self.env.view_radius), robotState, info, self.env.convert_to_real_coordinates)
                    self.order_status = self.OrderStatus.reporting_availability
                elif self.order_status == self.OrderStatus.reporting_availability:
                    self.message_text += MessagePattern.order_finished() #This should only be sent once
                    self.order_status = self.OrderStatus.finished
                    self.leader_id = ""
            else:
                function_output = "approach('" + self.leader_id + "')"
    
        return function_output
    
    
    def sense_equals(self, max_utility_key, robotState, info):
    
        sense_args = max_utility_key.split("_")
        object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(int(sense_args[1]))]
        
        if self.movement.help_status == self.movement.HelpState.being_helped and self.helping_type == self.HelpType.carrying: #Cancel cooperation if previous help was from picking an object
            _,self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text, message=MessagePattern.carry_help_finish())
            
        
        if int(sense_args[2]) == robotState.get_num_robots(): #If we don't need help, just sense object
            function_output = "sense_object('" + object_id + "'," + str([]) + ")"
        else: #If we need help, go towards agent and ask for help
            agent_id = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(int(sense_args[2]))]
            function_output = "ask_for_sensing('" + object_id + "','" + agent_id + "')"
            wait_time = random.randrange(self.non_request,self.non_request+20)
            self.help_request_time[int(sense_args[2])] = [time.time(), wait_time] #We will not be able to request help again from this agent until some time passes
            
            
        return function_output
        
        
    def carry_equals(self, max_utility_key, possible_actions, robotState, info):
    
        pickup_args = max_utility_key.split("_")
                
        ob_idx = int(pickup_args[1])
        object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(ob_idx)]
        
        if self.movement.help_status == self.movement.HelpState.being_helped: #If already being helped
        
            if self.helping_type == self.HelpType.sensing: #Cancel help if we requested help only for sensing
                _,self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text, message=MessagePattern.carry_help_finish())
            elif self.helping_type == self.HelpType.carrying and len(self.movement.help_status_info[0]) > robotState.get("objects", "weight", ob_idx)-1: #If we have more agents than needed, reject some of them
                remove = len(self.movement.help_status_info[0]) - (robotState.get("objects", "weight", ob_idx)-1)
                
                for r in range(remove-1,-1,-1):
                    self.message_text += MessagePattern.carry_help_reject(self.movement.help_status_info[0][r])
                    del self.movement.help_status_info[0][r]
        
        print("OBJECT TO CARRY:", robotState.get("objects", "danger_status", ob_idx), robotState.get("objects", "estimate_correct_percentage", ob_idx))
        
        if robotState.get("objects", "weight", ob_idx) == 1 or (robotState.get("objects", "weight", ob_idx) > 1 and self.movement.help_status == self.movement.HelpState.being_helped and len(self.movement.help_status_info[0]) == robotState.get("objects", "weight", ob_idx)-1): #If we can pickup alone the object or we already have enough agents helping
            function_output = "collect_object('" + object_id + "')"
        else: #Otherwise let's ask for help
            try:
                robot_id = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(possible_actions[ob_idx]["pickup_args"])]
            except:
                pdb.set_trace()
            function_output = "ask_for_help('" + object_id + "','" + robot_id + "')"
            wait_time = random.randrange(self.non_request,self.non_request+20)
            self.help_request_time[possible_actions[ob_idx]["pickup_args"]] = [time.time(), wait_time]
            
        return function_output
    
    def decision(self,messages, robotState, info, output, nearby_other_agents, help_requests):


        utility = {}
        preexcluded = []
        function_output = ""
        
        leader =  "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "order"
        
        if leader:
            for r in range(robotState.get_num_robots()+1):
                if (eval(robotState.get("agents", "team", r)) and not (r == robotState.get_num_robots() and (not self.agent_requesting_order))) or (r not in self.nearby_other_agents and r != robotState.get_num_robots()):
                    preexcluded.append(r)
        
            print("Excluded:", preexcluded)
            
            #if robotState.get_num_robots() in preexcluded:
            #    pdb.set_trace()
            
        if self.movement.help_status != self.movement.HelpState.accepted or ("hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "order"):
            possible_actions,utility = self.calculate_overall_utility(preexcluded, robotState, info)
        else: #If there is no possible action to do, just wait
            utility["wait"] = 100
        
        
        ego_location = np.where(robotState.latest_map == 5)
        
        #max_utility = 0
        max_utility_key = ""
        
        """
        for uk in utility.keys(): #get the action with the highest utility
            if utility[uk] > max_utility: #and uk != self.past_decision:
                max_utility = utility[uk]
                max_utility_key = uk
        """
        
        #pdb.set_trace()
        sorted_utility = sorted(utility.items(), key=lambda item: item[1])
        sorted_utility.reverse()
        
        
        if leader:
        
            print("UTILITY:", sorted_utility)
            self.agent_requesting_order = False
        
            available = [r for r in range(robotState.get_num_robots()+1) if r not in preexcluded]
            
            location_exclude = []

            for u in sorted_utility:
            
                if not available:
                    break
            
                key_parts = u[0].split("_")
            
                if key_parts[0] == "sense" or key_parts[0] == "pickup":
                
                    if key_parts[0] == "sense":
                        participants = key_parts[2:]
                    else:
                        weight = robotState.get("objects", "weight", int(key_parts[1]))
                        
                        if len(available) < weight:
                            continue
                        
                        #if weight > 2:
                        #    pdb.set_trace()
                        participants = available[:weight].copy()
                        
                        obj_pick = int(key_parts[1])
                        
                        if obj_pick not in self.collect_attempts:
                            self.collect_attempts[obj_pick] = 0
                        
                        
                        if self.collect_attempts[obj_pick] >= 3:
                            print("already max attempts")
                            #continue
                                
                        self.collect_attempts[obj_pick] += 1
                    
                    if all(int(p) in available for p in participants):
                        
                        other_robots_ids = []
                        leader_present = False
                        for p_idx, p in enumerate(participants):
                        
                            available.remove(int(p)) 
                            
                            if int(p) == robotState.get_num_robots():
                                robot_id = self.env.robot_id
                                leader_present = True
                            else:
                                robot_id = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(int(p))]

                            
                            if "sense" in u[0]:

                                if robot_id != self.env.robot_id and self.team_structure["hierarchy"][robot_id] == "obey":
                                    location = robotState.get("objects", "last_seen_location", int(key_parts[1])) #robotState.items[int(key_parts[1])]["item_location"]
                                    object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(int(key_parts[1]))]
                                    
                                    self.message_text += MessagePattern.order_sense(robot_id, object_id, location, self.env.convert_to_real_coordinates)
                                elif robot_id != self.env.robot_id and self.team_structure["hierarchy"][robot_id] == "order":
                                    max_utility_key = key_parts[0] + "_" + key_parts[1] + "_" + p
                                    function_output = self.sense_equals(max_utility_key, robotState, info)
                                    break
                                else: #if it's the leader
                                    object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(int(key_parts[1]))]
                                    function_output = "sense_object('" + object_id + "'," + str([]) + ")"
                                    
                                if p_idx == len(participants)-1:
                                    for p in participants:
                                        if not (robot_id != self.env.robot_id and self.team_structure["hierarchy"][robot_id] == "order"):
                                            robotState.set("agents", "team", int(p), str([int(p) for p in participants]), info["time"])
                                            
                            elif "pickup" in u[0]:
                                
                                object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(int(key_parts[1]))]
                                
                                if not p_idx:
                                    self.message_text += MessagePattern.item(robotState,int(key_parts[1]),object_id, info, self.env.robot_id, self.env.convert_to_real_coordinates)
                                
                                if len(participants) == 1:
                                    if not leader_present and self.team_structure["hierarchy"][robot_id] == "obey":
                                        self.message_text += MessagePattern.order_collect(robot_id, object_id)
                                        robotState.set("agents", "team", int(p), str([int(p)]), info["time"])
                                    elif not leader_present and self.team_structure["hierarchy"][robot_id] == "order":
                                        function_output = self.carry_equals(u[0], possible_actions, robotState, info)
                                    else:
                                        function_output = "collect_object('" + object_id + "')"
                                        robotState.set("agents", "team", int(p), str([int(p)]), info["time"])
                                else:
                                
                                    if robot_id != self.env.robot_id and self.team_structure["hierarchy"][robot_id] == "order":
                                        ob_idx = int(key_parts[1])
                                        if "pickup_args" in possible_actions[ob_idx]: #agent is actually not available
                                            function_output = self.carry_equals(u[0], possible_actions, robotState, info)
                                            if "ask_for_help" in function_output:
                                                break
                                        else:
                                            break
                                    
                                    if p_idx == len(participants)-1:
                                    
                                        if leader_present:
                                            
                                            if robot_id != self.env.robot_id:
                                                other_robots_ids.append(robot_id)
                                                robot_id = self.env.robot_id
                                                
                                            function_output = "collect_object('" + object_id + "')"
                                            self.movement.help_status = self.movement.HelpState.being_helped
                                            self.movement.help_status_info[0].extend(other_robots_ids)
                                            self.movement.help_status_info[2] = []
                                            
                                        self.message_text += MessagePattern.agent(robot_id, int(p), robotState, self.env.convert_to_real_coordinates)
                                        self.message_text += MessagePattern.order_collect_group(robot_id, other_robots_ids, object_id)
                                        
                                        for p in participants:
                                            if not (robot_id != self.env.robot_id and self.team_structure["hierarchy"][robot_id] == "order"):
                                                robotState.set("agents", "team", int(p), str([int(p) for p in participants]), info["time"])
                                        
                                    elif not (robot_id != self.env.robot_id and self.team_structure["hierarchy"][robot_id] == "order"):
                                        if robot_id != self.env.robot_id:
                                            other_robots_ids.append(robot_id)
                                    
                elif "explore" in u[0]:
                            
                    robot_idx = -1
                    
                    for a in available:
                        if int(a) != robotState.get_num_robots():
                            robot_id = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(int(a))]
                            
                            if self.team_structure["hierarchy"][robot_id] == "obey":
                                robot_idx = a
                                available.remove(a)
                                break
                        else:
                            robot_idx = a
                            available.remove(a)
                            break
                            
                    #robot_idx = available.pop() 
                    
                    if robot_idx == -1:
                        continue
                    
                    try:
                        robotState.set("agents", "team", robot_idx, str([robot_idx]), info["time"])
                    except:
                        pdb.set_trace()
                            
                    location = self.closest_distance_explore(robotState, location_exclude)
                    
                    location_exclude.append(location)
                    
                    
                    if robot_idx != robotState.get_num_robots():         
                        robot_id = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(robot_idx)]
                        self.message_text += MessagePattern.order_explore(robot_id, location, self.env.convert_to_real_coordinates)
                    else:
                        function_output = "explore()"
                
                else: #End
                    break
                
            if not function_output and robotState.get_num_robots() in available:
            
                if not preexcluded:
                    self.finished = True
                    print("Preparing to finish!")
            
                true_ending_locations = [loc for loc in self.ending_locations if self.occMap[loc[0],loc[1]] == 0 or self.occMap[loc[0],loc[1]] == -2]
                
                target_location = random.choice(true_ending_locations)  
                if [ego_location[0][0],ego_location[1][0]] in self.ending_locations: #If we are already in the ending locations just stay there
                    target_location = [ego_location[0][0],ego_location[1][0]]
                    
                robotState.set("agents", "team", robotState.get_num_robots(), "[]", info["time"])
                      
                function_output = "go_to_meeting_point(" + str(target_location) + ")"            
                        
        else:
        
            max_utility_key = sorted_utility[0][0] #get the action with the highest utility
            
            tmp_finish = False
            
            #if pickup_ready:
            #    pdb.set_trace()
                
            
            if "sense" in max_utility_key: #If a sensing action is the highest utility action
                function_output = self.sense_equals(max_utility_key, robotState, info)
                    
            elif "pickup" in max_utility_key: #If a pickup action is the highest utility action
                function_output = self.carry_equals(max_utility_key, possible_actions, robotState, info)
                    
            elif "explore" in max_utility_key: #If exploring is the highest utility action
            
                if self.movement.help_status == self.movement.HelpState.being_helped:
                    _,self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text, message=MessagePattern.carry_help_finish())
            
                function_output = "explore()"
                
            elif "wait" in max_utility_key: #If we should wait instead
            
                function_output = "wait()"
                
            elif "interdependency" in self.team_structure and self.team_structure["interdependency"][self.env.robot_id] == "follower":
            
                followed_id = self.get_closest_robot("interdependency", "followed", robotState, info)
                
                function_output = "approach('" + followed_id + "')"
                
                
            else: #otherwise let's go to the final meeting place
                true_ending_locations = [loc for loc in self.ending_locations if self.occMap[loc[0],loc[1]] == 0 or self.occMap[loc[0],loc[1]] == -2]
                
                target_location = random.choice(true_ending_locations)  
                if [ego_location[0][0],ego_location[1][0]] in self.ending_locations: #If we are already in the ending locations just stay there
                    target_location = [ego_location[0][0],ego_location[1][0]]
                    
                
                if self.movement.help_status == self.movement.HelpState.being_helped:
                    _,self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text, message=MessagePattern.carry_help_finish())
                    
                function_output = "go_to_meeting_point(" + str(target_location) + ")"
                
                tmp_finish = True
                
            
            if not tmp_finish and self.finished: #If we haven't finished yet, say so
                self.finished = False
                self.message_text += MessagePattern.finish_reject()
                
            if max_utility_key:
                print("UTILITY >>>>>>> ", function_output, max_utility_key, utility[max_utility_key], utility)
            else:
                print("No possible action utility")
            
        print(function_output, self.message_text, self.agent_requesting_order)
        
        #if not function_output and not self.message_text:
        #    pdb.set_trace()
        
        return function_output
            
    
