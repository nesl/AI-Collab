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
from optimized_strategy import DynamicSensorPlanner

#TODO when agents order the same thing at the same time, an error occurs

class OptimizedControl:

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
        
        
        self.hierarchy_finished = False
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
        
        if self.team_structure["hierarchy"][self.env.robot_id] == "order":
            self.order_status = self.OrderStatus.giving_order
        else:
            self.order_status = self.OrderStatus.finished
            
        self.order_status_info = []
        self.finished = False
        self.told_to_finish = False
        self.collect_attempts = {}
        self.agent_requesting_order = {r[0]:False for r in self.env.map_config['all_robots'] if r[0] != self.env.robot_id}
        self.trigger = False
        self.return_waiting_time = 0
        self.functions_to_execute = []
        self.functions_executed = False
        self.room_object_ids = []
        self.help_agent_ids = set()
        self.plan = {}
        self.planner = []
        self.carry_agents = {}
        self.previous_plan = {}
        
        
        self.return_times = []
        
        if "return" in self.team_structure and bool(self.team_structure["return"][self.env.robot_id]):
            return_time = 5*60
            self.return_times = list(range(return_time,int(self.env.map_config["timer_limit"]), return_time))
        
        
        self.leader_id = ""
        
        
        self.extended_goal_coords = env.goal_coords.copy()
        self.extended_goal_coords.extend([(g[0]+op[0],g[1]+op[1]) for g in env.goal_coords for op in [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]] if [g[0]+op[0],g[1]+op[1]] not in env.goal_coords])
        
        self.ending_locations = [[x,y] for x in range(8,13) for y in range(15,19)] #ending locations
        self.ending_locations.remove([12,18]) #To ensure all locations are within communication range
        self.ending_locations.remove([8,18])
        
        self.human_to_ai_text = []
        if not all(robot[1] for robot in env.neighbors_info): #Check if there are human peers    
            self.human_to_ai_text = Human2AIText(env, robotState, team_structure)
       
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
        cancelling_order = 5
        giving_order = 6
        
        
    def action_description(self,function_str):
    
        function_description = ""
    
        try:
            if "go_to_location" in function_str:
            
                if re.search('\[ *-?\d+(\.(\d+)?)? *, *-?\d+(\.(\d+)?)? *\]',function_str):
                    location = re.search('\[ *-?\d+(\.(\d+)?)? *, *-?\d+(\.(\d+)?)? *\]',function_str).group()
                    xy_grid = eval(location)
                    xy_world = self.env.convert_to_real_coordinates(xy_grid)
                    if xy_world:
                        function_description = "I'm going to location [" + str(xy_world[0]) + "," + str(xy_world[1]) + "]. "
                else:
                    arguments = function_str.split(",")
                    object_id = eval(arguments[0][arguments[0].find("(") + 1:])
                
                    if str(object_id)[0].isalpha():
                        if "room" in object_id:
                            room_match = re.search("(\d+)",object_id)
                            if room_match:
                                room_number = room_match.group(1)
                                room = "room " + room_number
                            elif "goal" in object_id:
                                room = "the goal area"
                            else:
                                room = "the main area"
                            
                            function_description = "I'm going to " + room + ". "    
                             
                        else:
                        
                            function_description = "I'm going with agent " + object_id + ". "
                        
                    elif isinstance(object_id, list):
                    
                        function_description = "I'm going to location " + str(object_id) + ". "
                    
                    elif object_id == -1:
                        function_description = "I'm going to the goal location. "
                    elif object_id == -2:
                        function_description = "I'm going to explore. "
                    else:
                        function_description = "I'm going towards object " + str(object_id) + ". "
                    
            elif "sense_room" in function_str:
                arguments = function_str.split(",")
                argument = eval(arguments[0][arguments[0].find("(") + 1:])
                
                function_description = "I'm going to sense all objects in room " + str(argument) + ". "
            
            elif "sense_object" in function_str:
                arguments = function_str.split(",")
                argument = eval(arguments[0][arguments[0].find("(") + 1:])
                
                function_description = "I'm going to sense object " + str(argument) + ". "     
                
            elif "follow" in function_str:
                arguments = function_str.split(",")
                argument = eval(arguments[0][arguments[0].find("(") + 1:])
                
                function_description = "I'm going to follow agent " + str(argument) + ". "
                
            elif "collect_object" in function_str:
                arguments = function_str.split(",")
                argument = eval(arguments[0][arguments[0].find("(") + 1:])
                
                function_description = "I'm going to collect object " + str(argument) + ". "
                
            elif "approach" in function_str:
                arguments = function_str.split(",")
                argument = eval(arguments[0][arguments[0].find("(") + 1:])
                
                function_description = "I'm going to approach agent " + str(argument) + ". "
            
            elif "explore" in function_str:
                function_description = "I'm going to explore the area. "
                
            elif "wait" in function_str:
                function_description = "I'm going to wait. "
                
            elif "go_to_meeting_point" in function_str:
                function_description = "I'm going to the meeting point. "
                
            elif "ask_for_help" in function_str:
                arguments = function_str.split(",")
                argument = eval(arguments[0][arguments[0].find("(") + 1:])
                argument2 = eval(arguments[1])
                
                function_description = "I'm going to ask agent " + str(argument2) + " for help to carry object " + str(argument) + ". "
                
            elif "ask_for_help_to_carry" in function_str:
                arguments = function_str.split(",")
                argument = eval(arguments[0][arguments[0].find("(") + 1:])
                
                function_description = "I'm going to ask for help to carry object " + str(argument) + ". "
             
            elif "ask_for_sensing" in function_str:   

                arguments = function_str.split(",")
                argument = eval(arguments[0][arguments[0].find("(") + 1:])
                argument2 = eval(arguments[1])

                function_description = "I'm going to ask agent " + str(argument2) + " for help to sense object " + str(argument) + ". "


            elif "drop" in function_str:
                function_description = "I'm going to drop an object. "   
        
        except:
            print("Error description")
            #pdb.set_trace()
            
            
        return function_description
            
       
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
                
                #if self.team_structure["role"][self.env.robot_id] != "sensing": 
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
                    
                    if self.helping_type == self.HelpType.sensing and self.movement.help_status == self.movement.HelpState.asking and self.movement.help_status_info and rm[0] in self.movement.help_status_info[0]: #HERE
                        self.movement.help_status = self.movement.HelpState.no_request
                    
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
                
                self.trigger = True
                
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
                            if self.team_structure["hierarchy"][self.env.robot_id] == "obey":
                                
                                if rematch.group(2) in info['object_key_to_index']:
                                
                                    if self.order_status == self.OrderStatus.finished:
                                        object_idx = info['object_key_to_index'][rematch.group(2)]
                                        
                                        object_location = robotState.get("objects", "last_seen_location", object_idx)
                                        if (object_location[0] == -1 and object_location[1] == -1): #might be because object is already in goal
                                            self.message_text += "I don't know where is object " + rematch.group(2) + ". Provide some information first. "
                                        else:
                                        
                                            object_weight = robotState.get("objects", "weight", object_idx)
                                            
                                            
                                            if not object_weight:
                                                self.message_text += "First give me the weight value of object " + rematch.group(2) + ". "
                                            else:
                                                if object_weight > 1 and not (self.movement.help_status == self.movement.HelpState.being_helped and len(self.movement.help_status_info[0]) == object_weight-1):
                                                    self.create_action_function("ask_for_help('" + rematch.group(2) + "','" +  rm[0] + "')")
                                                else:
                                                    self.create_action_function("collect_object('" + rematch.group(2) + "')")
                                                    #pdb.set_trace()
                                                
                                                self.order_status = self.OrderStatus.ongoing
                                                
                                                #But first i NNed to self.movement.help_status
                                                
                                                self.message_text += MessagePattern.order_response(rm[0], "collect")
                                                if self.movement.help_status == self.movement.HelpState.helping:
                                                    self.message_text += "I'll do it after I finish helping. "
                                                    
                                                self.leader_id = rm[0]
                                    else:
                                        self.message_text += MessagePattern.order_response_negative(rm[0], self.leader_id)
                                else:
                                    self.message_text += "I don't know where is object " + rematch.group(2) + ". Provide some information first. "
                                    
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
                                    self.order_status = self.OrderStatus.ongoing
                                else:
                                    if self.order_status == self.OrderStatus.finished:
                                        
                                        location = list(eval(rematch.group(3)))
                            
                                        max_real_coords = self.env.convert_to_real_coordinates((robotState.latest_map.shape[0]-1, robotState.latest_map.shape[1]-1))
                                        
                                        if location[0] > max_real_coords[0] or location[1] > max_real_coords[1]: #last_seen[0] == 99.99 and last_seen[1] == 99.99:
                                            assigned_target_location = robotState.get("objects", "last_seen_location", info['object_key_to_index'][object_id])
                                        else:
                                            assigned_target_location = self.env.convert_to_grid_coordinates(location)
                                        
                                        self.create_action_function("sense_object(''," +  str(assigned_target_location) + ")")
                                        
                                        self.order_status = self.OrderStatus.ongoing
                                        self.message_text += MessagePattern.order_response(rm[0], "sense")
                                        self.leader_id = rm[0]
                                    else:
                                        self.message_text += MessagePattern.order_response_negative(rm[0], self.leader_id)
                            else:
                                 self.message_text += MessagePattern.order_not_obey(rm[0])
                        
            if re.search(MessagePattern.order_sense_multiple_regex(),rm[1]):
            
                template_match = True
            
                if "hierarchy" in self.team_structure and self.team_structure["role"][self.env.robot_id] != "lifter":  
                    for rematch in re.finditer(MessagePattern.order_sense_multiple_regex(),rm[1]):
                        if rematch.group(1) == self.env.robot_id:
                            if self.team_structure["hierarchy"][self.env.robot_id] == "obey" and self.team_structure["hierarchy"][rm[0]] == "order":
                            
                                object_ids = rematch.group(2)[1:].replace("]", "").split(",")
                                
                                coords_regex = '\(-?\d+\.\d+,-?\d+\.\d+\)'
                                
                                locations = []
                                
                                for coord in re.finditer(coords_regex,rematch.group(4)):
                                    locations.append(eval(coord.group(1)))
                                
                                delete_objs = []
                                for ob_idx,ob in enumerate(object_ids):
                                    if ob in info['object_key_to_index'] and info['object_key_to_index'][ob] in robotState.get_object_keys() and robotState.get("object_estimates", "danger_status", [info['object_key_to_index'][ob],robotState.get_num_robots()]): #send info if already have it
                                        self.message_text += MessagePattern.item(robotState,info['object_key_to_index'][ob],ob, info, self.env.robot_id, self.env.convert_to_real_coordinates) 
                                        
                                        delete_objs.append(ob_idx)
                                
                                delete_objs.reverse()      
                                for ob_idx in delete_objs:
                                    del object_ids[ob_idx]
                                    del locations[ob_idx]
                                
                                if not object_ids:
                                    self.order_status = self.OrderStatus.ongoing
                                else:
                                    if self.order_status == self.OrderStatus.finished:
                                        
                                        max_real_coords = self.env.convert_to_real_coordinates((robotState.latest_map.shape[0]-1, robotState.latest_map.shape[1]-1))
                                        
                                        assigned_target_locations = []
                                        for l_idx,location in enumerate(locations):
                                        
                                            if location[0] > max_real_coords[0] or location[1] > max_real_coords[1]: #last_seen[0] == 99.99 and last_seen[1] == 99.99:
                                                assigned_target_location = robotState.get("objects", "last_seen_location", info['object_key_to_index'][object_ids[l_idx]])
                                            else:
                                                assigned_target_location = self.env.convert_to_grid_coordinates(location)
                                            
                                            assigned_target_locations.append(assigned_target_location)
                                            
                                        self.create_action_function("sense_multiple_objects(" +  str(assigned_target_locations) + ")")
                                        
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
                                        if (object_location[0] == -1 and object_location[1] == -1):#might be because object is already in goal
                                            self.message_text += "I don't know where is object " + rematch.group(2) + ". Provide some information first. "
                                        else:
                                        
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
                            
                            if self.movement.help_status == self.movement.HelpState.being_helped:
                                _,self.message_text,_ = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_finish())
                            else:
                                self.movement.help_status = self.movement.HelpState.no_request
                                self.movement.help_status_info[0] = []
                            
                        elif t != robot_idx:
                        
                            robot_id = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(t)]
                            self.message_text += MessagePattern.order_cancel(robot_id)
                            
                    
                        
                    self.message_text += "Understood " + rm[0] + ". "
                    
            if re.search(MessagePattern.order_cancel_regex(), rm[1]):
                template_match = True
            
                for rematch in re.finditer(MessagePattern.order_cancel_regex(),rm[1]):
            
                    if "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "obey" and self.env.robot_id == rematch.group(1): # and self.leader_id == rm[0]:
                    
                        
                        self.order_status = self.OrderStatus.finished
                        #self.leader_id = ""
                        self.action_function = ""
                        self.top_action_sequence = 0
                        self.message_text += "Ok " + rm[0] + ". I will not fulfill your order. "
                        
                        if self.movement.help_status == self.movement.HelpState.being_helped:
                            _,self.message_text,_ = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_finish())
                        else:
                            self.movement.help_status = self.movement.HelpState.no_request
                            self.movement.help_status_info[0] = []
                            
                        self.action_index = self.State.decision_state
                        self.action_function = ""
                        self.top_action_sequence = 0
                        
                        if robotState.object_held:
                            self.create_action_function("drop()")
                            self.order_status = self.OrderStatus.ongoing
                            
                        self.leader_id = rm[0]
                                        
                
                        
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
                
            if re.search(MessagePattern.order_finished_regex(),rm[1]):  
            
                template_match = True
            
                if "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "order" and not re.search(MessagePattern.order_response_regex(),rm[1]):
                    robot_idx = info['robot_key_to_index'][rm[0]]
                    
                    self.other_agents[robot_idx].previous_assignment  = self.other_agents[robot_idx].assignment
                    self.other_agents[robot_idx].assignment = "" 
                    
                    robotState.set("agents", "team", robot_idx, "[]", info["time"])
                    
                    if (all(robotState.get("agents", "type", -1)) or (not robotState.get("agents", "type", robot_idx) and self.team_structure["hierarchy"][rm[0]] == "obey")) and not (self.order_status == self.OrderStatus.giving_order and rm[0] in self.plan.keys() and self.plan[rm[0]][0] != self.previous_plan[rm[0]][0]): #If there is no human or the human is obeying
                        self.agent_requesting_order[rm[0]] = True
                        
                        print(self.agent_requesting_order)
                        
                        if not all(self.agent_requesting_order[r] for r in self.agent_requesting_order.keys()):
                            self.give_new_order(rm[0], robotState, info)
                    
                    self.message_text += "Thanks " + rm[0] + ". "
                      
            if re.search(MessagePattern.sensing_ask_help_regex(),rm[1]):
            
                template_match = True
            
                rematch = re.search(MessagePattern.sensing_ask_help_regex(),rm[1])
                
                if rematch.group(1) == str(self.env.robot_id) and self.team_structure["role"][self.env.robot_id] != "lifter" and not ("hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "obey"):
                
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
                
                
                available_robots = [r[0] for r in self.env.map_config['all_robots']]
                leaders = [tm for tm in self.team_structure["hierarchy"].keys() if self.team_structure["hierarchy"][tm] == "order" and tm in available_robots]
                
                if "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "obey" and rm[0] in leaders:
                    self.finished = True
                        
                    self.message_text += "Let's finish, " + rm[0] + ". "

                        
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
                
                if rematch.group(1) == str(self.env.robot_id):
                    if self.action_index == self.movement.State.wait_random:
                        self.action_index = self.movement.last_action_index
                        self.pending_location = []
                    self.message_text += "Ok " + rm[0] + ". "
            
            if re.search(MessagePattern.sensing_ask_help_incorrect_regex(),rm[1]):
                template_match = True    
            
            #template_match = True #CNL only
            
            if not template_match and translated_messages_index >= 0 and translated_messages_index >= rm_idx: #This means the translated message doesn't make sense
                print("understand here 1")

                self.message_text += "I didn't understand you " + rm[0] + ". "
                continue
            
            #print(not template_match, not robotState.get("agents", "type", info['robot_key_to_index'][rm[0]]), rm[1])
            if not template_match and not robotState.get("agents", "type", info['robot_key_to_index'][rm[0]]): #Human sent a message, we need to translate it. We put this condition at the end so that humans can also send messages that conform to the templates
            
                if 'RESET' in rm[1]:
                    self.message_history = []
                    continue
                elif 'DEBUG' in rm[1]:
                    pdb.set_trace()
                    
            
                self.env.sio.emit("text_processing", (True))
                asking_for_help = False
                if "ask_for_help" in self.action_function:
                    asking_for_help = True
                elif "ask_for_sensing" in self.action_function and self.movement.help_status != self.movement.HelpState.being_helped:
                    asking_for_help = True
                else:
                    if self.message_history and info["time"] - self.message_history[-1]["Time"] > 30 and len(self.message_history) > 5:
                        self.human_to_ai_text.summarize_messages(self.message_history,robotState,info)
                        self.message_history = []
                try:
                    
                    translated_message,message_to_user,functions = self.human_to_ai_text.convert_to_ai(rm[0], rm[1], info, robotState, self.message_history, True, asking_for_help)
                    
                except:
                    translated_message = ""
                    message_to_user = ""
                    functions = []
                    #pdb.set_trace()
                    print("Error in translation")
                '''
                try:
                    translated_message,message_to_user = self.human_to_ai_text.convert_to_ai(rm[0], rm[1], info, robotState, self.message_history, True)
                except:
                    translated_message = ""
                    message_to_user = ""
                    print("Error in translation")
                '''
                
                if asking_for_help:
                    if functions:
                        print(functions)
                        eval("self." + functions[0][:-1] + ",robotState, info)")
                else:
                
                    if functions:
                        if self.message_history and len(self.message_history) > 10:
                            self.human_to_ai_text.summarize_messages(self.message_history,robotState,info)
                            self.message_history = []
                            
                        #cancel eveything, make sure to drop all objects maybe?
                        print("GOT FUNCTIONS",functions)
                        
                        if not ("hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "obey"):
                            if not robotState.object_held and (self.movement.help_status == self.movement.HelpState.no_request or rm[0] in self.movement.help_status_info[0]):
                                self.movement.help_status = self.movement.HelpState.no_request #Check if this works
                                self.functions_to_execute = functions
                                self.action_function = ""
                                self.action_index = self.State.decision_state
                            else:
                                translated_message = ""
                                if robotState.object_held:
                                    translated_message += MessagePattern.carry_help_participant_reject_object()
                            
                                elif rm[0] in self.movement.help_status_info[0]:
                        
                                    if self.movement.help_status == self.movement.HelpState.being_helped:
                                        translated_message += MessagePattern.carry_help_participant_affirm_being_helped(rm[0])
                                    elif self.movement.help_status == self.movement.HelpState.asking:
                                        translated_message += MessagePattern.carry_help_participant_asking(rm[0])
                                    else:
                                        translated_message += MessagePattern.carry_help_participant_affirm(rm[0])
                                else:
                                    translated_message += MessagePattern.carry_help_participant_reject(rm[0])
                                    
                                    if self.movement.help_status == self.movement.HelpState.asking:
                                        translated_message += MessagePattern.carry_help_participant_reject_asking()    
                                    elif self.movement.help_status_info[0]:
                                        if self.movement.help_status == self.movement.HelpState.being_helped:
                                            translated_message += MessagePattern.carry_help_participant_reject_helping(self.movement.help_status_info[0][0])
                                        else:
                                            translated_message += MessagePattern.carry_help_participant_reject_other(self.movement.help_status_info[0][0])
                        else:
                            self.movement.cancel_cooperation(self.State.decision_state,self.message_text)
                            self.functions_to_execute = functions
                            self.action_function = ""
                            self.action_index = self.State.decision_state
                            self.functions_executed = True
                            
                self.env.sio.emit("text_processing", (False))
                    
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
                tmp_message_history.append({"Sender": rm[0], "Message": tmp_message, "Time": rm[2]})
        
                
        if tmp_message_history:
            self.message_history.extend(tmp_message_history)
    
    def accept_help(self, robot_id, robotState, info):
        if self.movement.help_status == self.movement.HelpState.asking or self.movement.help_status == self.movement.HelpState.being_helped:

            num_agents_needed = robotState.get("objects", "weight", self.chosen_object_idx)
            
            rm = [robot_id,MessagePattern.carry_help_accept(self.env.robot_id)]
            
            return_value,self.message_text,_ = self.movement.message_processing_carry_help_accept(rm, {"weight": num_agents_needed}, self.message_text, self.helping_type == self.HelpType.sensing)
            
            
            if return_value == 1:

                
                object_location = robotState.get("objects", "last_seen_location", self.chosen_object_idx)
                if not (object_location[0] == -1 and object_location[1] == -1):
                    self.target_location = object_location
                    self.object_of_interest = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.chosen_object_idx)]
                    #self.target_object_idx = self.heavy_objects["index"][self.chosen_heavy_object]
                    
                    self.action_index = self.State.decision_state
    
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
        if not ("explore(" in self.action_function or "wait(" in self.action_function or "drop(" in self.action_function or "sleep(" in self.action_function):
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
        
        
        if not self.hierarchy_finished:
            self.hierarchy_finished = True
            available_robots = [r[0] for r in self.env.map_config['all_robots']]
            for tm in self.team_structure["hierarchy"].keys():
                if self.team_structure["hierarchy"][tm] == "obey" and tm in available_robots:
                    self.other_agents[info['robot_key_to_index'][tm]].finished = True
        
        
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
            
                
            
                robotState.set("agents", "current_state", robotState.get_num_robots(), self.action_function, info["time"])
            
                if not self.action_function or self.help_requests:
                    self.action_sequence = 0
                    self.top_action_sequence = 0
                    
                    external_function = False
                    if self.functions_to_execute:
                        function_str = self.functions_to_execute.pop(0)
                        external_function = True
                        
                    else:
                    
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
                        
                        robotState.current_action_description = self.action_description(self.action_function)
                        
                        if external_function:
                            self.message_text += robotState.current_action_description
                        
                    else:
                        #self.action_function = self.create_action_function("wait")
                        print("action_function 0", self.agent_requesting_order)

                
                print(self.action_function)   
                #pdb.set_trace()                     
                #try:
                action, action_finished,function_output = eval(self.action_function)
                print("function output", function_output)
                #except:
                #    pdb.set_trace()
                
                if message_order not in self.message_text:
                    self.message_text = message_order + self.message_text
                    message_order = ""
                #except:
                #    pdb.set_trace()

                
                if action_finished:
                    self.action_sequence = 0
                    self.top_action_sequence = 0
                    
                    external_function = False
                    if self.functions_to_execute:
                        function_str = self.functions_to_execute.pop(0)
                        external_function = True
                    else:
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
                        robotState.current_action_description = self.action_description(self.action_function)
                        if external_function:
                            self.message_text += robotState.current_action_description
                    else: #No function selected
                        self.action_function = ""
                        print("action_function 1", self.agent_requesting_order)
                        
                        
                if all(self.agent_requesting_order[r] for r in self.agent_requesting_order.keys()) and self.order_status == self.OrderStatus.cancelling_order:
                    
                    self.agent_requesting_order = {r[0]:False for r in self.env.map_config['all_robots'] if r[0] != self.env.robot_id}
                    
                    function_str = self.decision(messages, robotState, info, function_output, self.nearby_other_agents, self.help_requests)
                    
                    if function_str:
                        self.create_action_function(function_str)
                        print("Created function")
                        self.action_sequence = 0
                        self.top_action_sequence = 0


            else:
            
                robotState.set("agents", "current_state", robotState.get_num_robots(), self.action_index.name, info["time"])
                
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
                
                if self.action_index == self.movement.State.follow or self.action_index == self.movement.State.obey:
                    robotState.current_action_description = self.action_description("follow(\"" + self.movement.help_status_info[0][0] + "\", robotState, next_observation, info)")
                
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


        print("action index:",self.action_index, "action:", Action(action["action"]), ego_location, self.action_function, self.movement.help_status, self.top_action_sequence, self.order_status, self.plan, info['time'])
        
        if "end_participation" in self.action_function:
            print("end participation")
            terminated = True
        
        #print("Finished?", [p.finished for p in self.other_agents], self.finished)
        
        if all(p.finished for p in self.other_agents) and self.finished:
            print("finished participation")
            terminated = True

        return action,terminated
        
    def tell_agent(self, message, robotState, next_observation, info):
    
        action = self.sample_action_space
        action["action"] = -1
        action["robot"] = 0
        finished = True

        output = []
        
        message = self.human_to_ai_text.sql_request2(self.message_history,robotState,message)
        
        action,finished,output = self.send_message(message, robotState, next_observation, info)

        return action,finished,output
        
    def follow(self, agent_id, robotState, next_observation, info):
    
        finished = False
        action = self.sample_action_space
        action["robot"] = 0
        action["action"] = Action.get_occupancy_map.value
        output = []
        
        self.helping_type = self.HelpType.carrying
        rm = [agent_id,MessagePattern.carry_help("0", 1)]
        self.movement.message_processing_carry_help(rm, robotState, self.action_index, self.message_text)
        
        rm = [agent_id,MessagePattern.follow(self.env.robot_id)]
        self.action_index,_,_ = self.movement.message_processing_help(rm, self.action_index, self.helping_type == self.HelpType.sensing, self.State.decision_state)
        
        return action, finished, output           
    
    
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
        
    def sense_multiple_objects(self, grid_locations, robotState, next_observation, info):
    
        finished = False
        action = self.sample_action_space
        action["robot"] = 0
        action["action"] = Action.get_occupancy_map.value
        output = []
        
        
        
        return action, finished, output
        
    def sense_object(self, object_id, grid_location, robotState, next_observation, info):
        
        finished = False
        action = self.sample_action_space
        action["robot"] = 0
        action["action"] = Action.get_occupancy_map.value
        output = []
        
        try:
            if str(object_id) in info['object_key_to_index'] and robotState.get("objects", "already_sensed", info['object_key_to_index'][str(object_id)]) == "Yes":
                return action, True, [] 
        except:
            print("Don't know object sense_object")
        
        if self.top_action_sequence == 0:
        
            if grid_location and not (grid_location[0] == -1 and grid_location[0] == -1):
                chosen_location = grid_location
                object_id = grid_location
            else:
                try:
                    chosen_location = robotState.get("objects", "last_seen_location", info['object_key_to_index'][str(object_id)]) #robotState.items[info['object_key_to_index'][str(object_id)]]["item_location"]
                    self.object_of_interest = object_id
                except:
                    self.message_text += "I don't know where object " + str(object_id) + " is. "
                    return action, True, []
        
            if (chosen_location[0] == -1 and chosen_location[1] == -1):# or self.occMap[chosen_location[0],chosen_location[1]] != 2: #if there is no object in the correct place
                print("object not found for sensing")

                finished = True
        
            action, temp_finished, output = self.go_to_location(object_id, robotState, next_observation, info)
            if temp_finished:
                self.top_action_sequence += 1
        elif self.top_action_sequence == 1:
            action, finished, output = self.activate_sensor(robotState, next_observation, info)
            
    
        return action, finished, output
        
    def sense_room(self, room, robotState, next_observation, info):
    
        finished = False
        action = self.sample_action_space
        action["robot"] = 0
        action["action"] = Action.get_occupancy_map.value
        output = []
        ego_location = np.where(robotState.latest_map == 5)
        
        if room not in ['1','2','3','4']:
            finished = True
            return action, finished, output 
        
        if self.top_action_sequence == 0:
            current_room = self.env.get_room([ego_location[0][0],ego_location[1][0]], True)
            #print(current_room, "room " + room, current_room == "room " + room)
            if current_room == "room " + room:
                self.top_action_sequence += 1
                '''
                room_coords = self.env.get_coords_room(self, robotState.latest_map, room, objects=True)
                cells = robotState.latest_map[room_coords[:,0],room_coords[:,1]]
                
                object_cells = np.argwhere(cells == 2)
                
                self.room_object_coords = room_coords[object_cells].squeeze().tolist()
                '''
                
                self.room_object_ids = []
                for ob_idx in range(robotState.get_num_objects()): #For all possible objects
                    ob_location = robotState.get("objects", "last_seen_location", ob_idx)
                    if self.env.get_room(ob_location, True) == current_room:
                        object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(ob_idx)]
                        self.room_object_ids.append(object_id)
                
                if not self.room_object_ids:
                    finished = True
                    
                #if not self.room_object_ids:
                #    pdb.set_trace()
                print("Objects", self.room_object_ids)
            else:
                action, temp_finished, output = self.go_to_location("room " + room, robotState, next_observation, info)
                
        elif self.top_action_sequence == 1:
        
            while self.room_object_ids and robotState.get("object_estimates", "danger_status", [info['object_key_to_index'][self.room_object_ids[0]],robotState.get_num_robots()]):
                print("Objects1", self.room_object_ids)
                self.room_object_ids.pop(0)
                
        
            if not self.room_object_ids:
                finished = True
            else:
                
                object_id = self.room_object_ids[0]
                action, temp_finished, output = self.go_to_location(object_id, robotState, next_observation, info)
                
                if not temp_finished:
                    self.top_action_sequence += 1
                else:
                    print("Objects2", self.room_object_ids)
                    self.room_object_ids.pop(0)
                    self.action_sequence = 0
                    action, _, output = self.activate_sensor(robotState, next_observation, info)
                    self.top_action_sequence = 3
          
        elif self.top_action_sequence == 2:
            object_id = self.room_object_ids[0]
            action, temp_finished, output = self.go_to_location(object_id, robotState, next_observation, info)
            
            if temp_finished:
                print("Objects3", self.room_object_ids)
                self.room_object_ids.pop(0)
                self.action_sequence = 0
                action, _, output = self.activate_sensor(robotState, next_observation, info)
                self.top_action_sequence = 3
                
        elif self.top_action_sequence == 3: 
            action, temp_finished, output = self.activate_sensor(robotState, next_observation, info)
            if temp_finished:
                self.top_action_sequence = 1
            
        return action, finished, output     
        
    def collect_object(self, object_id, robotState, next_observation, info):
    
        finished = False
        output = []
        
        action = self.sample_action_space
        
        ego_location = np.where(robotState.latest_map == 5)
        
        self.chosen_object_idx = info['object_key_to_index'][str(object_id)]
        
        if not len(self.movement.help_status_info[0])+1 >= robotState.get("objects", "weight", info['object_key_to_index'][str(object_id)]):
            finished = True
            action["action"] = Action.get_occupancy_map.value
            return action, finished, output
        
        
        if self.top_action_sequence == 0:
        
            self.object_of_interest = object_id

            chosen_location = robotState.get("objects", "last_seen_location", info['object_key_to_index'][str(object_id)]) #robotState.items[info['object_key_to_index'][str(object_id)]]["item_location"]
            
            if (chosen_location[0] == -1 and chosen_location[1] == -1) or tuple(chosen_location) in self.extended_goal_coords: #or self.occMap[chosen_location[0],chosen_location[1]] != 2 if there is no object in the correct place
                finished = True
                action["action"],self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_finish())
                self.object_of_interest = ""
                if tuple(chosen_location) in self.extended_goal_coords:
                    self.message_text += "The object is already in the goal area. "
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
                        self.object_of_interest = ""
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
                            break
                    
                    
                    
            elif not combinations_found: #No way of moving                          
                action["action"],self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_finish())
                finished = True
                self.object_of_interest = ""
            elif time.time() - self.movement.help_status_info[1] > self.help_time_limit2: #time.time() - self.movement.asked_time > self.help_time_limit2:                           
                action["action"],self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_complain())
                finished = True
                self.object_of_interest = ""
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
            
            
            if(not robotState.object_held):
                finished = True
                self.object_of_interest = ""
                output = self.held_objects
            else:
                action, _, output = self.drop(robotState, next_observation, info)
            
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
    

        
    def ask_for_help_to_carry(self, object_id, robotState, next_observation ,info):
    
        action = self.sample_action_space
        action["robot"] = 0
        action["action"] = Action.get_occupancy_map.value
        finished = False
        output = []
        
        ego_location = np.where(robotState.latest_map == 5)
        
        self.chosen_object_idx = info['object_key_to_index'][str(object_id)]
        
        
        if self.top_action_sequence == 0:
            self.help_agent_ids = [r for r in range(robotState.get_num_robots())]
            self.top_action_sequence += 1
            
        elif self.top_action_sequence == 1:
            
            distances = []
            for ha in self.help_agent_ids:
                loc = robotState.get("agents", "last_seen_location", ha)
                if (loc[0] == -1 and loc[1] == -1):
                    distances.append(float("inf"))
                else:
                    distances.append(np.linalg.norm(np.array(loc) - np.array([ego_location[0][0],ego_location[1][0]])))
                    
            self.help_agent_ids = [x for _,x in sorted(zip(distances,self.help_agent_ids))]
            
            self.top_action_sequence += 1
            
            
                
        elif self.top_action_sequence == 2:

            nearby_agents = set(self.help_agent_ids).intersection(set(self.nearby_other_agents))
            if not nearby_agents:
            
                robot_idx = self.help_agent_ids[0]
                
                try:
                    robot_id = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(robot_idx)]
                except:
                    pdb.set_trace()
                chosen_location = robotState.get("agents", "last_seen_location", robot_idx) #robotState.robots[info['robot_key_to_index'][str(robot_id)]]["neighbor_location"]
                
                if (chosen_location[0] == -1 and chosen_location[1] == -1): #if there is no agent in the correct place
                    print("no more helping")
                    if self.nearby_other_agents:
                        action,temp_finished,_ = self.ask_info(robot_id, MessagePattern.ask_for_agent(robot_id), robotState, next_observation, info)
                        if temp_finished:
                            self.action_sequence = 0
                            chosen_location = robotState.get("agents", "last_seen_location", robot_idx) #robotState.robots[info['robot_key_to_index'][str(robot_id)]]["neighbor_location"]
                            if (chosen_location[0] == -1 and chosen_location[1] == -1):
                                del self.help_agent_ids[0]
                                if not self.help_agent_ids:
                                    finished = True
                    else:
                        del self.help_agent_ids[0]
                        if not self.help_agent_ids:
                            finished = True
                        self.action_sequence = 0
                
                if not (chosen_location[0] == -1 and chosen_location[1] == -1):
                    action, temp_finished, output = self.go_to_location(robot_id, robotState, next_observation, info)
                    
                    real_distance = self.env.compute_real_distance([chosen_location[0],chosen_location[1]],[ego_location[0][0],ego_location[1][0]])
                        
                    distance_limit = self.env.map_config['communication_distance_limit']-1
                    
                    if real_distance < distance_limit:
                        self.top_action_sequence += 1
                        #self.movement.asked_time = time.time()
                        #self.movement.being_helped_locations = []
                        self.movement.help_status_info[1] = time.time()
                        self.movement.help_status_info[2] = []

            else:
                self.top_action_sequence += 1
                    
                self.movement.help_status_info[1] = time.time()
                self.movement.help_status_info[2] = []
                
        elif self.top_action_sequence == 3:
        
        
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
                
            
            message_to_send = MessagePattern.carry_help(str(object_id),robotState.get("objects", "weight", object_idx)-1-robots_already_helping)
            
            self.message_text += message_to_send
         
            #self.movement.asked_help = True
            #self.movement.asked_time = time.time()
            
            self.movement.help_status_info[1] = time.time()
            
            
            self.chosen_object_idx = object_idx
            print("ASKING HELP")
            self.top_action_sequence += 1
            
        elif self.top_action_sequence == 4:
            object_idx =info['object_key_to_index'][str(object_id)]

            
            if (self.movement.help_status != self.movement.HelpState.asking and len(self.movement.help_status_info[0])+1 >= robotState.get("objects", "weight", object_idx)) or time.time() - self.movement.help_status_info[1] > self.movement.wait_time_limit:
            
                self.help_agent_ids = set(self.help_agent_ids) - set(self.help_agent_ids).intersection(set(self.nearby_other_agents))
            
                if (self.movement.help_status != self.movement.HelpState.asking and len(self.movement.help_status_info[0])+1 >= robotState.get("objects", "weight", object_idx)):
                    finished = True
                elif not self.help_agent_ids:
                    finished = True
                    _,self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text, message=MessagePattern.carry_help_finish())
                else:
                    self.top_action_sequence = 1
                    
            action["action"] = Action.get_occupancy_map.value
        
        
        
        return action,finished,output
    
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
    
    def closest_distance_explore(self, robotState, info, exclude):
    
        ego_location = np.where(robotState.latest_map == 5)
        
        
        agent_view_radius = int(self.env.view_radius)
        if exclude: #If points get excluded
            modified_map = np.copy(robotState.latest_map)
            
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
            
            skip_goal = False
            if "interdependency" in self.team_structure and self.team_structure["interdependency"][self.env.robot_id] == "followed" and se_idx < len(still_to_explore[0])-1: #for autonomous strategies don't explore nearby locations
                for other_agent_idx in range(robotState.get_num_robots()):
                    
                    robot_id = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(other_agent_idx)]
                
                    if self.team_structure["interdependency"][robot_id] == "followed":
                        agent_goal = self.other_agents[other_agent_idx].other_location["goal_location"]
                        agent_location = robotState.get("agents", "last_seen_location", other_agent_idx)
                        if agent_goal:
                            first_condition = self.env.compute_real_distance(agent_goal,[ego_location[0][0],ego_location[1][0]]) < agent_view_radius
                        else:
                            first_condition = False
                        if not (agent_location[0] == -1 and agent_location[1] == -1):
                            second_condition = self.env.compute_real_distance(agent_location,[ego_location[0][0],ego_location[1][0]]) < agent_view_radius
                        else:
                            second_condition = False
                            
                        if first_condition or second_condition:
                            skip_goal = True
                            break
            
            if skip_goal:
                continue
            
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
        place = ""
        
        """
        if action_sequence == 0:
            action_sequence += 1
            action = Action.get_occupancy_map.value
        """
        
        if object_id == -1: #Return to middle of the room
            x = 10
            y = 10
            place = "to the middle of the room"
        elif object_id == -2: #Explore
        
            if not self.explore_location:
            
                x,y = self.closest_distance_explore(robotState, info, [])
                
                self.explore_location = [x,y]
            else:
                x = self.explore_location[0]
                y = self.explore_location[1]
            
            xy_world = self.env.convert_to_real_coordinates([x,y])
            if xy_world:
                place = "towards [" + str(xy_world[0]) + "," + str(xy_world[1]) + "]"
            
        elif str(object_id)[0].isalpha(): #Agent
            
            if "room" in object_id:
            
                current_room = self.env.get_room([ego_location[0][0],ego_location[1][0]], True)
                
                room_match = re.search("(\d+)",object_id)
                if room_match:
                    room_number = room_match.group(1)
                    room = "room " + room_number
                elif "goal" in current_room:
                    room = "goal area"
                else:
                    room = "main area"
            
                if current_room == room:
                    action["action"] = Action.get_occupancy_map.value
                    output = [ego_location[0][0],ego_location[1][0]]
                    finished = True
                    return action,finished,output
                    
                location_list = self.env.get_coords_room(robotState.latest_map, object_id.replace("room ", "").strip())
                
                if location_list.size == 0:
                    action["action"] = Action.get_occupancy_map.value
                    output = [ego_location[0][0],ego_location[1][0]]
                    finished = True
                    return action,finished,output
                
                x,y = random.choice(location_list)
                
                xy_world = self.env.convert_to_real_coordinates([x,y])
                if xy_world:
                    place = "towards [" + str(xy_world[0]) + "," + str(xy_world[1]) + "] in " + room
            else:
            
                robot_idx = info['robot_key_to_index'][str(object_id)]
                
                robo_location = robotState.get("agents", "last_seen_location", robot_idx)
                
                if (robo_location[0] == -1 and robo_location[1] == -1):
                    action["action"] = Action.get_occupancy_map.value
                    return action,True,output
                
                place = "with " + str(object_id)
                x,y = robo_location
            
        elif isinstance(object_id, list):    
        
            x = object_id[0]
            y = object_id[1]
            
            xy_world = self.env.convert_to_real_coordinates([x,y])
            if xy_world:
                place = "towards [" + str(xy_world[0]) + "," + str(xy_world[1]) + "]"
        else:
            try:
            
                item_idx = info['object_key_to_index'][str(object_id)]
            
                object_location = robotState.get("objects", "last_seen_location", item_idx)
                if (object_location[0] == -1 and object_location[1] == -1):
                    action["action"] = Action.get_occupancy_map.value
                    return action,True,output
            
                x,y = object_location
                
                place = "towards object " + str(object_id)
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
                self.message_text += "Something is blocking the way, I cannot go " + place + ". "
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
                self.message_history.append({"Sender": self.env.robot_id, "Message": tmp_message, "Time": info["time"]})
            else:
                tmp_message = ""
            
            if any(not robotState.get("agents", "type", r_idx) for r_idx in self.nearby_other_agents):
                message_ai = rematch_str
                message = tmp_message

        else:
            self.message_history.append({"Sender": self.env.robot_id, "Message": message, "Time": info["time"]})

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
                    self.held_objects = [str(object_id)]
                    robotState.set("agents", "carrying_object", robotState.get_num_robots(), str(object_id), info["time"])
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
        elif self.occMap[target_location[0],target_location[1]] == 5 and not (self.return_times and info["time"] > self.return_times[0]):
        
            robot_disabled = robotState.get("agents", "disabled", -1)
        
            print([r == 1 for r in robot_disabled], self.nearby_other_agents)
            if len(self.nearby_other_agents) == robotState.get_num_robots()-sum(r == 1 for r in robot_disabled):
            
                '''
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
                '''        
                if "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "obey":
                    pass
                else: #TODO test this, maybe in a hierarchy don't finish when they say
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
        
        if self.return_times and info["time"] > self.return_times[0]:
        
            if self.occMap[target_location[0],target_location[1]] == 5:
            
                robot_disabled = robotState.get("agents", "disabled", -1)
                
                available_robots = [r[0] for r in self.env.map_config['all_robots']]
                print("Done???", [[a, info['robot_key_to_index'][a],self.nearby_other_agents, self.team_structure["return"].keys(), self.env.robot_id, available_robots, info['robot_key_to_index'].keys(), robot_disabled, bool(self.team_structure["return"][a])] for a in self.team_structure["return"].keys() if a != self.env.robot_id and a in info['robot_key_to_index'].keys()])
                if all(True if info['robot_key_to_index'][a] in self.nearby_other_agents else False for a in self.team_structure["return"].keys() if a != self.env.robot_id and a in available_robots and a in info['robot_key_to_index'].keys() not in robot_disabled and bool(self.team_structure["return"][a])) or time.time() - self.return_waiting_time >= 120: #check that all required robots are nearby or wait for 2 minute
                    finished = True
                    self.return_times.pop(0)

                    print("DONE WAITING")
                else:
                    finished = False
            else:
                self.return_waiting_time = time.time()
                finished = False
             
        return action,finished,output
        
        
    def drop(self,robotState, next_observation, info):

        action = self.sample_action_space
        action["action"] = -1
        finished = True

        output = self.held_objects
        
        action["action"] = Action.drop_object.value
        
        robotState.set("agents", "carrying_object", robotState.get_num_robots(), "None", info["time"])
        

        return action,finished,output
        
    def move_away_from(self, agent_id, robotState, next_observation, info):
    
        action = self.sample_action_space
        action["robot"] = 0
        action["action"] = Action.get_occupancy_map.value
        finished = True
        output = []
        
        rm = [agent_id,MessagePattern.move_request(self.env.robot_id)]
        self.message_text,self.action_index,_ = self.movement.message_processing_move_request(rm, robotState, info, self.action_index, self.message_text, self.other_agents, self.helping_type == self.HelpType.sensing)
        
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
        
    def sleep(self,robotState, next_observation, info):
        action = self.sample_action_space

        finished = False

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
        
        

        
        
    def get_closest_robot(self, team_structure_category, leader_type, robotState, info):
    
        available_robots = [r[0] for r in self.env.map_config['all_robots']]
        leaders = [tm for tm in self.team_structure[team_structure_category].keys() if self.team_structure[team_structure_category][tm] == leader_type and tm in available_robots]
    
        ego_location = np.where(robotState.latest_map == 5)
    
        try:
            leader_min_distance = [leaders[0],float("inf")]
        except:
            pdb.set_trace()
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
            leader_id = leaders[0]
            
        
        return leader_id
            
    def decision_obey(self,messages, robotState, info, output, nearby_other_agents, next_observation):
    
        if self.functions_executed:
            self.functions_executed = False
            self.order_status = self.OrderStatus.reporting_availability
    
        if self.order_status == self.OrderStatus.ongoing:
            self.order_status = self.OrderStatus.reporting_output
            
            self.order_status_info = []
            if output:
                self.order_status_info = [self.action_function, output]
            

        if self.order_status == self.OrderStatus.reporting_output and "ask_for_help" in self.action_function:
            func_arguments_idx = self.action_function.index("(") + 1
            object_id = eval(self.action_function[func_arguments_idx:].split(",")[0])
            
            ob_idx = info['object_key_to_index'][object_id]
            
            weight = robotState.get("objects", "weight", ob_idx)
            
            if weight > 1 and self.movement.help_status == self.movement.HelpState.being_helped and len(self.movement.help_status_info[0]) == weight-1: #If we can pickup alone the object or we already have enough agents helping
                return "collect_object('" + object_id + "')"
            

        ego_location = np.where(robotState.latest_map == 5)
        
        
      
        function_output = "wait()"
        
        #if "collect_object" in self.action_function:
            #pdb.set_trace()
        
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
                        object_idx = info["object_key_to_index"][self.order_status_info[1][0]]
                    except:
                        pdb.set_trace()
                    self.message_text += MessagePattern.item(robotState,object_idx,self.order_status_info[1][0], info, self.env.robot_id, self.env.convert_to_real_coordinates)
                elif "go_to_location" in self.order_status_info[0]:
                    self.message_text += MessagePattern.surroundings(self.order_status_info[1], int(self.env.view_radius), robotState, info, self.env.convert_to_real_coordinates)
            self.order_status = self.OrderStatus.reporting_availability
        elif self.order_status == self.OrderStatus.reporting_availability:
            self.message_text += MessagePattern.order_finished() #This should only be sent once
            self.order_status = self.OrderStatus.finished
            #self.leader_id = ""
            
    
        return function_output
    

    def update_planner(self, robotState):
        agents = []
        objects = []
        
        locations = {'SAFE':(10,10)}
            
        object_weights = {}
        agents_initial_positions = {}
        pd_pb = {}
        
        for ob in robotState.get_all_objects():
            key = str(ob[0])
            objects.append(key)
            ob_location = robotState.get("object_estimates", "last_seen_location", (ob[1] ,robotState.get_num_robots())) #robotState.get("objects", "last_seen_location", ob[1])
            locations[key] = ob_location
            object_weights[key] = ob[2]
            
        for ag in robotState.get_all_robots():
        
            if ag[0] != self.env.robot_id:
        
                agents.append(ag[0])
                ag_location = robotState.get("agents", "last_seen_location", ag[1])
                agents_initial_positions[ag[0]] = ag_location
                pb = robotState.get("agents", "sensor_benign", ag[1])
                pd = robotState.get("agents", "sensor_dangerous", ag[1])
                
                pd_pb[ag[0]] = (pd,pb)
            

        occMap = np.copy(robotState.latest_map)
        
        occMap[occMap > 2] = 0

        if not self.planner:
            self.planner = DynamicSensorPlanner(agents, objects, locations, agents_initial_positions, pd_pb, object_weights, [], occMap, self.extended_goal_coords)
        else:
        
            sql_estimates = robotState.get_all_sensing_estimates()
            estimates = []
        
            for e in sql_estimates:
                estimate_value = robotState.Danger_Status[e[2]].value

                if estimate_value:
                    estimates.append((str(e[0]),e[1],estimate_value-1))
        
            self.planner.update_state(estimates, object_weights, agents_initial_positions, locations, [], occMap)
        
        
        
    def give_new_order(self, agent_id, robotState, info):
    
        if agent_id in self.plan and len(self.plan[agent_id]) > 1 and ('carry' in self.plan[agent_id][1] or 'sense' in self.plan[agent_id][1]):
    
            del self.plan[agent_id][0]
    
            if 'carry' in self.plan[agent_id][0]:
                ob_id = self.plan[agent_id][0].split("_")[1]
                idx = info['object_key_to_index'][ob_id]
                
                weight = robotState.get("objects", "weight", idx)
                
                if weight == 1:
                    self.message_text += MessagePattern.item(robotState,idx,ob_id, info, self.env.robot_id, self.env.convert_to_real_coordinates)
                    self.message_text += MessagePattern.order_collect(agent_id, ob_id)
                    robo_idx = info['robot_key_to_index'][agent_id]
                    robotState.set("agents", "team", int(robo_idx), str([int(robo_idx)]), info["time"])
                
                    
            elif 'sense' in self.plan[agent_id][0]:
                ob_id = self.plan[agent_id][0].split("_")[1]
                idx = info['object_key_to_index'][ob_id]
                location = robotState.get("object_estimates", "last_seen_location", (idx ,robotState.get_num_robots()))
                self.message_text += MessagePattern.order_sense(agent_id, ob_id, location, self.env.convert_to_real_coordinates)
        
    
    def decision(self,messages, robotState, info, output, nearby_other_agents, help_requests):


        print("ORDER:", self.order_status, info["time"])

        if self.order_status == self.OrderStatus.cancelling_order or not self.plan:
            self.update_planner(robotState)
            if not self.plan:
                plan = self.planner.replan()
                for agent_plan in plan:
                    agent_id = agent_plan[0]
                    
                    orders = [p[0] for p in agent_plan[1] if 'SAFE' not in p[0]]
                    
                    #first_order = agent_plan[1][0][0]
                    self.plan[agent_id] = orders #first_order
            else:
                plan = self.planner.replan()


        changed_carry = []
        
        if self.order_status == self.OrderStatus.cancelling_order:
        
            all_agents = [r[0] for r in self.env.map_config['all_robots'] if r[0] != self.env.robot_id]
        
            for agent_plan in plan:
                agent_id = agent_plan[0]
                #first_order = agent_plan[1][0][0]
                orders = [p[0] for p in agent_plan[1] if 'SAFE' not in p[0]]
                
                all_agents.remove(agent_id)
                
                #if first_order != self.plan[agent_id]:
                if agent_id in self.plan.keys():
                    if orders[0] != self.plan[agent_id][0]:
                        if 'carry' in self.plan[agent_id][0]:
                            ob_id = self.plan[agent_id][0].split("_")[1]
                            if ob_id in self.carry_agents:
                                if agent_id in self.carry_agents[ob_id]:
                                    self.carry_agents[ob_id].remove(agent_id)
                       
                        self.message_text += MessagePattern.order_cancel(agent_id)
                        
                    self.previous_plan[agent_id] = self.plan[agent_id]
                    
                self.plan[agent_id] = orders #first_order    
                    
            self.order_status = self.OrderStatus.giving_order
            function_output = "wait()"
            
            for agent_id in all_agents:
                if agent_id in self.plan.keys():
                    del self.plan[agent_id]
                    del self.previous_plan[agent_id]
        
        elif self.order_status == self.OrderStatus.giving_order:
        
            all_agents = [r[0] for r in self.env.map_config['all_robots'] if r[0] != self.env.robot_id]
        
            print("COMPARISON", self.plan, self.previous_plan)
        
            #for agent_plan in plan:
            for agent_id in self.plan.keys():
                #agent_id = agent_plan[0]
                #first_order = agent_plan[1][0][0]
                #if agent_id not in self.plan or first_order != self.plan[agent_id]:
                
                all_agents.remove(agent_id)
                
                #Check that carrying a heavy object has enough agents doing it
                while True:
                    if self.plan[agent_id] and 'carry' in self.plan[agent_id][0]:
                        ob_id = self.plan[agent_id][0].split("_")[1]
                        idx = info['object_key_to_index'][ob_id]
                        weight = robotState.get("objects", "weight", idx)
                        num_helpers = 1
                        if weight > 1:
                            for agent_id2 in self.plan.keys():
                                if agent_id != agent_id2 and self.plan[agent_id][0] == self.plan[agent_id2][0]:
                                    num_helpers += 1
                            if num_helpers < weight:
                                del self.plan[agent_id][0]
                            else:
                                break
                        else:
                            break
                    else:
                        break
                            
                if not self.plan[agent_id]:
                    all_agents.append(agent_id)
                    del self.plan[agent_id]
                    del self.previous_plan[agent_id]
                    continue
                
                if agent_id not in self.previous_plan or self.plan[agent_id][0] != self.previous_plan[agent_id][0]:
                
                    '''
                    if agent_id in self.plan.keys():
                        previous_plan = self.plan[agent_id]
                        if 'carry' in previous_plan:
                            ob_id = previous_plan.split("_")[1]
                            if ob_id in self.carry_agents:
                                if agent_id in self.carry_agents[ob_id]:
                                    self.carry_agents[ob_id].remove(agent_id)
                    try:         
                        self.plan[agent_id] = first_order
                    except:
                        pdb.set_trace()
                    
                    '''
                    
                    self.previous_plan[agent_id] = self.plan[agent_id]
                    
                    if 'carry' in self.plan[agent_id][0]:
                        ob_id = self.plan[agent_id][0].split("_")[1]
                        idx = info['object_key_to_index'][ob_id]
                        
                        weight = robotState.get("objects", "weight", idx)
                        
                        if weight > 1:
                            if ob_id not in self.carry_agents.keys():
                                self.carry_agents[ob_id] = []
                            if agent_id not in self.carry_agents[ob_id]:
                                self.carry_agents[ob_id].append(agent_id)
                                if ob_id not in changed_carry:
                                    changed_carry.append(ob_id)
                        else:
                            self.message_text += MessagePattern.item(robotState,idx,ob_id, info, self.env.robot_id, self.env.convert_to_real_coordinates)
                            self.message_text += MessagePattern.order_collect(agent_id, ob_id)
                            robo_idx = info['robot_key_to_index'][agent_id]
                            robotState.set("agents", "team", int(robo_idx), str([int(robo_idx)]), info["time"])
                        
                            
                    elif 'sense' in self.plan[agent_id][0]:
                        
                        if 'CLUSTER' in self.plan[agent_id][0]:
                            cluster_num = self.plan[agent_id][0].split("CLUSTER")[1]
                            ob_locations = []
                            for ob_id in self.planner.clusters[int(cluster_num)]['objects']:
                                idx = info['object_key_to_index'][ob_id]
                                ob_locations.append(robotState.get("object_estimates", "last_seen_location", (idx ,robotState.get_num_robots())))
                                
                            self.message_text += MessagePattern.order_sense_multiple(agent_id, self.planner.clusters[int(cluster_num)]['objects'], ob_locations, self.env.convert_to_real_coordinates)
                            
                        else:
                            ob_id = self.plan[agent_id][0].split("_")[1]
                            idx = info['object_key_to_index'][ob_id]
                            location = robotState.get("object_estimates", "last_seen_location", (idx ,robotState.get_num_robots()))
                            self.message_text += MessagePattern.order_sense(agent_id, ob_id, location, self.env.convert_to_real_coordinates)
                        
                        
                        
              
            
            for r in self.env.map_config['all_robots']:
                if r[0] != self.env.robot_id:
                    if r[0] in all_agents:
                        self.agent_requesting_order[r[0]] = True
                    else:
                        self.agent_requesting_order[r[0]] = False
                        
            print(self.agent_requesting_order, changed_carry, self.carry_agents)
                       
            for ob in changed_carry:
            
                robot_id = self.carry_agents[ob][0]
                robo_idx = info['robot_key_to_index'][robot_id]
                other_robots_ids = self.carry_agents[ob][1:]
            
                self.message_text += MessagePattern.agent(robot_id, int(robo_idx), robotState, self.env.convert_to_real_coordinates)
                self.message_text += MessagePattern.order_collect_group(robot_id, other_robots_ids, ob)
                
                robo_idxs = [info['robot_key_to_index'][rid] for rid in self.carry_agents[ob]]
                
                for p in robo_idxs:
                    robotState.set("agents", "team", int(p), str([int(p) for p in robo_idxs]), info["time"])       
            

            function_output = "sleep()"
            self.order_status = self.OrderStatus.cancelling_order
        
        return function_output
            
    
