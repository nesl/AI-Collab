import numpy as np
from gym_collab.envs.action import Action
import os
import re
import json
import pdb
from cnl import MessagePattern
from movement_strategy import Movement
from enum import Enum
import time
from collections import deque
from itertools import combinations
import pyAgrum as gum
import math
import random
from process_text_strategy import Human2AIText
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
        self.message_text = self.Message(self.env.robot_id)
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
        self.decide_order = False
        self.commander_order_status = self.OrderStatus.finished
        
        if self.team_structure["hierarchy"][self.env.robot_id] == "order":
            self.commander_order_status = self.OrderStatus.giving_order
            self.decide_order = True
        #else:
        self.order_status = self.OrderStatus.finished
            
        self.order_status_info = []
        self.finished = False
        self.told_to_finish = False
        self.collect_attempts = {}
        self.agent_requesting_order = {r[0]:False for r in robotState.get_all_robots()} # if r[0] != self.env.robot_id}
        self.trigger = False
        self.return_waiting_time = 0
        self.functions_to_execute = []
        self.functions_to_execute_outputs = []
        self.functions_executed = False
        self.room_object_ids = []
        self.help_agent_ids = set()
        self.plan = {}
        self.planner = []
        self.carry_agents = {}
        self.previous_plan = {}
        self.self_messages = []
        self.new_rooms = {}
        self.all_rooms = {}
        self.optimization_metrics = []
        self.providing_plan = []
        self.legacy_plan = {}
        self.waiting_for_response = False
        self.time_last_suggestion_interval = 60 #60
        self.time_last_suggestion = time.time()-self.time_last_suggestion_interval*random.random()
        
        
        self.profiling = {'sensing':[],'check_item': [], 'moving_straight':[],'moving_turn':[],'moving_180_turn':[], 'previous_action':[], 'current_time': 0, 'previous_ego_location': []}
        
        self.return_times = []
        
        if "return" in self.team_structure and bool(self.team_structure["return"][self.env.robot_id]):
            return_time = 5*60
            self.return_times = list(range(return_time,int(self.env.map_config["timer_limit"]), return_time))
        
        
        self.leader_id = ""
        
        
        self.extended_goal_coords = env.goal_coords.copy()

        #self.extended_goal_coords.extend([(g[0]+op[0],g[1]+op[1]) for g in env.goal_coords for op in [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]] if [g[0]+op[0],g[1]+op[1]] not in env.goal_coords])
        
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
            self.rooms = []
            
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
        
    
    class Message():
        def __init__(self, robot_id):
            self.message = []
            self.recipient = []
            self.robot_id = robot_id
            
        def top_insert(self, message, recipient):
            
            if not (isinstance(recipient,str) and recipient == self.robot_id):
                if isinstance(recipient,list):
                    new_recipient_list = []
                    for r in recipient:
                        if r != self.robot_id:
                            new_recipient_list.append(r)
                    
                    if new_recipient_list:
                        self.message.insert(0,message)
                        self.recipient.insert(0,new_recipient_list)
                else:     
                    self.message.insert(0,message)
                    self.recipient.insert(0,recipient)
            
        def insert(self, message, recipient):
        
            if not (isinstance(recipient,str) and recipient == self.robot_id):
                
                if isinstance(recipient,list):
                    new_recipient_list = []
                    for r in recipient:
                        if r != self.robot_id:
                            new_recipient_list.append(r)
                    
                    if new_recipient_list:
                        self.message.append(message)
                        self.recipient.append(new_recipient_list)
                else:         
                    self.message.append(message)
                    self.recipient.append(recipient)
            
        def clear(self):
            self.message = []
            self.recipient = []
            
        def get(self):
            return self.message
            
        def get_str(self):
            return ''.join(self.message)
            
        def search(self,regex):
            for m in range(len(self.message)):
                match_pattern = re.search(regex,self.message[m])
                if match_pattern:
                    return match_pattern,m
                    
            return [],-1
            
        def get_all(self):
            return [self.message.copy(),self.recipient.copy()]
            
        
        
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
                
                if not argument:
                    xy_grid = eval(arguments[1] + ',' + arguments[2])
                    xy_world = self.env.convert_to_real_coordinates(xy_grid)
                    if xy_world:
                        function_description = "I'm going to sense object at location " + str(xy_world) + ". " 
                else:
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
                
                
            elif "ask_for_help_to_carry" in function_str:
                arguments = function_str.split(",")
                argument = eval(arguments[0][arguments[0].find("(") + 1:])
                
                function_description = "I'm going to ask for help to carry object " + str(argument) + ". "
                
                
            elif "ask_for_help" in function_str:
                arguments = function_str.split(",")
                argument = eval(arguments[0][arguments[0].find("(") + 1:])
                argument2 = eval(arguments[1])
                
                function_description = "I'm going to ask agent " + str(argument2) + " for help to carry object " + str(argument) + ". "
                
            elif "ask_for_sensing" in function_str:   

                arguments = function_str.split(",")
                argument = eval(arguments[0][arguments[0].find("(") + 1:])
                argument2 = eval(arguments[1])

                function_description = "I'm going to ask agent " + str(argument2) + " for help to sense object " + str(argument) + ". "


            elif "drop" in function_str:
                function_description = "I'm going to drop an object. "   
        
        except:
            print("Error description")
            pdb.set_trace()
            
            
        return function_description
            
    def self_message_create(self, text_message, robotState, info):
        return (self.env.robot_id, text_message, info['time'])
    
    def message_processing(self,received_messages, robotState, info):
    
        time_log = info["time"]
        
        objects_str = {}
    
        tmp_message_history = []
        translated_messages_index = -1
        
        message_text_save = ""
    
        for rm_idx,rm in enumerate(received_messages):
            
            print("Received message:", rm)
            template_match = False
            tmp_message = rm[1]
            
            agent_idx = info['robot_key_to_index'][rm[0]]
            
            #Ignore any messages sent to itself
            #if rm[0] == str(self.env.robot_id):
            #    message_text_save = self.message_text
            
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
                        
                    return_value,_ = self.movement.message_processing_carry_help_accept(rm, {"weight": num_agents_needed}, self.message_text, following)
                    
                    
                    if return_value == 1:

                        
                        object_location = robotState.get("objects", "last_seen_location", self.chosen_object_idx)
                        if not (object_location[0] == -1 and object_location[1] == -1):
                            self.target_location = object_location
                            self.object_of_interest = object_id
                            #self.target_object_idx = self.heavy_objects["index"][self.chosen_heavy_object]
                            
                            self.action_index = self.State.decision_state
                            
                        else: #Somehow we end here
                            self.message_text.insert(MessagePattern.carry_help_reject(rm[0]),rm[0])
                            
                        

            if re.search(MessagePattern.order_cancel_regex(), rm[1]):
                template_match = True
            
                for rematch in re.finditer(MessagePattern.order_cancel_regex(),rm[1]):
            
                    if "hierarchy" in self.team_structure and (self.team_structure["hierarchy"][self.env.robot_id] == "obey" or (self.team_structure["hierarchy"][self.env.robot_id] == "order" and self.env.robot_id == rm[0])) and self.env.robot_id == rematch.group(1): # and self.leader_id == rm[0]:
                    
                        if self.functions_to_execute_outputs and rm[0] != self.env.robot_id:
                            if "sense_object" in self.action_function:
                
                                objects_reported = []
                                
                                output = []
                                for sf in self.functions_to_execute_outputs:
                                    output.extend(sf)
                                
                                for ob in output:

                                    if not isinstance(ob,list):
                                        continue
                                    try:
                                        object_idx = info["object_key_to_index"][ob[0]]
                                    except:
                                        pdb.set_trace()
                                    #this_time = robotState.items[object_idx]["item_time"][0]
                                    #if this_time >= most_recent:
                                    if ob[0] in objects_reported:
                                        continue
                                    objects_reported.append(ob[0])
                                    
                                    if ob[0] not in self.other_agents[info['robot_key_to_index'][str(self.leader_id)]].items_info_provided:
                                        self.message_text.insert(MessagePattern.item(robotState,object_idx,ob[0], info, self.env.robot_id, self.env.convert_to_real_coordinates), rm[0])
                                        self.other_agents[info['robot_key_to_index'][str(self.leader_id)]].items_info_provided.append(ob[0])
                                        
                            elif "collect_object" in self.action_function:
                            
                                for ob in self.functions_to_execute_outputs:
                            
                                    try:
                                        object_idx = info["object_key_to_index"][ob]
                                    except:
                                        pdb.set_trace()
                                    self.message_text.insert(MessagePattern.item(robotState,object_idx,ob, info, self.env.robot_id, self.env.convert_to_real_coordinates),rm[0])
                            elif "go_to_location" in self.action_function:
                                self.message_text.insert(MessagePattern.surroundings(self.functions_to_execute_outputs, int(self.env.view_radius), robotState, info, self.env.convert_to_real_coordinates),rm[0])
                        
                        
                        self.order_status = self.OrderStatus.finished
                        #self.leader_id = ""
                        self.action_function = ""
                        self.top_action_sequence = 0
                        self.message_text.insert("Ok " + rm[0] + ". I will not fulfill your order. ",rm[0])
                        self.functions_to_execute = []
                        
                        if self.movement.help_status == self.movement.HelpState.being_helped:
                            _,_ = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_finish())
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


            if re.search(MessagePattern.carry_help_regex(),rm[1]) or re.search(MessagePattern.carry_help_participant_affirm_regex(),rm[1]):
            
                if re.search(MessagePattern.carry_help_participant_affirm_regex(),rm[1]):
                    rs = re.search(MessagePattern.carry_help_participant_affirm_regex(),rm[1])
                    rm[1] = rm[1].replace(rs.group(), MessagePattern.carry_help(rs.group(1),1))
            
                rematch = re.search(MessagePattern.carry_help_regex(),rm[1])
                
                template_match = True
                
                #Calculate utility. When to collaborate? -> When to accept or offer collaboration? -> Collaboration score of others and expectation of such collaboration score
                #Finish other strategies
                
                try:
                    self.other_agents[agent_idx].observations.append("Asked me to help carry object " + rematch.group(2))
                except:
                    pdb.set_trace()
                
                #if self.team_structure["role"][self.env.robot_id] != "sensing": 
                self.action_index,_ = self.movement.message_processing_carry_help(rm, robotState, self.action_index, self.message_text)
                
                if MessagePattern.carry_help_accept(rm[0]) in self.message_text.get_str():
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
                
                sent_by_leader = False
                if self.movement.help_status_info[0]:
                    sent_by_leader = self.movement.help_status_info[0][0] == rm[0]
                    
            
                self.action_index,_ = self.movement.message_processing_help(rm, self.action_index, self.helping_type == self.HelpType.sensing, self.State.decision_state, self.message_text)
                
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
                            self.message_text.insert("Why do you want me to follow you? ", rm[0])
                        elif (following_match and following_match.group(1) == str(self.env.robot_id)):
                            self.message_text.insert("Why do you want to follow me? ",rm[0])
                        
                        
                elif MessagePattern.carry_help_cancel() in rm[1]:
                    self.other_agents[agent_idx].observations.append("Cancelled his request for help")
                    
                elif MessagePattern.carry_help_finish() in rm[1]:
                    self.other_agents[agent_idx].observations.append("Finished moving heavy object with help from others")
                
                elif MessagePattern.carry_help_complain() in rm[1]:
                    self.other_agents[agent_idx].observations.append("Dismissed his team for not collaborating effectively")
                    
                
                if MessagePattern.carry_help_cancel() in rm[1] or MessagePattern.carry_help_reject(self.env.robot_id) in rm[1] or MessagePattern.carry_help_finish() in rm[1] or MessagePattern.carry_help_complain() in rm[1]:
                    if sent_by_leader:
                        self.message_text.insert("Ok " + rm[0] + ". I won't help anymore. " ,rm[0])
                        self.order_status = self.OrderStatus.reporting_availability
                        self.action_index = self.State.decision_state
                        self.action_function = ""
                        self.top_action_sequence = 0
                        self.functions_to_execute = []
                    
                        
                elif not re.search(MessagePattern.follow_regex(),rm[1]) and not re.search(MessagePattern.following_regex(),rm[1]):
                    self.message_text.insert("Ok " + rm[0] + ". ", rm[0])
                    
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
                    
                    self.message_text.insert("Ok " + rm[0] + ". Don't help me then. ", rm[0])
                    
                    if self.helping_type == self.HelpType.sensing and self.movement.help_status == self.movement.HelpState.asking and self.movement.help_status_info and rm[0] in self.movement.help_status_info[0]: #HERE
                        self.movement.help_status = self.movement.HelpState.no_request
                    
            if re.search(MessagePattern.location_regex(),rm[1]):
            
                template_match = True
                
                carrying_variable = self.other_agents[agent_idx].carrying
                team_variable = self.other_agents[agent_idx].team
                
                if not (self.movement.help_status == self.movement.HelpState.being_helped and rm[0] in self.movement.help_status_info[0] and self.action_index == self.State.drop_object):            
                    self.action_index,_ = self.movement.message_processing_location(rm, robotState, info, self.other_agents, self.target_location, self.action_index, self.message_text, self.State.decision_state, self.next_loc)
                    
                
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
                
                
            if re.search(MessagePattern.move_request_regex(),rm[1]) or re.search(MessagePattern.move_request_alt_regex(),rm[1]):
                template_match = True
                
                receivers = []
                for rematch in re.finditer(MessagePattern.move_request_regex(),rm[1]):
                    receivers.append(rematch.group(1))
                
                for rematch in re.finditer(MessagePattern.move_request_alt_regex(),rm[1]):
                    receivers.extend(eval(rematch.group(1)))
                    
                    
                if str(self.env.robot_id) in receivers:
                    self.other_agents[agent_idx].observations.append("Asked me to move")
                    self.action_index,_ = self.movement.message_processing_move_request(rm, robotState, info, self.action_index, self.message_text, self.other_agents, self.helping_type == self.HelpType.sensing)
                else:
                    self.other_agents[agent_idx].observations.append("Asked " + rematch.group(1) + " to move")
                

                
                
                    
            if re.search(MessagePattern.sensing_help_regex(),rm[1]): #"What do you know about object " in rm[1]:
                rematch = re.search(MessagePattern.sensing_help_regex(),rm[1])
                
                template_match = True
                
                object_id = rematch.group(1) #rm[1].strip().split()[-1] 
                
                if object_id in info['object_key_to_index']:
                
                    object_idx = info['object_key_to_index'][object_id]
                    
                    self.other_agents[agent_idx].observations.append("Asked me for information about object " + str(object_id))
                    
                    self.message_text.insert(MessagePattern.item(robotState,object_idx,object_id, info, self.env.robot_id, self.env.convert_to_real_coordinates),rm[0])
                    
                    if not self.message_text.get():
                         self.message_text.insert(MessagePattern.sensing_help_negative_response(object_id),rm[0])
                         
                else:
                    self.message_text.insert(MessagePattern.sensing_help_negative_response(object_id),rm[0])
                    
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
                    
                    tmp_message = tmp_message.replace(rematch.group(), "[Object " + object_id + " Info]").strip()
                    
                    if not robotState.get("agents", "type", info['robot_key_to_index'][rm[0]]) and self.planner and rm[0] in self.plan.keys() and self.plan[rm[0]] and "sense" in self.plan[rm[0]][0] and not self.planner.path_monitoring[rm[0]].init_state and self.planner.path_monitoring[rm[0]].moving_finished and not self.planner.path_monitoring[rm[0]].sensing_finished:
                        

                        if "ROOM" in self.plan[rm[0]][0]:
                            room = self.plan[rm[0]][0].split('ROOM')[1]
                            
                            grid_loc = robotState.get("objects", "last_seen_location", info['object_key_to_index'][object_id])
                            room_num = self.env.get_room(grid_loc,True,constrained=False).replace("room ", "")
                        else:
                            room = object_id
                            room_num = self.plan[rm[0]][0].split('_')[1]
                        
                        
                        if room_num == room: #get room location of object
                            self.planner.path_monitoring[rm[0]].update_model(time.time()-self.planner.path_monitoring[rm[0]].initial_time,"sensing")
                            self.planner.path_monitoring[rm[0]].sensing_finished = True
                            print("MODEL TIMES:", rm[0], self.planner.path_monitoring[rm[0]].times)
                            #pdb.set_trace()

                        
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
                            _,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text)
                            self.message_text.insert("Thanks " + rm[0] + ". ",rm[0])
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
                        self.message_text.insert(MessagePattern.agent(rematch.group(1), robot_idx, robotState, self.env.convert_to_real_coordinates),rm[0])
                    else:
                        self.message_text.insert(MessagePattern.agent_not_found(rematch.group(1)),rm[0])
                    
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
        
            if robotState.get("agents", "type", info['robot_key_to_index'][rm[0]]) and re.search(MessagePattern.order_collect_regex(),rm[1]):
            
                template_match = True
            
                if "hierarchy" in self.team_structure and self.team_structure["role"][self.env.robot_id] != "sensing":  
            
                    for rematch in re.finditer(MessagePattern.order_collect_regex(),rm[1]):
                        if rematch.group(1) == self.env.robot_id:
                            if self.team_structure["hierarchy"][self.env.robot_id] == "obey" or (self.team_structure["hierarchy"][self.env.robot_id] == "order" and self.env.robot_id == rm[0]):
                                
                                if rematch.group(2) in info['object_key_to_index']:
                                
                                    if self.order_status == self.OrderStatus.finished:
                                        object_idx = info['object_key_to_index'][rematch.group(2)]
                                        
                                        object_location = robotState.get("objects", "last_seen_location", object_idx)
                                        if (object_location[0] == -1 and object_location[1] == -1): #might be because object is already in goal
                                            self.message_text.insert("I don't know where is object " + rematch.group(2) + ". Provide some information first. ",rm[0])
                                            self.order_status = self.OrderStatus.ongoing
                                        else:
                                        
                                            object_weight = robotState.get("objects", "weight", object_idx)
                                            
                                            
                                            if not object_weight:
                                                self.message_text.insert("First give me the weight value of object " + rematch.group(2) + ". ",rm[0])
                                            else:
                                            
                                            
                                                self.functions_to_execute_outputs = []
                                            
                                                if object_weight > 1 and not (self.movement.help_status == self.movement.HelpState.being_helped and len(self.movement.help_status_info[0]) == object_weight-1):
                                                    self.create_action_function("ask_for_help('" + rematch.group(2) + "','" +  rm[0] + "')")
                                                else:
                                                    self.create_action_function("collect_object('" + rematch.group(2) + "')")
                                                    #pdb.set_trace()
                                                
                                                self.order_status = self.OrderStatus.ongoing
                                                
                                                #But first i NNed to self.movement.help_status
                                                
                                                self.message_text.insert(MessagePattern.order_response(rm[0], "collect"),rm[0])
                                                if self.movement.help_status == self.movement.HelpState.helping:
                                                    self.message_text.insert("I'll do it after I finish helping. ",rm[0])
                                                    
                                                self.leader_id = rm[0]
                                    else:
                                        self.message_text.insert(MessagePattern.order_response_negative(rm[0], self.leader_id),rm[0])
                                else:
                                    self.message_text.insert("I don't know where is object " + rematch.group(2) + ". Provide some information first. ",rm[0])
                                    
                            else:
                                 self.message_text.insert(MessagePattern.order_not_obey(rm[0]),rm[0])
                                
            if robotState.get("agents", "type", info['robot_key_to_index'][rm[0]]) and re.search(MessagePattern.order_sense_regex(),rm[1]):
            
                template_match = True
            
                if "hierarchy" in self.team_structure and self.team_structure["role"][self.env.robot_id] != "lifter":  
                    for rematch in re.finditer(MessagePattern.order_sense_regex(),rm[1]):
                        if rematch.group(1) == self.env.robot_id:
                            if (self.team_structure["hierarchy"][self.env.robot_id] == "obey" and self.team_structure["hierarchy"][rm[0]] == "order") or (self.team_structure["hierarchy"][self.env.robot_id] == "order" and self.env.robot_id == rm[0]):
                            
                                object_id = rematch.group(2)
                                if object_id in info['object_key_to_index'] and info['object_key_to_index'][object_id] in robotState.get_object_keys() and robotState.get("object_estimates", "danger_status", [info['object_key_to_index'][object_id],robotState.get_num_robots()]): #send info if already have it
                                    self.message_text.insert(MessagePattern.item(robotState,info['object_key_to_index'][object_id],object_id, info, self.env.robot_id, self.env.convert_to_real_coordinates),rm[0])
                                    self.order_status = self.OrderStatus.ongoing
                                else:
                                    if self.order_status == self.OrderStatus.finished:
                                    
                                        self.functions_to_execute_outputs = []
                                        
                                        location = list(eval(rematch.group(3)))
                            
                                        max_real_coords = self.env.convert_to_real_coordinates((robotState.latest_map.shape[0]-1, robotState.latest_map.shape[1]-1))
                                        
                                        if location[0] > max_real_coords[0] or location[1] > max_real_coords[1]: #last_seen[0] == 99.99 and last_seen[1] == 99.99:
                                            assigned_target_location = robotState.get("objects", "last_seen_location", info['object_key_to_index'][object_id])
                                        else:
                                            assigned_target_location = self.env.convert_to_grid_coordinates(location)
                                        
                                        self.create_action_function("sense_object(''," +  str(assigned_target_location) + ")")
                                        
                                        self.order_status = self.OrderStatus.ongoing
                                        self.message_text.insert(MessagePattern.order_response(rm[0], "sense"),rm[0])
                                        self.leader_id = rm[0]
                                    else:
                                        self.message_text.insert(MessagePattern.order_response_negative(rm[0], self.leader_id),rm[0])
                            else:
                                 self.message_text.insert(MessagePattern.order_not_obey(rm[0]),rm[0])
            
            if robotState.get("agents", "type", info['robot_key_to_index'][rm[0]]) and re.search(MessagePattern.order_sense_room_regex(),rm[1]):
            
                template_match = True
            
                if "hierarchy" in self.team_structure and self.team_structure["role"][self.env.robot_id] != "lifter":  
                    for rematch in re.finditer(MessagePattern.order_sense_room_regex(),rm[1]):
                        if rematch.group(1) == self.env.robot_id:
                            if (self.team_structure["hierarchy"][self.env.robot_id] == "obey" and self.team_structure["hierarchy"][rm[0]] == "order") or (self.team_structure["hierarchy"][self.env.robot_id] == "order" and self.env.robot_id == rm[0]):
                            
                                room = rematch.group(2)
                                
                                if self.order_status == self.OrderStatus.finished:
                                
                                    self.functions_to_execute_outputs = []
                                    
                                    self.create_action_function("sense_room(\""+ str(room) +"\")")
                                    
                                    self.order_status = self.OrderStatus.ongoing
                                    self.message_text.insert(MessagePattern.order_response(rm[0], "sense"),rm[0])
                                    self.leader_id = rm[0]
                                else:
                                    self.message_text.insert(MessagePattern.order_response_negative(rm[0], self.leader_id),rm[0])
                            else:
                                 self.message_text.insert(MessagePattern.order_not_obey(rm[0]),rm[0])
                     
            if re.search(MessagePattern.order_sense_multiple_regex(),rm[1]):
            
                template_match = True
            
                if "hierarchy" in self.team_structure and self.team_structure["role"][self.env.robot_id] != "lifter":  
                    for rematch in re.finditer(MessagePattern.order_sense_multiple_regex(),rm[1]):
                        if rematch.group(1) == self.env.robot_id:
                            if (self.team_structure["hierarchy"][self.env.robot_id] == "obey" and self.team_structure["hierarchy"][rm[0]] == "order") or (self.team_structure["hierarchy"][self.env.robot_id] == "order" and self.env.robot_id == rm[0]):
                            
                                print("MULTIPLE REQUEST")
                                object_ids = rematch.group(2)[1:].replace("]", "").split(",")
                                
                                coords_regex = '\( *-?\d+(\.(\d+)?)? *, *-?\d+(\.(\d+)?)? *\)'
                                
                                locations = []
                                
                                for coord in re.finditer(coords_regex,rematch.group(4)):
                                    locations.append(eval(coord.group()))
                                
                                delete_objs = []
                                for ob_idx,ob in enumerate(object_ids):
                                    if ob in info['object_key_to_index'] and info['object_key_to_index'][ob] in robotState.get_object_keys() and robotState.get("object_estimates", "danger_status", [info['object_key_to_index'][ob],robotState.get_num_robots()]): #send info if already have it
                                        self.message_text.insert(MessagePattern.item(robotState,info['object_key_to_index'][ob],ob, info, self.env.robot_id, self.env.convert_to_real_coordinates) ,rm[0])
                                        
                                        delete_objs.append(ob_idx)
                                
                                print(object_ids, delete_objs)
                                delete_objs.reverse()   
                                for ob_idx in delete_objs:
                                    del object_ids[ob_idx]
                                    del locations[ob_idx]
                                
                                print(object_ids, delete_objs)   

                                if not object_ids:
                                    self.order_status = self.OrderStatus.ongoing
                                else:
                                    if self.order_status == self.OrderStatus.finished:
                                        
                                        max_real_coords = self.env.convert_to_real_coordinates((robotState.latest_map.shape[0]-1, robotState.latest_map.shape[1]-1))
                                        self.functions_to_execute_outputs = []
                                        #assigned_target_locations = []
                                        for l_idx,location in enumerate(locations):
                                        
                                            if location[0] > max_real_coords[0] or location[1] > max_real_coords[1]: #last_seen[0] == 99.99 and last_seen[1] == 99.99:
                                                assigned_target_location = robotState.get("objects", "last_seen_location", info['object_key_to_index'][object_ids[l_idx]])
                                            else:
                                                assigned_target_location = self.env.convert_to_grid_coordinates(location)
                                            
                                            #assigned_target_locations.append(assigned_target_location)
                                            self.functions_to_execute.append("sense_object('" + object_ids[l_idx] + "'," +  str(assigned_target_location) + ")")
                                            
                                        #self.create_action_function("sense_object(''," +  str(assigned_target_locations) + ")")
                                        self.order_status = self.OrderStatus.ongoing
                                        self.message_text.insert(MessagePattern.order_response(rm[0], "sense"),rm[0])
                                        self.leader_id = rm[0]
                                        
                                        self.movement.cancel_cooperation(self.State.decision_state,self.message_text)
                                        self.action_function = ""
                                        self.action_index = self.State.decision_state
                                        self.functions_executed = True
                                    else:
                                        self.message_text.insert(MessagePattern.order_response_negative(rm[0], self.leader_id),rm[0])
                            else:
                                 self.message_text.insert(MessagePattern.order_not_obey(rm[0]),rm[0])
                        
            if re.search(MessagePattern.order_explore_regex(),rm[1]):
            
                template_match = True
            
                if "hierarchy" in self.team_structure:  
                    for rematch in re.finditer(MessagePattern.order_explore_regex(),rm[1]):
                        if rematch.group(1) == self.env.robot_id:
                            if (self.team_structure["hierarchy"][self.env.robot_id] == "obey" and self.team_structure["hierarchy"][rm[0]] == "order") or (self.team_structure["hierarchy"][self.env.robot_id] == "order" and self.env.robot_id == rm[0]):
                                if self.order_status == self.OrderStatus.finished:
                                
                                    self.functions_to_execute_outputs = []
                                
                                    assigned_target_location = self.env.convert_to_grid_coordinates(eval(rematch.group(2)))
                                    self.create_action_function("go_to_location(" +  str(assigned_target_location) + ")")
                                    
                                    self.order_status = self.OrderStatus.ongoing
                                    self.message_text.insert(MessagePattern.order_response(rm[0], "explore"),rm[0])
                                    self.leader_id = rm[0]
                                else:
                                    self.message_text.insert(MessagePattern.order_response_negative(rm[0], self.leader_id),rm[0])
                            else:
                                 self.message_text.insert(MessagePattern.order_not_obey(rm[0]),rm[0])
                        
            if robotState.get("agents", "type", info['robot_key_to_index'][rm[0]]) and re.search(MessagePattern.order_collect_group_regex(),rm[1]):
            
                template_match = True
                
                if "hierarchy" in self.team_structure and self.team_structure["role"][self.env.robot_id] != "sensing":
            
                    for rematch in re.finditer(MessagePattern.order_collect_group_regex(),rm[1]):
                    
                        teammates = rematch.group(2)[1:].replace("]", "").split(",")
                        
                        if rematch.group(1) == self.env.robot_id or self.env.robot_id in teammates:
                        
                            if (self.team_structure["hierarchy"][self.env.robot_id] == "obey" and self.team_structure["hierarchy"][rm[0]] == "order") or (self.team_structure["hierarchy"][self.env.robot_id] == "order" and self.env.robot_id == rm[0]): 
                                if self.order_status == self.OrderStatus.finished:
                                
                                    self.functions_to_execute_outputs = []
                                
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
                                            self.message_text.insert("I don't know where is object " + rematch.group(2) + ". Provide some information first. ",rm[0])
                                            self.order_status = self.OrderStatus.ongoing
                                        else:
                                        
                                            self.create_action_function("collect_object('" + rematch.group(4) + "')")
                                            match_pattern,m_idx = self.message_text.search(MessagePattern.location_regex())
                                            
                                            if match_pattern and not match_pattern.group(7):
                                                self.message_text.message[m_idx] = self.message_text.message[m_idx].replace(match_pattern.group(), match_pattern.group() + " Helping " + self.env.robot_id + ". ")

                                            self.order_status = self.OrderStatus.ongoing
                                        
                                            
                                    elif self.env.robot_id in teammates:

                                        self.movement.help_status = self.movement.HelpState.helping
                                        
                                        self.movement.help_status_info[0] = [rematch.group(1)]
                                        
                                        teammates_sub = teammates.copy()
                                        teammates_sub.remove(self.env.robot_id)
                                        
                                        self.movement.help_status_info[6].extend(teammates_sub)
                                        
                                        self.action_index = self.movement.State.follow
                                        
                                        self.order_status = self.OrderStatus.ongoing
                                        
                                        print("TEAMMATE HERE", self.movement.help_status_info)
                                    
                                    self.leader_id = rm[0]
                                    self.message_text.insert(MessagePattern.order_response(rm[0], "collect"),rm[0])
                
                                else:
                                    self.message_text.insert(MessagePattern.order_response_negative(rm[0], self.leader_id),rm[0])
                            else:
                                 self.message_text.insert(MessagePattern.order_not_obey(rm[0]),rm[0])
                        
                        
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
                                _,_ = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_finish())
                            else:
                                self.movement.help_status = self.movement.HelpState.no_request
                                self.movement.help_status_info[0] = []
                            
                        elif t != robot_idx:
                        
                            robot_id = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(t)]
                            self.message_text.insert(MessagePattern.order_cancel(robot_id),robot_id)
                            
                    
                        
                    self.message_text.insert("Understood " + rm[0] + ". ",rm[0])
                    
                
                        
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
                
                for x in range(min_x,max_x+1):
                    for y in range(min_y,max_y+1):
                        if (x,y) in robotState.saved_locations.keys():
                            del robotState.saved_locations[(x,y)]
                
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
                
                
            if re.search(MessagePattern.order_sense_room_empty_regex(),rm[1]) or MessagePattern.order_sense_room_empty_alt() in rm[1]:  
            
                template_match = True
            
                if "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "order":
                    
                    rematch = re.search(MessagePattern.order_sense_room_empty_regex(),rm[1])
                                       
                    if (self.team_structure["hierarchy"][rm[0]] == "obey" or (self.team_structure["hierarchy"][self.env.robot_id] == "order" and self.env.robot_id == rm[0])): 
                        
                        room_num = ""
                        if rematch:
                            room_num = str(rematch.group(1))
                        elif self.plan and rm[0] in self.plan and "ROOM" in self.plan[rm[0]][0]:
                            room_num = self.plan[rm[0]][0].split("ROOM")[1]
                            
                        if room_num in self.new_rooms.keys():
                            del self.new_rooms[room_num]
                            
                        
                        if not robotState.get("agents", "type", info['robot_key_to_index'][rm[0]]) and self.planner and self.planner.path_monitoring[rm[0]] and self.plan and rm[0] in self.plan and "sense" in self.plan[rm[0]][0] and self.planner.path_monitoring[rm[0]].moving_finished and not self.planner.path_monitoring[rm[0]].sensing_finished:
                        
                        
                            self.planner.path_monitoring[rm[0]].update_model(time.time()-self.planner.path_monitoring[rm[0]].initial_time,"sensing")
                            self.planner.path_monitoring[rm[0]].sensing_finished = True
                            print("MODEL TIMES:", rm[0], self.planner.path_monitoring[rm[0]].times)
                            #pdb.set_trace()
                    
                
            #What happens if a human lies about room being empty?
            if re.search(MessagePattern.order_finished_regex(),rm[1]):  
            
                template_match = True
            
                if "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "order" and not re.search(MessagePattern.order_response_regex(),rm[1]):
                    robot_idx = info['robot_key_to_index'][rm[0]]
                    
                    if not self.env.robot_id == rm[0]:
                        self.other_agents[robot_idx].previous_assignment  = self.other_agents[robot_idx].assignment
                        self.other_agents[robot_idx].assignment = "" 
                    
                    robotState.set("agents", "team", robot_idx, "[]", info["time"])
                    
                    if (self.team_structure["hierarchy"][rm[0]] == "obey" or (self.team_structure["hierarchy"][self.env.robot_id] == "order" and self.env.robot_id == rm[0])) and not (self.commander_order_status == self.OrderStatus.giving_order and rm[0] in self.plan.keys() and self.plan[rm[0]][0] != self.previous_plan[rm[0]][0]): #If there is no human or the human is obeying #(not robotState.get("agents", "type", robot_idx) and self.team_structure["hierarchy"][rm[0]] == "obey") all(robotState.get("agents", "type", -1)) 
                        self.agent_requesting_order[rm[0]] = True
                        
                        print(self.agent_requesting_order)
                        
                        
                        if rm[0] in self.plan.keys() and self.plan[rm[0]]:
                            success = 0
                            if "sense" in self.plan[rm[0]][0]: #What if object in an other room is sensed
                                if "ROOM" in self.plan[rm[0]][0]:
                                    room = self.plan[rm[0]][0].split('ROOM')[1]
                                    
                                    objects_in_room = self.planner.rooms[room]
                                    
                                    if objects_in_room:
                                        results = []
                                        for object_id in objects_in_room:
                                            danger_status = robotState.get("object_estimates", "danger_status", [info['object_key_to_index'][object_id],robot_idx])
                                            results.append(bool(danger_status))
                                        
                                        try:
                                            success = sum(results)/len(results)
                                        except:
                                            pdb.set_trace()
                                    elif room not in self.new_rooms.keys(): #room is empty
                                        success = 1
                                    else:
                                        danger_statuses = robotState.get_object_in_room("danger_status", ["room " + room,rm[0]])
                                        success = 0
                                        if danger_statuses:
                                            if any(bool(d) for d in danger_statuses):
                                                success = 1
                                      
                                            
                                            
                                else:
                                    #room = object_id
                                    object_id = self.plan[rm[0]][0].split('_')[1]
                                    
                                    danger_status = robotState.get("object_estimates", "danger_status", [info['object_key_to_index'][object_id],robot_idx])
                                
                                
                                    success = int(bool(danger_status))
                                    
                            elif "carry" in self.plan[rm[0]][0]:
                                object_id = self.plan[rm[0]][0].split('_')[1]
                                
                                current_object_location = robotState.get("objects", "last_seen_location", info['object_key_to_index'][object_id])
                                
                                current_room = self.env.get_room(current_object_location,True,constrained=False)
                                
                                if current_room == "goal area":
                                    success = 1
                                else:
                                
                                    object_location = self.planner.locations[object_id]
                                    try:
                                        goal_area = self.planner.locations["SAFE"]
                                    except:
                                        pdb.set_trace()                                    
                                    prior_path = len(self.movement.findPath(np.array(object_location),np.array(goal_area),robotState.latest_map))
                                    
                                    
                                    
                                    current_path = len(self.movement.findPath(np.array(current_object_location),np.array(goal_area),robotState.latest_map))
                                    
                                    if not prior_path:
                                        success = 1
                                    else:
                                        success = 1 - max(current_path/prior_path,1)
                            else:
                                pdb.set_trace()
                            
                            task = self.planner.node_type[self.plan[rm[0]][0]]
                            self.planner.path_monitoring[rm[0]].update_reliability(task, success)
                            print("UPDATED reliability:", task, success, rm[0], self.planner.path_monitoring[rm[0]].reliability[task])
                        
                        if rm[0] in self.plan.keys() and self.plan[rm[0]] and 'carry' in self.plan[rm[0]][0]:
                        
                            ob_id = self.plan[rm[0]][0].split("_")[1]
                            idx = info['object_key_to_index'][ob_id]
                            weight = robotState.get("objects", "weight", idx)
                            
                            if weight > 1:
                                carrying_agents = [s for s in self.plan.keys() if s in self.plan.keys() and self.plan[s] and self.plan[s][0] == self.plan[rm[0]][0]]
                                
                                if ob_id in self.carry_agents:
                                    del self.carry_agents[ob_id]
                            else:
                                carrying_agents = [rm[0]]
                            
                            for c in carrying_agents:
                            
                                if c != rm[0]:
                                    self.message_text.insert(MessagePattern.order_cancel(c),c)
                            
                                print("Deleted", c, self.plan[c][0])
                                del self.plan[c][0]
                                
                                if not self.plan[c]:
                                    del self.plan[c]
                                    del self.previous_plan[c]
                                            
                        
                        self.decide_order = True
                    
                        '''
                        if not all(self.agent_requesting_order[r] for r in self.agent_requesting_order.keys()):
                            self.give_new_order(rm[0], robotState, info)
                        '''
                    self.message_text.insert("Thanks " + rm[0] + ". ",rm[0])
                    
                    
                      
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
                        self.message_text.insert(MessagePattern.item(robotState,info['object_key_to_index'][object_id],object_id, info, self.env.robot_id, self.env.convert_to_real_coordinates),rm[0])
                    else:
                    
                        self.action_index,_ = self.movement.message_processing_carry_help(rm, robotState, self.action_index, self.message_text)
                        
                        if MessagePattern.carry_help_accept(rm[0]) in self.message_text.get_str(): #not robotState.object_held and self.movement.help_status == self.movement.HelpState.no_request and self.sense_request == self.State_Sense_Request.no_request:
                        
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
                        
                    self.message_text.insert("Let's finish, " + rm[0] + ". ",rm[0])

                        
                    print("FINISHING BY ORDER")
                else:
                    if not self.finished and "go_to_meeting_point" not in self.action_function: 
                        self.message_text.insert("I haven't finished yet. ",rm[0])
                    elif [ego_location[0][0],ego_location[1][0]] not in self.ending_locations:
                        self.message_text.insert("Let's finish. Let's go to the final meeting location, come with me. ",rm[0])
                    else:
                        self.message_text.insert("Let's finish. ",rm[0])
                    
                print("Finished other agent")
                    
                    
            if re.search(MessagePattern.finish_reject_regex(),rm[1]): #MessagePattern.finish_reject() in rm[1]:
            
                template_match = True
            
                robot_idx = info['robot_key_to_index'][rm[0]]
                
                self.other_agents[robot_idx].finished = False


            if re.search(MessagePattern.come_closer_regex(),rm[1]) or re.search(MessagePattern.come_closer_alt_regex(),rm[1]):
                template_match = True
            
                receivers = []
                for rematch in re.finditer(MessagePattern.come_closer_regex(),rm[1]):
                    receivers.append(rematch.group(1))
                
                for rematch in re.finditer(MessagePattern.come_closer_alt_regex(),rm[1]):
                    receivers.extend(eval(rematch.group(1)))
                    
                #rematch = re.search(MessagePattern.come_closer_regex(),rm[1])
                
                    
                
                if str(self.env.robot_id) in receivers:
                    if self.action_index == self.movement.State.wait_random:
                        self.action_index = self.movement.last_action_index
                        self.pending_location = []
                    self.message_text.insert("Ok " + rm[0] + ". ",rm[0])
            
            if re.search(MessagePattern.sensing_ask_help_incorrect_regex(),rm[1]):
                template_match = True    
                
            if re.search(MessagePattern.ask_for_object_regex(),rm[1]):
                template_match = True 
            
                for rematch in re.finditer(MessagePattern.ask_for_object_regex(),rm[1]):
                    
                    if rematch.group(1) in info['object_key_to_index'].keys():
                        object_idx = info['object_key_to_index'][rematch.group(1)]
                        object_location = robotState.get("objects", "last_seen_location", object_idx)
                        if not (object_location[0] == -1 and object_location[1] == -1):
                            self.message_text.insert(MessagePattern.item(robotState,object_idx,rematch.group(1), info, self.env.robot_id, self.env.convert_to_real_coordinates),rm[0])
                    else:
                        pdb.set_trace()
                        
            if re.search(MessagePattern.updates_regex(),rm[1]):
                template_match = True 
            
                for rematch in re.finditer(MessagePattern.updates_regex(),rm[1]):
                    
                    if rematch.group(1) == self.env.robot_id:
                        self.message_text.insert(MessagePattern.report_progress(),rm[0])
            
            if robotState.get("agents", "type", info['robot_key_to_index'][rm[0]]) and MessagePattern.plan_evaluation_bad() in rm[1]:
            
                template_match = True
                
                if "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "obey":
                    self.time_last_suggestion = time.time()
            
            #template_match = True #CNL only
            
            if not template_match and translated_messages_index >= 0 and translated_messages_index >= rm_idx: #This means the translated message doesn't make sense
                print("understand here 1")

                self.message_text.insert("I didn't understand you " + rm[0] + ". ",rm[0])
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
                
                if "hierarchy" in self.team_structure and (self.team_structure["hierarchy"][self.env.robot_id] == "order" or self.team_structure["hierarchy"][self.env.robot_id] == "obey"):
                    strategy_query = True
                else:
                    strategy_query = False

                if self.waiting_for_response:
                    pass
                    
                translated_message,message_to_user,functions = self.human_to_ai_text.convert_to_ai(rm[0], rm[1], info, robotState, list(self.message_history)[-20:], True, asking_for_help, strategy_query, self.plan)
                '''
                try:
                
                    translated_message,message_to_user,functions = self.human_to_ai_text.convert_to_ai(rm[0], rm[1], info, robotState, self.message_history[-20:], True, asking_for_help, strategy_query)
                    
                except:
                    translated_message = ""
                    message_to_user = ""
                    functions = []
                    #pdb.set_trace()
                    print("Error in translation")
                '''
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
                        #pdb.set_trace()
                        
                        """
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
                        """
                        
                        skip_states = []
                        skip_states2 = []
                        plan_similarities = []
                        if "hierarchy" in self.team_structure:
                        
                            carry_functions = set()
                            for agent_key in functions.keys():
                                if "carry" in functions[agent_key][0]:
                                    carry_functions.add(functions[agent_key][0])
                                
                                if agent_key in self.plan and self.plan[agent_key] and functions[agent_key][0] == self.plan[agent_key][0]:
                                    plan_similarities.append(True)
                                else:
                                    plan_similarities.append(False)
                                   
                            if self.team_structure["hierarchy"][self.env.robot_id] == "order" and not all(plan_similarities): 
                                skip_states = []
                                skip_states2 = []
                                for sp in self.plan.keys():
                                    if 'carry' in self.plan[sp][0] and not self.agent_requesting_order[sp]:
                                        skip_states.append([self.plan[sp][0],sp])
                                        if self.plan[sp][0] not in carry_functions:
                                            skip_states2.append([self.plan[sp][0],sp])
                 
                        
                        obeying = "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "obey"
                        objective_explanation = ""
                        
                        time_limit_suggestion = (time.time() - self.time_last_suggestion) < self.time_last_suggestion_interval
                        
                        if obeying:
                            add_functions = {}
                            for f in functions.keys():
                                if "leader" in functions[f]:
                                    led = functions[f].split("_")[3]
                                    if led not in functions.keys():
                                        add_functions[led] = functions[f]
                                        
                            for af in add_functions.keys():
                                functions[af] = add_functions[af]
                        
                        
                        if not all(plan_similarities) and (not (obeying and self.env.robot_id not in functions.keys()) or (obeying and not time_limit_suggestion)):
          
                            self.update_planner(robotState,skip_states)
                            
                            any_feasible_action = False
                            
                            final_function_descriptions = ""
                            leader = ""
                            
                            agent_function_leader = {}
                            for agent_key in functions.keys():
                                for agent_function in functions[agent_key]:
                                    if "sense" in agent_function or "carry" in agent_function:
                                    
                                        '''
                                        skip_states = []
                                        for sp in self.plan.keys():
                                            if sp != rm[0]:
                                                skip_states.append([self.plan[sp][0],sp])
                                        '''     
                                        
                                        feasible_action = False
                                        explanation = ""
                                        leader = ""
                                        function_description = ""
                                        
                                        if "sense" in agent_function:
                                        
                                            ob_room = agent_function.split("_")[1]
                                            
                                            if "ROOM" not in agent_function and ob_room not in self.planner.rooms["extra"]:
                                                room_num = str(-1)
                                                for r_key in self.planner.rooms.keys():
                                                    if ob_room in self.planner.rooms[r_key]:
                                                        room_num = r_key
                                                        break
                                                agent_function = "sense_ROOM" + room_num
                                            
                                            new_ob_room = agent_function.split("_")[1]
                                            
                                            if "ROOM" in agent_function:
                                                function_description = "Agent " + agent_key + " should sense room " + agent_function.split("ROOM")[1] #+ ". "
                                            else:
                                                function_description = "Agent " + agent_key + " should sense object " + agent_function.split("_")[1] #+ ". "
                                                
                                            if not ("ROOM" in agent_function and agent_function.split("ROOM")[1] not in self.planner.rooms.keys()) and (rm[0],new_ob_room) not in self.planner.sensed_clusters and agent_function in self.planner.nodes:
                                                feasible_action = True
                                            elif (rm[0],new_ob_room) in self.planner.sensed_clusters:
                                                explanation = "We have already sensed that object/area " + new_ob_room + ". "
                                                if obeying:
                                                    feasible_action = True
                                            elif "ROOM" in agent_function and agent_function.split("ROOM")[1] not in self.planner.rooms.keys():
                                                if "ROOM" not in ob_room:
                                                    explanation = "I have no knowledge about object " + ob_room + ". "
                                                else:
                                                    explanation = "Room " + ob_room.split("ROOM")[1] + " doesn't seem to exist. "
                                        
                                        elif "carry" in agent_function:
                                            object_id = agent_function.split("_")[1]
                                            
                                            function_description = "Agent " + agent_key + " should carry object " + object_id #+ ". "
                                            
                                            ob_location = []
                                            if object_id in info['object_key_to_index'].keys():
                                                object_idx = info['object_key_to_index'][object_id]
                                                ob_location = robotState.get("objects", "last_seen_location", object_idx)
                                            
                                            if "leader" in agent_function: #TODO
                                            
                                                leader = agent_function.split("_")[3]
                                            
                                                agent_function = "carry_" + object_id
                                                
                                                agent_function_leader[leader] = "carry_" + object_id
                                                    
                                                if not ob_location or (leader == self.env.robot_id and ob_location[0] == -1 and ob_location[1] == -1):
                                                    explanation = "I don't know where that object is. "
                                                elif obeying:
                                                    feasible_action = True
                                            elif object_id not in self.planner.objects_to_carry:
                                                if not ob_location or (ob_location[0] == -1 and ob_location[1] == -1):
                                                    explanation = "I don't know where that object is. "
                                                else:
                                                    explanation = "We still don't have enough certainty over object " + object_id + ", so far it is " + str(round(self.planner.belief[object_id]*100,2)) + "% of being dangerous. "
                                                        
                                                    if obeying:
                                                        feasible_action = True
                                            elif object_id in self.planner.already_carried:
                                                explanation = "We have already disposed of object " + object_id + ". "
                                                if obeying:
                                                    feasible_action = True
                                            
                                            elif agent_function in self.planner.nodes:
                                                if self.planner.object_weights[object_id] > 1:
                                                    if not obeying and len(self.planner.agents) - len(skip_states2) >= self.planner.object_weights[object_id]:
                                                        feasible_action = True
                                                    elif obeying:
                                                        explanation = "Who is going to be the leader when carrying object " + object_id + ". "
                                                        
                                                else:
                                                    feasible_action = True
                                            else:
                                                pdb.set_trace()
                                                explanation = "We don't have any information about object " + object_id + ". "
                                                
                                        
                                        if explanation in objective_explanation:
                                            explanation = ""
                                        
                                        objective_explanation += explanation
                                        
                                        if final_function_descriptions:
                                            final_function_descriptions += ", "
                                        
                                        final_function_descriptions += function_description
                                        
                                        if not feasible_action:
                                            break
                                        else:
                                            skip_states2.append([agent_function,agent_key, leader])
                                            any_feasible_action = True
                                            
                                            break
                                
                            
                            
                            for skp2 in skip_states2:
                                if skp2[1] in agent_function_leader.keys():
                                    del agent_function_leader[skp2[1]]
                            
                            for afl in agent_function_leader.keys():
                                skip_states2.append([agent_function_leader[afl],afl, afl])
                            
                            plan_description = ""


                            if not (obeying and time_limit_suggestion):
                            
                                self.time_last_suggestion = time.time()
                            
                                obeying_present = False
                                if obeying:
                                    proposed_plan,proposed_objectives = self.planner.replan([], pretend=True)
                                    plan_description = self.get_plan_description(proposed_plan)
                                    if any(a[1] == self.env.robot_id for a in skip_states2):
                                        obeying_present = True
                                    
                                if any_feasible_action:
                                
                                    #skip_states.append([functions[0],rm[0]])
                                    
                                    if obeying:
                                        to_carry = []
                                        for sk in skip_states2:
                                            if "carry" in sk[0] and sk[0].split("_")[1] not in to_carry and sk[0].split("_")[1] not in self.planner.objects_to_carry:
                                                to_carry.append(sk[0].split("_")[1])
                                        if to_carry:
                                            self.update_planner(robotState,skip_states, objects_to_carry=to_carry)
                                    
                                    plan2,objectives = self.planner.replan(skip_states2, pretend=True)
                                    
                                    if plan2:
                                        overall_objective = 0
                                        previous_overall_objective = 0
                                        overall_per_objective = []
                                        
                                        if obeying:
                                            optimization_metrics_comparison = proposed_objectives
                                        else:
                                            optimization_metrics_comparison = self.optimization_metrics
                                        
                                        ignored_objectives = ['completeness','reliability','agent utilization']
                                        for objc_idx in range(len(objectives[0])):
                                        
                                            if obeying and objectives[2][objc_idx] not in ignored_objectives:
                                            
                                                overall_objective += objectives[0][objc_idx]*objectives[1][objc_idx]
                                                previous_overall_objective += optimization_metrics_comparison[0][objc_idx]*optimization_metrics_comparison[1][objc_idx]
                                                
                                                if obeying:
                                                    overall_per_objective.append(optimization_metrics_comparison[0][objc_idx] > objectives[0][objc_idx])
                                                else:
                                                    overall_per_objective.append(optimization_metrics_comparison[0][objc_idx] >= objectives[0][objc_idx])
                                                    
                                                explanation = ""
                                                
                                                name = objectives[2][objc_idx]
                                                
                                                if name == 'distance to travel':
                                                    if overall_per_objective[-1]:
                                                        explanation = "You are reducing how much agents need to move. "
                                                    else:
                                                        explanation = "You are making agents move too much. "
                                                elif name == 'uncertainty reduction':
                                                    if overall_per_objective[-1]:
                                                        explanation = "You are helping reduce uncertainty over objects' danger status. "
                                                    else:
                                                        explanation = "You are not helping reduce uncertainty over objects' danger status. "
                                                elif name == 'reliability':
                                                    if overall_per_objective[-1]:
                                                        explanation = "There is a reduction on the utilization of unrealiable agents. "
                                                    else:
                                                        explanation = "There is no reduction on the utilization of unrealiable agents. "
                                                elif name == "completeness":
                                                    if overall_per_objective[-1]:
                                                        explanation = "There is a increase in the completeness of the plan. "
                                                    else:
                                                        explanation = "There is a decrease in the completeness of the plan. "
                                                
                                                elif name == "agent utilization":
                                                    if overall_per_objective[-1]:
                                                        explanation = "There is a increase in the agent utilization of the plan. "
                                                    else:
                                                        explanation = "There is a decrease in the agent utilization of the plan. "
                                                        
                                                objective_explanation += explanation
                                                    
                                        
                                        print(overall_objective, previous_overall_objective, overall_per_objective)
                                        if sum(overall_per_objective) >= len(overall_per_objective)/2:
                                        
                                            if not obeying:
                                                objective_explanation = "Your plan seems good: [" + final_function_descriptions + "]. " + objective_explanation + "Let's do that. "
                                                self.providing_plan,self.optimization_metrics = plan2,objectives
                                                self.agent_requesting_order[rm[0]] = True
                                                
                                                for p in plan2:
                                                    self.planner.set_monitoring(p[0],p[1][0][0],skip_states)
                                            else:
                                                objective_explanation = self.human_to_ai_text.noop
                                            
                                       
                                            
                                            
                                        else:
                                            if not obeying:
                                                objective_explanation = MessagePattern.plan_evaluation_bad() + "[" + final_function_descriptions + "]. " + objective_explanation + "Let's stick with the previous plan. "
                                            else:
                                                
                                                if self.env.robot_id not in functions.keys():
                                                    extra_explanation = ""
                                                else:
                                                    extra_explanation = "I will follow your orders, although "
                                                
                                                objective_explanation = MessagePattern.plan_evaluation_bad() + "[" + final_function_descriptions + "]. " + objective_explanation + extra_explanation + "I think you may consider a better plan: [" + plan_description + "] "
                                    else:
                                        if not obeying:
                                            objective_explanation = MessagePattern.plan_evaluation_bad() + "[" + final_function_descriptions + "]. " + explanation + "Let's stick with the previous plan. "
                                        else:
                                            if self.env.robot_id not in functions.keys():
                                                extra_explanation = ""
                                            else:
                                                extra_explanation = "I will follow your orders, although "
                                            objective_explanation = MessagePattern.plan_evaluation_bad() + "[" + final_function_descriptions + "]. " + objective_explanation + extra_explanation + "I think you may consider a better plan: [" + plan_description + "] "
                            
                                    if obeying and obeying_present and not (self.env.robot_id in self.plan and self.plan[self.env.robot_id] and self.plan[self.env.robot_id][0] == functions[self.env.robot_id][0]):                                
                                        self.movement.cancel_cooperation(self.State.decision_state,self.message_text)
                                        self.action_function = ""
                                        self.action_index = self.State.decision_state
                                        self.functions_executed = True
                                        self.functions_to_execute = self.commands_to_functions(functions, skip_states2, rm[0], robotState, info)
                                        self.plan = functions
                            
                                elif obeying and not obeying_present:
                                    objective_explanation = MessagePattern.plan_evaluation_bad() + "[" + final_function_descriptions + "]. " + objective_explanation
                                else:
                                    
                                    if not obeying:
                                        objective_explanation = MessagePattern.plan_evaluation_bad() + "[" + final_function_descriptions + "]. " + explanation + "Let's stick with the previous plan. "
                                    else:
                                        objective_explanation = MessagePattern.plan_evaluation_bad() + "[" + final_function_descriptions + "]. " + objective_explanation + "I cannot follow your orders. "
                                    
                                
                                
                            elif obeying and self.env.robot_id in functions.keys() and not (self.env.robot_id in self.plan and self.plan[self.env.robot_id] and self.plan[self.env.robot_id][0] == functions[self.env.robot_id][0]):
                                self.movement.cancel_cooperation(self.State.decision_state,self.message_text)
                                self.action_function = ""
                                self.action_index = self.State.decision_state
                                self.functions_executed = True
                                self.functions_to_execute = self.commands_to_functions(functions, skip_states2, rm[0], robotState, info)
                                self.plan = functions
                                
                                if not objective_explanation:
                                    objective_explanation = self.human_to_ai_text.noop
                            else:
                                if not objective_explanation:
                                    objective_explanation = self.human_to_ai_text.noop
                                
                        elif obeying and self.env.robot_id not in functions.keys():
                            print("Plan not targeted towards me")
                            objective_explanation = self.human_to_ai_text.noop
                        elif not obeying:
                            objective_explanation = "We are already doing the plan you are proposing! "
                           
                        translated_message = objective_explanation
                            
                        message_to_user = True 
                            
                self.env.sio.emit("text_processing", (False))
                    
                if translated_message:
                    if translated_message == self.human_to_ai_text.noop:
                        pass
                    elif message_to_user:
                        self.message_text.insert(translated_message,rm[0])
                    else:
                        if translated_messages_index == -1:
                            translated_messages_index = len(received_messages)
                        received_messages.append((rm[0], translated_message, rm[2])) #Add it to the list of received messages
                else:
                    print("understand here 2")
                    self.message_text.insert("I didn't understand you " + rm[0] + ". ",rm[0])
   
   
            if tmp_message:
                tmp_message_history.append({"Sender": rm[0], "Message": tmp_message, "Time": rm[2]})
        
            
            #Ignore any messages sent to itself
            #if rm[0] == str(self.env.robot_id):
            #    self.message_text = message_text_save
                
        if tmp_message_history:
            self.message_history.extend(tmp_message_history)
    
    
    def commands_to_functions(self, functions, keys_leaders, sender, robotState, info):
        command = functions[self.env.robot_id][0]
        
        function_to_execute = []
        
        self.order_status = self.OrderStatus.ongoing
        self.leader_id = sender
        
        if "carry" in command:
        
            object_id = command.split("_")[1]
            
            object_weight = 0
            if object_id in info['object_key_to_index'].keys():
                object_idx = info['object_key_to_index'][object_id]
                object_weight = robotState.get("objects", "weight", object_idx)
            
            teammates = []
            leader = ""
            for kl in keys_leaders:
                if "carry_" + object_id in kl[0]:
                    if kl[1] != self.env.robot_id:
                        teammates.append(kl[1])
                    else:
                        leader = kl[2]
                        if leader != self.env.robot_id:
                            teammates.append(kl[1])
            
            #assert object_weight >= 1
            
            if object_weight > 1 or (not object_weight and leader and leader != self.env.robot_id):
                if leader:
                    '''
                    teammates = []
                    for ag in functions.keys():
                        if functions[ag][0] == command and ag != leader:
                            teammates.append(ag)
                    '''
                    if leader == self.env.robot_id:
                        if len(teammates)+1 >= object_weight:
                            self.movement.help_status = self.movement.HelpState.being_helped
                            self.movement.help_status_info[2] = []
                            self.movement.help_status_info[0] = []
                            self.movement.help_status_info[6] = []
                            self.movement.help_status_info[1] = time.time()
                            
                            self.movement.help_status_info[0].extend(teammates)
                            
                            match_pattern,m_idx = self.message_text.search(MessagePattern.location_regex())
                                            
                            if match_pattern and not match_pattern.group(7):
                                self.message_text.message[m_idx] = self.message_text.message[m_idx].replace(match_pattern.group(), match_pattern.group() + " Helping " + self.env.robot_id + ". ")

                            self.message_text.insert(MessagePattern.order_response(sender, "collect") + "I will lead this operation. ",sender)
                            function_to_execute.append("collect_object(" + object_id + ")")
                            

                        else:
                            self.message_text.insert("Not enough agents are helping me, I'm asking for help. ",sender)
                            function_to_execute.append("ask_for_help_to_carry(\"" + object_id + "\")")
                            function_to_execute.append("collect_object(" + object_id + ")")
                    else:
                        self.movement.help_status = self.movement.HelpState.helping
                                        
                        self.movement.help_status_info[0] = [leader]
                        
                        teammates_sub = teammates.copy()
                        try:
                            teammates_sub.remove(self.env.robot_id)
                        except:
                            pdb.set_trace()
                        
                        self.movement.help_status_info[6].extend(teammates_sub)
                        
                        self.action_index = self.movement.State.follow
                        
                        print("TEAMMATE HERE 2", self.movement.help_status_info)
                    
                        self.message_text.insert(MessagePattern.order_response(sender, "collect") + "I will follow your lead, " + leader + ". ",sender)
                else:
                    function_to_execute.append("ask_for_help_to_carry(\"" + object_id + "\")")
                
                    function_to_execute.append("collect_object(" + object_id + ")")
            
            else:
                function_to_execute.append("collect_object(" + object_id + ")")
        elif "sense" in command:
            if "ROOM" in command:
                function_to_execute.append("sense_room(\"" + command.split("ROOM")[1] + "\")")
            else:
                function_to_execute.append("sense_object(" + command.split("_")[1] + ",[])")
                
                
        return function_to_execute
        
    def get_plan_description(self, plan):
        description = ""
        for p in plan:
        
            if description:
                description += ", "
                
            description += p[0] + " should "
            action = p[1][0][0]
            
            if "carry" in action:
                object_id = action.split("_")[1]
                description += "carry object " + object_id
                
            elif "sense" in action:
                description += "sense "
                if "ROOM" in action:
                    description += "room " + action.split("ROOM")[1]
                else:
                    description += "object " + action.split("_")[1]
                    
        return description
    
    def accept_help(self, robot_id, robotState, info):
        if self.movement.help_status == self.movement.HelpState.asking or self.movement.help_status == self.movement.HelpState.being_helped:

            num_agents_needed = robotState.get("objects", "weight", self.chosen_object_idx)
            
            rm = [robot_id,MessagePattern.carry_help_accept(self.env.robot_id)]
            
            return_value,_ = self.movement.message_processing_carry_help_accept(rm, {"weight": num_agents_needed}, self.message_text, self.helping_type == self.HelpType.sensing)
            
            
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
    
    def identify_room(self, robotState, info):
        for idx in range(len(self.other_agents)):
            other_robot_location = robotState.get("agents", "last_seen_location", idx)
            
            robot_id = self.env.map_config['all_robots'][idx][0]
            other_robot_time = robotState.get("agents", "last_seen_time", idx)
            
            robot_location_known = True
            
            if "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "order" and (other_robot_location[0] == -1 and other_robot_location[1] == -1 or info['time'] - other_robot_time >= 60):
                
                if info['time'] - self.last_update_message_sent[robot_id] >= 60:
                    self.message_text.insert(MessagePattern.updates(robot_id),robot_id)#just once every minute
                    self.last_update_message_sent[robot_id] = info['time']
                robot_location_known = False
            elif "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "obey" and (other_robot_location[0] == -1 and other_robot_location[1] == -1):
                robot_location_known = False
            else:
                room_num = self.env.get_room(other_robot_location,True,constrained=False).replace("room ", "")
            
            if robot_location_known and ((room_num and not self.other_agents[idx].rooms) or (self.other_agents[idx].rooms and room_num != self.other_agents[idx].rooms[-1])):
                self.other_agents[idx].rooms.append(room_num)
                robot_id = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(idx)]
                if self.planner and self.planner.path_monitoring[robot_id] and not self.planner.path_monitoring[robot_id].init_state:
                    if not self.planner.path_monitoring[robot_id].current_node:
                        self.planner.path_monitoring[robot_id].set_current_node(self.other_agents[idx].rooms[-2])
                    moving_result = self.planner.path_monitoring[robot_id].move_to(room_num)
                    if moving_result == 1:
                        self.message_text.insert(MessagePattern.not_following_order(robot_id),robot_id)
                        
                        if self.team_structure["hierarchy"][self.env.robot_id] == "order" and robot_id in self.plan and self.plan[robot_id]:
                            if "sense" in self.plan[robot_id][0]:
                                if "ROOM" in self.plan[robot_id][0]:
                                    self.message_text.insert("You should go towards room " + self.planner.path_monitoring[robot_id].goal + ". ",robot_id)
                                else:
                                    self.message_text.insert("You should go towards object " + self.plan[robot_id][0].split("_")[1] + ". ",robot_id)
                                    
                            elif "carry" in self.plan[robot_id][0]:
                                if "area" in self.planner.path_monitoring[robot_id].goal:
                                    self.message_text.insert("You should go towards the " + self.planner.path_monitoring[robot_id].goal + ". ",robot_id)
                                else:
                                    self.message_text.insert("You should go towards object " + self.plan[robot_id][0].split("_")[1] + ". ",robot_id)
                        
                        sanction_limit = 1
                        if self.planner.path_monitoring[robot_id].consecutive_increases - self.planner.path_monitoring[robot_id].consecutive_increases_limit >= sanction_limit:
                            try:
                                if robot_id in self.plan.keys():
                                    self.planner.path_monitoring[robot_id].update_reliability(self.planner.node_type[self.plan[robot_id][0]], 0)
                                elif robot_id in self.legacy_plan.keys():
                                    self.planner.path_monitoring[robot_id].update_reliability(self.planner.node_type[self.legacy_plan[robot_id][0]], 0)
                            except:
                                pdb.set_trace()
                            
                    elif moving_result == 2:
                        if robot_id in self.plan and self.plan[robot_id] and "carry" in self.plan[robot_id][0]:
                            self.planner.path_monitoring[robot_id].set('goal area', self.planner.occMap, self.planner.original_room_locations)
                            self.planner.path_monitoring[robot_id].moving_finished = True
                        
            if self.other_agents[idx].rooms:
                print("ROOM", idx, self.other_agents[idx].rooms[-1])

    def control(self,messages, robotState, info, next_observation):
        #print("Messages", messages)
        
        
        terminated = False
        message_order = ""
        
        self.occMap = np.copy(robotState.latest_map)
        
        ego_location = np.where(self.occMap == 5)
        
        self.modify_occMap(robotState, self.occMap, ego_location, info)
        
        self.nearby_other_agents, self.disabled_agents = self.get_neighboring_agents(robotState, ego_location)
        
        self.identify_room(robotState, info)
        
        if self.team_structure["hierarchy"][self.env.robot_id] == "obey":
            #if robotState.dropped_objects:
            #    pdb.set_trace()
            for od in robotState.dropped_objects:
                for agent_od in od:
                    self.planner.path_monitoring[agent_od[0]].update_reliability("carry_heavy", 0)
                    
            robotState.dropped_objects = []
        
        if not self.hierarchy_finished:
            self.hierarchy_finished = True
            available_robots = [r[0] for r in self.env.map_config['all_robots']]
            for tm in self.team_structure["hierarchy"].keys():
                if self.team_structure["hierarchy"][tm] == "obey" and tm in available_robots:
                    self.other_agents[info['robot_key_to_index'][tm]].finished = True
        
        
        messages.extend(self.self_messages)
        self.self_messages = []
        if messages: #Process received messages
            self.message_processing(messages, robotState, info)
            
        
        if self.movement.help_status == self.movement.HelpState.accepted and time.time() - self.movement.help_status_info[1] > self.help_time_limit2: #reset help state machine after a while if accepted a request
            self.movement.help_status = self.movement.HelpState.no_request
            self.movement.help_status_info[0] = []
        
        #self.message_text += MessagePattern.exchange_sensing_info(robotState, info, self.nearby_other_agents, self.other_agents, self.env.robot_id, self.env.convert_to_real_coordinates) #Exchange info about objects sensing measurements
        
        if not self.message_text.get():
        
            self.action_history.append(self.action_index)
            
            if len(self.action_history) > 2 and self.action_history[-1] == self.movement.State.wait_free and self.action_history[-2] != self.movement.State.wait_free and self.action_history[-3] == self.movement.State.wait_free and not robotState.object_held:
            
                if not self.movement.help_status == self.movement.HelpState.helping: #(self.movement.help_status == self.movement.HelpState.helping and self.other_agents[info["robot_key_to_index"][self.movement.help_status_info[0][0]]].carrying):
                
            
                    self.action_index = self.State.decision_state
                    self.action_function = ""
                    print("action_function -1", self.agent_requesting_order)
                    
                    print("Cancel help")
                    
                    if self.movement.help_status == self.movement.HelpState.being_helped:
                        _,_ = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_finish())
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
                            if self.team_structure["hierarchy"][self.env.robot_id] == "obey" or not self.decide_order:
                                function_str = self.decision_obey(messages, robotState, info, [], self.nearby_other_agents, next_observation)
                            #elif self.team_structure["hierarchy"][self.env.robot_id] == "order":
                            #    function_str = self.decision_order(messages, robotState, info, [], self.nearby_other_agents)
                            else:
                                function_str = self.decision(messages, robotState, info, [], self.nearby_other_agents, self.help_requests)
                                message_order = self.message_text.get_all()
                                #self.self_messages.append(("A","I think i should sense room 4", info['time'])) #DEBUG
                                if self.action_function:
                                    function_str = ''
                        else:
                            function_str = self.decision(messages, robotState, info, [], self.nearby_other_agents, self.help_requests)
                    
                    if function_str:
                    
                        print("Starting...")
                    
                        self.create_action_function(function_str)
                        
                        robotState.current_action_description = self.action_description(self.action_function)
                        
                        if external_function:
                            self.message_text.insert(robotState.current_action_description,"")
                        
                    else:
                        #self.action_function = self.create_action_function("wait")
                        print("action_function 0", self.agent_requesting_order)

                
                print(self.action_function)   
                #pdb.set_trace()                     
                #try:
                action, action_finished,function_output = eval(self.action_function)
                print("function output", function_output, action_finished)
                #except:
                #    pdb.set_trace()
                
                if message_order and ''.join(message_order[0]) not in self.message_text.get_str():
                    self.message_text.top_insert(message_order[0], message_order[1])
                    message_order = ""
                #except:
                #    pdb.set_trace()

                
                if action_finished:
                    self.action_sequence = 0
                    self.top_action_sequence = 0
                    
                    external_function = False
                    if self.functions_to_execute:
                        print("Functions to execute", self.functions_to_execute)
                        function_str = self.functions_to_execute.pop(0)
                        external_function = True
                        if function_output:
                            self.functions_to_execute_outputs.append(function_output)
                    else:
                        if "hierarchy" in self.team_structure:
                            if self.team_structure["hierarchy"][self.env.robot_id] == "obey" or not self.decide_order:
                                function_str = self.decision_obey(messages, robotState, info, function_output, self.nearby_other_agents, next_observation)
                                
                            #elif self.team_structure["hierarchy"][self.env.robot_id] == "order":
                            #    function_str = self.decision_order(messages, robotState, info, function_output, self.nearby_other_agents)
                            else:
                                robotState.set("agents", "team", robotState.get_num_robots(), "[]", info["time"])
                                function_str = self.decision(messages, robotState, info, function_output, self.nearby_other_agents, self.help_requests)
                            
                                if self.action_function:
                                    function_str = ''
                        else:
                            function_str = self.decision(messages, robotState, info, function_output, self.nearby_other_agents, self.help_requests)
                            
                    
                    if function_str:
                        self.create_action_function(function_str)
                        robotState.current_action_description = self.action_description(self.action_function)
                        if external_function:
                            self.message_text.insert(robotState.current_action_description,"")
                    else: #No function selected
                        self.action_function = ""
                        print("action_function 1", self.agent_requesting_order)
                        
                        
                if any(self.agent_requesting_order[r] for r in self.agent_requesting_order.keys()) and self.commander_order_status == self.OrderStatus.cancelling_order:
                    
                    """
                    for r in robotState.get_all_robots():
                        
                        #if r[0] != self.env.robot_id:
                        '''
                            if self.agent_requesting_order[r[0]]:
                                if r[0] in self.plan.keys() and self.plan[r[0]] and 'carry' in self.plan[r[0]][0]:
                                
                                    ob_id = self.plan[r[0]][0].split("_")[1]
                                    idx = info['object_key_to_index'][ob_id]
                                    weight = robotState.get("objects", "weight", idx)
                                    
                                    if weight > 1:
                                        carrying_agents = [s for s in self.plan.keys() if s in self.plan.keys() and self.plan[s] and self.plan[s][0] == self.plan[r[0]][0]]
                                    else:
                                        carrying_agents = [r[0]]
                                    
                                    for c in carrying_agents:
                                        print("Deleted", c, self.plan[c][0])
                                        del self.plan[c][0]
                                        
                                        if not self.plan[c]:
                                            del self.plan[c]
                                            del self.previous_plan[c]
                        '''             
                        self.agent_requesting_order[r[0]] = False
                    """
                    #self.agent_requesting_order = {r[0]:False for r in self.env.map_config['all_robots'] if r[0] != self.env.robot_id}
                    
                    function_str = self.decision(messages, robotState, info, function_output, self.nearby_other_agents, self.help_requests)
                    
                    
                    if function_str and not self.action_function:
                        self.create_action_function(function_str)
                        print("Created function")
                        self.action_sequence = 0
                        self.top_action_sequence = 0
                        
                elif self.commander_order_status == self.OrderStatus.giving_order:
                    self.decision(messages, robotState, info, function_output, self.nearby_other_agents, self.help_requests)
                    

            else:
            
                robotState.set("agents", "current_state", robotState.get_num_robots(), self.action_index.name, info["time"])
                
                action = self.sample_action_space
                action["action"] = -1
                action["num_cells_move"] = 1
            
                previous_action_index = self.action_index
                
                self.action_index,self.target_location,self.next_loc, low_action = self.movement.movement_state_machine(self.occMap, info, robotState, self.action_index, self.message_text, self.target_location,self.State.decision_state, self.next_loc, ego_location, -1)
                self.object_of_interest = ""
                
                if previous_action_index == self.movement.State.wait_message and not self.movement.help_status == self.movement.HelpState.asking: #self.movement.asked_help:
                    self.action_function = ""
                    robotState.set("agents", "team", robotState.get_num_robots(), "[]", info["time"])
                    print("action_function 2", self.agent_requesting_order)
                
                action["action"] = low_action
                
                if self.action_index == self.movement.State.follow or self.action_index == self.movement.State.obey:
                    robotState.current_action_description = self.action_description("follow(\"" + self.movement.help_status_info[0][0] + "\", robotState, next_observation, info)")
                    
                    
                if any(self.agent_requesting_order[r] for r in self.agent_requesting_order.keys()) and self.commander_order_status == self.OrderStatus.cancelling_order:
                    #for r in robotState.get_all_robots():
                    #    self.agent_requesting_order[r[0]] = False
                    
                    function_str = self.decision(messages, robotState, info, [], self.nearby_other_agents, self.help_requests)
                elif self.commander_order_status == self.OrderStatus.giving_order:
                    function_str = self.decision(messages, robotState, info, [], self.nearby_other_agents, self.help_requests)
                
                
                
            #print("Locationss", self.next_loc, self.target_location, ego_location)  
            if self.nearby_other_agents: #If there are nearby robots, announce next location and goal
                self.next_loc = self.movement.send_state_info(action["action"], self.next_loc, self.target_location, self.message_text, self.other_agents, self.nearby_other_agents, ego_location, robotState, self.object_of_interest, self.held_objects)  
                #pdb.set_trace()
               
                
            if self.message_text.get(): #Send message first before doing action
                
                rematch, _ = self.message_text.search(MessagePattern.location_regex())
                
                if rematch:
                    target_goal = eval(rematch.group(2))
                    target_loc = eval(rematch.group(3))
                    
                    if target_goal != target_loc and not (self.previous_message and self.previous_message[0] == target_goal and self.previous_message[1] == target_loc): #Only if there was a change of location do we prioritize this message

                        self.previous_message = [target_goal,target_loc]

                        
                        action,_,_ = self.send_message(self.message_text, robotState, next_observation, info)
                        print("SENDING MESSAGE", info['time'], self.message_text.get())
                        self.message_text.clear()

            
                    
                
        else:
        
            action,_,_ = self.send_message(self.message_text, robotState, next_observation, info)
            print("SENDING MESSAGE2", info['time'], self.message_text.get())
            self.message_text.clear()
            

        

        if action["action"] == -1 or action["action"] == "":
            
            action["action"] = Action.get_occupancy_map.value
            print("STUCK")

        if self.planner:
            for p in self.planner.path_monitoring.keys():
                print("reliability", p, self.planner.path_monitoring[p].reliability)

        print("action index:",self.action_index, "action:", Action(action["action"]), ego_location, self.action_function, self.movement.help_status, self.top_action_sequence, self.order_status, self.plan, self.commander_order_status, info['time'])
        
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
        self.action_index,_ = self.movement.message_processing_help(rm, self.action_index, self.helping_type == self.HelpType.sensing, self.State.decision_state, self.message_text)
        
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
                            self.message_text.insert(MessagePattern.object_not_found(agent_id, object_id),agent_id)
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
                self.message_text.insert(MessagePattern.item(robotState,ob_key,object_id, info, self.env.robot_id, self.env.convert_to_real_coordinates),agent_id)
            else:
                self.message_text.insert(MessagePattern.object_not_found(agent_id, object_id),agent_id)
                
            self.movement.help_status = self.movement.HelpState.no_request
            
            finished = True
        
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
                    self.message_text.insert("I don't know where object " + str(object_id) + " is. ",str(self.leader_id))
                    return action, True, []
        
            if (chosen_location[0] == -1 and chosen_location[1] == -1):# or self.occMap[chosen_location[0],chosen_location[1]] != 2: #if there is no object in the correct place
                print("object not found for sensing")

                finished = True
        
            action, temp_finished, _ = self.go_to_location(object_id, robotState, next_observation, info)
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
        
        '''
        if room not in ['1','2','3','4']:
            finished = True
            return action, finished, output 
        '''
        
        if self.top_action_sequence == 0:
        
            #if room == "0":
            #    pdb.set_trace()
        
            current_room = self.env.get_room([ego_location[0][0],ego_location[1][0]], True)
            #print(current_room, "room " + room, current_room == "room " + room)
            if current_room == "room " + room:
                '''
                room_coords = self.env.get_coords_room(self, robotState.latest_map, room, objects=True)
                cells = robotState.latest_map[room_coords[:,0],room_coords[:,1]]
                
                object_cells = np.argwhere(cells == 2)
                
                self.room_object_coords = room_coords[object_cells].squeeze().tolist()
                '''
                
                self.action_sequence = 0
                action, _, output = self.activate_sensor(robotState, next_observation, info)
                self.top_action_sequence = 3

                self.room_object_ids = []                
                '''
                self.top_action_sequence += 1
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
                '''
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
                print('hello000000000', output)
                while self.room_object_ids and robotState.get("object_estimates", "danger_status", [info['object_key_to_index'][self.room_object_ids[0]],robotState.get_num_robots()]):
                    print("Objects4", self.room_object_ids)
                    self.room_object_ids.pop(0)
                    
            
                if not self.room_object_ids:
                    finished = True
                    if not output:
                        
                        if self.team_structure["hierarchy"][self.env.robot_id] == "order":
                            self.self_messages.append(self.self_message_create(MessagePattern.order_sense_room_empty(room), robotState, info))
                        else:
                            self.message_text.insert(MessagePattern.order_sense_room_empty(room),str(self.leader_id))
            
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
                action["action"],self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_finish())
                self.object_of_interest = ""
                if tuple(chosen_location) in self.extended_goal_coords:
                    self.message_text.insert("The object is already in the goal area. ",str(self.leader_id))
            else:
            
                wait = False
                for agent_id in self.movement.help_status_info[0]:
                    agent_idx = info['robot_key_to_index'][agent_id]
                    other_robot_location = robotState.get("agents", "last_seen_location", agent_idx) #robotState.robots[agent_idx]["neighbor_location"]
                    if (other_robot_location[0] == -1 and other_robot_location[1] == -1):
                        wait = True
                    else:
                        real_distance = self.env.compute_real_distance([other_robot_location[0],other_robot_location[1]],[ego_location[0][0],ego_location[1][0]])
                
                        distance_limit = self.movement.distance_limit-1 #self.env.map_config['communication_distance_limit']-1
            
                        if real_distance >= distance_limit:
                            wait = True
                            if not robotState.get("agents", "type", agent_idx): #robotState.robots[agent_idx]["neighbor_type"]:
                                self.message_text.insert(MessagePattern.come_closer(agent_id),agent_id)
                if wait:          
                    if not ("hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "obey") and time.time() - self.movement.help_status_info[1] > self.help_time_limit2: #time.time() - self.movement.asked_time > self.help_time_limit2:                           
                        action["action"],self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_complain())
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
        
            wait_for_others,combinations_found = self.movement.wait_for_others_func(self.occMap, info, robotState, self.nearby_other_agents, [], ego_location,self.message_text)
        
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
                    while True:
                        g_coord = random.choice(self.env.reduced_goal_coords)
                        if not self.occMap[g_coord[0],g_coord[1]]:
                            self.target_location = g_coord
                            break
                        
                    '''
                    for g_coord in self.env.goal_coords:
                        if not self.occMap[g_coord[0],g_coord[1]]:
                            self.target_location = g_coord
                            break
                    '''
                    
                    
            elif not combinations_found: #No way of moving                          
                action["action"],self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_finish())
                finished = True
                self.object_of_interest = ""
                print("PROBLEM 2", combinations_found)
            elif not ("hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "obey") and time.time() - self.movement.help_status_info[1] > self.help_time_limit2: #time.time() - self.movement.asked_time > self.help_time_limit2:                           
                action["action"],self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_complain())
                finished = True
                self.object_of_interest = ""
                print("PROBLEM 3")
            else:
                action["action"] = Action.get_occupancy_map.value 
                print("PROBLEM 4")
                
        elif self.top_action_sequence == 2:

            self.action_index = self.State.drop_object
            
            if not robotState.object_held:            
                action["action"],self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_complain())
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
                    action["action"], self.next_loc, self.action_index = self.movement.go_to_location(self.target_location[0],self.target_location[1], self.occMap, robotState, info, ego_location, self.action_index, self.message_text, help_sensing=self.helping_type == self.HelpType.sensing)
                    
                    
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
                    wait_for_others,combinations_found = self.movement.wait_for_others_func(self.occMap, info, robotState, self.nearby_other_agents, self.previous_next_loc, ego_location,self.message_text)
                    
                    if not combinations_found: #No way of moving
                        self.top_action_sequence += 1
                        action["action"] = Action.get_occupancy_map.value
                        _,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_finish())
                
                
                        
                if loop_done or not wait_for_others: #If carrying heavy objects, wait for others
                    
                    action["action"], self.next_loc, self.action_index = self.movement.go_to_location(self.target_location[0],self.target_location[1], self.occMap, robotState, info, ego_location, self.action_index, self.message_text, help_sensing=self.helping_type == self.HelpType.sensing)
                    
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
                    _,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text,message=MessagePattern.carry_help_complain())
                    
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
                self.message_text.insert(MessagePattern.carry_help_finish(),str(self.leader_id))
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
            
            self.message_text.insert(message_to_send,"")
         
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
                    _,self.action_index = self.movement.cancel_cooperation(self.State.decision_state,self.message_text, message=MessagePattern.carry_help_finish())
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
                
            distance_limit = self.movement.distance_limit-1 #self.env.map_config['communication_distance_limit']-1
            
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
            
            self.message_text.insert(message_to_send,"")
         
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
            
            self.message_text.insert(message_to_send,"")
            
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
                #print(location_list)
                if location_list.size == 0:
                    action["action"] = Action.get_occupancy_map.value
                    output = [ego_location[0][0],ego_location[1][0]]
                    finished = True
                    return action,finished,output
                
                #print("ROOM " + object_id + " location list: ", location_list)
                path_distance = []
                for loc in location_list:
                    path = self.movement.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([loc[0],loc[1]]),robotState.latest_map)
                    if path:
                        path_distance.append(len(path))
                    else:
                        path_distance.append(float('inf'))
                        
                sorted_path_distance = np.argsort(path_distance)
                x,y = location_list[sorted_path_distance[0]]
                '''
                while True:
                    x,y = random.choice(location_list)
                    if self.movement.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),robotState.latest_map):
                        break
                '''
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
            
            low_action, self.next_loc, self.action_index = self.movement.go_to_location(x, y, self.occMap, robotState, info, ego_location, self.action_index, self.message_text, help_sensing=self.helping_type == self.HelpType.sensing)
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
                self.message_text.insert("Something is blocking the way, I cannot go " + place + ". ",str(self.leader_id))
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
            self.profiling['current_time'] = time.time()
            
        elif self.action_sequence == 1:
            self.profiling['sensing'].append(time.time()-self.profiling['current_time'])
            self.profiling['current_time'] = time.time()
            
            '''
            self.item_list = {}
            for object_key in info["last_sensed"].keys():
                if info["last_sensed"][object_key]["time"] >= self.env.last_time_danger_estimates_received:
                    self.item_list[object_key] = info["last_sensed"][object_key]
            '''
            
            self.item_list = info["last_sensed"]
            print("LAST SENSED:", info["last_sensed"] )
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
            self.profiling['check_item'].append(time.time()-self.profiling['current_time'])
            self.profiling['current_time'] = time.time()
            object_key = self.item_list.pop(0)
            action["action"] = Action.check_item.value    
            action["item"] = info["object_key_to_index"][object_key]
            
          
            if not self.item_list:
                self.action_sequence += 1
           
                
        elif self.action_sequence == 3:
            #[“object id”, “object x,y location”, “weight”, “benign or dangerous”, “confidence percentage”
            self.profiling['check_item'].append(time.time()-self.profiling['current_time'])
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
                
            #self.profiling['sensing'].append(time.time() - self.profiling['current_time'])
            
            print(self.profiling)
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
        
        """
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

        """
        
        agents = [self.env.neighbors_info[i][0] for i in range(len(self.env.neighbors_info))]
        ai_agents = [self.env.neighbors_info[i][0] for i in range(len(self.env.neighbors_info)) if self.env.neighbors_info[i][1]]
        messages = ["" for a in agents]
        ego_location = np.where(robotState.latest_map == 5)
        ego_location = np.array([ego_location[0][0], ego_location[1][0]])
        neighbors = self.env.get_neighboring_agents(0,ego_location)
        messages.append(message.get_str())
        
        all_messages = message.get_all()
        
        for m_idx in range(len(all_messages[0])):
            message_for_ai = False

            if isinstance(all_messages[1][m_idx],list):
                if all_messages[1][m_idx]:
                    for a in all_messages[1][m_idx]:
                        try:
                            messages[agents.index(a)] += all_messages[0][m_idx]
                        except:
                            pdb.set_trace()
     
            else:
                if all_messages[1][m_idx] == "ai":
                    for a in ai_agents:
                        messages[agents.index(a)] += all_messages[0][m_idx]
                    message_for_ai = True
                elif all_messages[1][m_idx] == "all":
                    for ag in range(len(messages)):
                        messages[ag] += all_messages[0][m_idx]
                elif not all_messages[1][m_idx]:
                    pass
                else:
                    messages[agents.index(all_messages[1][m_idx])] += all_messages[0][m_idx]
            
            if not message_for_ai:
                for a in neighbors.keys():
                    if all_messages[0][m_idx] not in messages[agents.index(a)]:
                        messages[agents.index(a)] += all_messages[0][m_idx]

        action["message"] = messages
        #action["message_ai"] = message_ai

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
                    print("carrying objects", self.held_objects)
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
                self.message_text.insert(MessagePattern.item(robotState,ob_key,object_id, info, self.env.robot_id, self.env.convert_to_real_coordinates),"all")
        
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
                        self.message_text.insert(MessagePattern.finish(),"all")
                        
                        self.report_heavy_dangerous_objects(robotState, info)
                                
                        
                        self.finished = True
                        self.told_to_finish = True
            else: #Only when all agents are next to each other finish
                self.told_to_finish = False
                
            finished = True
            
            if action["action"] < 0:
                action["action"] = Action.get_occupancy_map.value
        else:
            low_action, self.next_loc, self.action_index = self.movement.go_to_location(target_location[0], target_location[1], self.occMap, robotState, info, ego_location, self.action_index, self.message_text, help_sensing=self.helping_type == self.HelpType.sensing)

            
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
        self.action_index,_ = self.movement.message_processing_move_request(rm, robotState, info, self.action_index, self.message_text, self.other_agents, self.helping_type == self.HelpType.sensing)
        
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
            self.message_text.insert(MessagePattern.carry_help_accept(agent_id),agent_id)
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
            self.message_text.insert(message,"")
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
            #self.order_status = self.OrderStatus.reporting_availability
    
        if self.order_status == self.OrderStatus.ongoing:
            self.order_status = self.OrderStatus.reporting_output
            self.order_status_info = []
            if output or self.functions_to_execute_outputs:
            
                if not output:
                    output = []
            
                for sf in self.functions_to_execute_outputs:
                    output.extend(sf)
            
                self.order_status_info = [self.action_function, output]
                self.functions_to_execute_outputs = []
            

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
            if self.order_status_info and not self.team_structure["hierarchy"][self.env.robot_id] == "order":
                if "sense_object" in self.order_status_info[0] or "sense_room" in self.order_status_info[0]:
                
                    """
                    most_recent = 0
                    for ob in self.order_status_info[1]:
                        object_idx = info["object_key_to_index"][ob[0]]
                        this_time = robotState.items[object_idx]["item_time"][0]
                        if this_time > most_recent:
                            most_recent = this_time
                    """
                    
                    try:
                        print('info PROVIDED:', self.other_agents[info['robot_key_to_index'][str(self.leader_id)]].items_info_provided, self.order_status_info[1])
                    except:
                        pdb.set_trace()
                    objects_reported = []
                    for ob in self.order_status_info[1]:
                    
                    
                        if not isinstance(ob,list):
                            continue
                    
                        try:
                            object_idx = info["object_key_to_index"][ob[0]]
                        except:
                            pdb.set_trace()
                        #this_time = robotState.items[object_idx]["item_time"][0]
                        #if this_time >= most_recent:
                        if ob[0] in objects_reported:
                            continue
                        objects_reported.append(ob[0])
                        
                        if ob[0] not in self.other_agents[info['robot_key_to_index'][str(self.leader_id)]].items_info_provided:
                            self.message_text.insert(MessagePattern.item(robotState,object_idx,ob[0], info, self.env.robot_id, self.env.convert_to_real_coordinates),str(self.leader_id))
                            self.other_agents[info['robot_key_to_index'][str(self.leader_id)]].items_info_provided.append(ob[0])
                     
                    print('MESSAGE:',self.message_text.get())

                elif "collect_object" in self.order_status_info[0]:
                
                    for ob in self.order_status_info[1]:
                
                        try:
                            object_idx = info["object_key_to_index"][ob]
                        except:
                            pdb.set_trace()
                        self.message_text.insert(MessagePattern.item(robotState,object_idx,ob, info, self.env.robot_id, self.env.convert_to_real_coordinates),str(self.leader_id))
                elif "go_to_location" in self.order_status_info[0]:
                    self.message_text.insert(MessagePattern.surroundings(self.order_status_info[1], int(self.env.view_radius), robotState, info, self.env.convert_to_real_coordinates),str(self.leader_id))
            self.order_status = self.OrderStatus.reporting_availability
        elif self.order_status == self.OrderStatus.reporting_availability:
            if self.team_structure["hierarchy"][self.env.robot_id] == "order":
                self.self_messages.append(self.self_message_create(MessagePattern.order_finished(), robotState, info))
            else:
                self.message_text.insert(MessagePattern.order_finished(), str(self.leader_id)) #This should only be sent once
            self.order_status = self.OrderStatus.finished
            #self.leader_id = ""
            
    
        return function_output
    

    def update_planner(self, robotState, skip_states, objects_to_carry=[]):
        agents = []
        objects = []
        
        safe_location = self.env.convert_to_grid_coordinates(self.env.map_config['goal_radius'][0][1])

        locations = {'SAFE':tuple(safe_location)} #{'SAFE':(10,10)}
            
        object_weights = {}
        agents_initial_positions = {}
        pd_pb = {}
        being_carried = []
        mark_carried = []
        
        for ob in robotState.get_all_objects():
            key = str(ob[0])
            objects.append(key)
            
           
            ob_location = robotState.get("objects", "last_seen_location", ob[1]) #robotState.cursor.execute("""SELECT aoe.last_seen_location, MAX(aoe.last_seen_time) FROM agent_object_estimates aoe INNER JOIN objects o ON aoe.object_id = o.object_id WHERE o.idx = ?;""", (ob[1],)).fetchall()[0][0] #robotState.get("object_estimates", "last_seen_location", (ob[1] ,robotState.get_num_robots())) #robotState.get("objects", "last_seen_location", ob[1])
            
            locations[key] = ob_location
            object_weights[key] = ob[2]
            carried_by = robotState.get("objects", "carried_by", ob[1])
            if carried_by:
                being_carried.append(key)
                try:
                    if str(ob_location[0]) + '_' + str(ob_location[1]) not in robotState.map_metadata.keys() or (all(True if map_object[4] else False for map_object in robotState.map_metadata[str(ob_location[0]) + '_' + str(ob_location[1])] if not map_object[0]) and robotState.latest_map[ob_location[0],ob_location[1]] == 2):
                        mark_carried.append(ob_location)
                except:
                    pdb.set_trace()
            else:
                if not ob_location or (ob_location[0] == -1 and ob_location[1] == -1): #Error here when agents move object and then drop it but don't report their location
                    del object_weights[key]
                    del locations[key]
                    objects.remove(key)
                    
                    self.message_text.insert(MessagePattern.ask_for_object(key),"all")
                    
                        
                    
        agents_type = {}
        for ag in robotState.get_all_robots():
        
            #if ag[0] != self.env.robot_id:
        
            
            
            if ag[0] != self.env.robot_id:
                ag_location = robotState.get("agents", "last_seen_location", ag[1])
            else:
                ego_location = np.where(robotState.latest_map == 5) 
                ag_location = [ego_location[0][0],ego_location[1][0]]
                
            agents_initial_positions[ag[0]] = ag_location
            pb = robotState.get("agents", "sensor_benign", ag[1])
            pd = robotState.get("agents", "sensor_dangerous", ag[1])
            
            pd_pb[ag[0]] = (pd,pb)
            
            a_type = robotState.get("agents", "type", ag[1])
            
            if a_type:
                agents_type[ag[0]] = "ai"
            else:
                agents_type[ag[0]] = "human"
                
            
            if not ag_location or (ag_location[0] == -1 and ag_location[1] == -1):
                self.message_text.insert(MessagePattern.ask_for_agent(ag[0]),"all")
            else:
                agents.append(ag[0])
            

        occMap = np.copy(robotState.latest_map)
        
        for m in mark_carried:
            occMap[m[0],m[1]] = 4
        
        occMap[occMap > 2] = 0
        
        
        sql_estimates = robotState.get_all_sensing_estimates()
        estimates = []
    
        for e in sql_estimates:
            estimate_value = robotState.Danger_Status[e[2]].value

            if estimate_value:
                estimates.append((str(e[0]),e[1],estimate_value-1))

        if not self.planner:
            rooms = self.env.map_config['rooms']
            for r in rooms.keys():
                room_array = []
                for l in rooms[r]:
                    room_array.append(self.env.convert_to_grid_coordinates(l))
                self.new_rooms[r] = room_array
                self.all_rooms[r] = room_array
            self.planner = DynamicSensorPlanner(agents, objects, estimates, locations, agents_initial_positions, pd_pb, object_weights, occMap, self.extended_goal_coords, self.new_rooms, agents_type, objects_to_carry)
        else:
        
            #If an object is discovered in an empty room, re-add the room to new_rooms
            rooms = robotState.get_all_object_rooms()
            for r in rooms:
                if r in self.all_rooms.keys() and r not in self.new_rooms.keys():
                    self.new_rooms[r] = self.all_rooms[r]
                    
            print("BEING CARRIED", being_carried)
            self.planner.update_state(agents, objects, estimates, object_weights, agents_initial_positions, locations, occMap, skip_states, self.new_rooms, being_carried, objects_to_carry)
        
        
        
    def give_new_order(self, agent_id, robotState, info):
    
        if agent_id in self.plan and len(self.plan[agent_id]) > 1 and ('carry' in self.plan[agent_id][1] or 'sense' in self.plan[agent_id][1] or 'REGION' in self.plan[agent_id][1]):
    
            while self.plan[agent_id]: #eliminate steps that require carrying heavy objects
                del self.plan[agent_id][0]
                if self.plan[agent_id] and 'carry' in self.plan[agent_id][0]:
                    ob_id = self.plan[agent_id][0].split("_")[1]
                    idx = info['object_key_to_index'][ob_id]
                
                    weight = robotState.get("objects", "weight", idx)
                    if weight > 1:
                        continue
                        
                break
                
            
            if not self.plan[agent_id]:
                del self.plan[agent_id]
                del self.previous_plan[agent_id]
            else:
    
                if 'carry' in self.plan[agent_id][0]:
                    ob_id = self.plan[agent_id][0].split("_")[1]
                    idx = info['object_key_to_index'][ob_id]
                    
                    weight = robotState.get("objects", "weight", idx)
                    
                    if weight == 1:
                        self.message_text.insert(MessagePattern.item(robotState,idx,ob_id, info, self.env.robot_id, self.env.convert_to_real_coordinates),agent_id)
                        self.message_text.insert(MessagePattern.order_collect(agent_id, ob_id),agent_id)
                        robo_idx = info['robot_key_to_index'][agent_id]
                        robotState.set("agents", "team", int(robo_idx), str([int(robo_idx)]), info["time"])
                    else:
                        if ob_id not in self.carry_agents.keys():
                            self.carry_agents[ob_id] = []
                            if agent_id not in self.carry_agents[ob_id]:
                                self.carry_agents[ob_id].append(agent_id)
                    
                        if self.carry_agents[ob_id] == weight:
                            robot_id = self.carry_agents[ob_id][0]
                            robo_idx = info['robot_key_to_index'][robot_id]
                            other_robots_ids = self.carry_agents[ob_id][1:]
                        
                            self.message_text.insert(MessagePattern.agent(robot_id, int(robo_idx), robotState, self.env.convert_to_real_coordinates),other_robots_ids)
                            self.message_text.insert(MessagePattern.order_collect_group(robot_id, other_robots_ids, ob_id),[robot_id,*other_robots_ids])
                            
                            robo_idxs = [info['robot_key_to_index'][rid] for rid in self.carry_agents[ob_id]]
                            
                            for p in robo_idxs:
                                robotState.set("agents", "team", int(p), str([int(p) for p in robo_idxs]), info["time"])  
                        
                elif 'sense' in self.plan[agent_id][0]:
                
                    if 'CLUSTER' in self.plan[agent_id][0]:
                        cluster_num = self.plan[agent_id][0].split("CLUSTER")[1]
                        ob_locations = []
                        for ob_id in self.planner.clusters[int(cluster_num)]['objects']:
                            idx = info['object_key_to_index'][ob_id]
                            location = robotState.get("objects", "last_seen_location", idx)
                            if not location or (location[0] == -1 and location[1] == -1):
                                pdb.set_trace()
                            ob_locations.append(location) #robotState.get("object_estimates", "last_seen_location", (idx ,robotState.get_num_robots())))
                            
                        try:
                            
                            self.message_text.insert(MessagePattern.order_sense_multiple(agent_id, self.planner.clusters[int(cluster_num)]['objects'], ob_locations, self.env.convert_to_real_coordinates),agent_id)
                        except:
                            pdb.set_trace()
                            
                    else:
                
                        ob_id = self.plan[agent_id][0].split("_")[1]
                        idx = info['object_key_to_index'][ob_id]
                        location = robotState.get("objects", "last_seen_location", idx) #robotState.get("object_estimates", "last_seen_location", (idx ,robotState.get_num_robots()))
                        if not location or (location[0] == -1 and location[1] == -1):
                                pdb.set_trace()
                        self.message_text.insert(MessagePattern.order_sense(agent_id, ob_id, location, self.env.convert_to_real_coordinates),agent_id)
            
                elif 'REGION' in self.plan[agent_id][0]:
                    location = self.planner.locations[self.plan[agent_id][0]]
                    self.message_text.insert(MessagePattern.order_explore(agent_id, location, self.env.convert_to_real_coordinates),agent_id)
                
                
        elif agent_id in self.plan and len(self.plan[agent_id]) == 1:
            del self.plan[agent_id]
            del self.previous_plan[agent_id]
                
    
    def decision(self,messages, robotState, info, output, nearby_other_agents, help_requests):


        print("ORDER:", self.order_status, info["time"])

        if (self.commander_order_status == self.OrderStatus.cancelling_order and not self.providing_plan) or not self.plan:
        
            for idx in range(len(self.other_agents)):
                print(self.other_agents[idx].rooms)
        
        
            skip_states = []
            for sp in self.plan.keys():
                if 'carry' in self.plan[sp][0] and not self.agent_requesting_order[sp]:
                    skip_states.append([self.plan[sp][0],sp])
        
            self.update_planner(robotState,skip_states)
            if not self.plan:
                plan,self.optimization_metrics = self.planner.replan()
                for agent_plan in plan:
                    agent_id = agent_plan[0]
                    
                    orders = [p[0] for p in agent_plan[1] if 'SAFE' not in p[0]]
                    
                    #first_order = agent_plan[1][0][0]
                    self.plan[agent_id] = orders #first_order
            else:
                
                plan,self.optimization_metrics = self.planner.replan(skip_states)
                
                #FINISH
                if not plan:
                    self.message_text.insert(MessagePattern.finish(),"all")
                        
                    self.report_heavy_dangerous_objects(robotState, info)
                    
                    self.told_to_finish = True

        elif self.providing_plan:
            plan = self.providing_plan
            
        changed_carry = []
        self.providing_plan = []
        
        if self.commander_order_status == self.OrderStatus.cancelling_order:
        
            all_agents = [r[0] for r in robotState.get_all_robots()] # if r[0] != self.env.robot_id]
        
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
                       
                        if agent_id == self.env.robot_id:
                            self.self_messages.append(self.self_message_create(MessagePattern.order_cancel(agent_id), robotState, info))
                        else:
                            self.message_text.insert(MessagePattern.order_cancel(agent_id),agent_id)
                        
                        print("Cancelling order for agent:", agent_id, orders[0], self.plan[agent_id][0])
                    self.previous_plan[agent_id] = self.plan[agent_id]
                    
                self.plan[agent_id] = orders #first_order    
                    
            self.commander_order_status = self.OrderStatus.giving_order
            function_output = "wait()"
            
            for agent_id in all_agents:
                if agent_id in self.plan.keys():
                    self.legacy_plan[agent_id] = self.plan[agent_id]
                    del self.plan[agent_id]
                    del self.previous_plan[agent_id]
        
        elif self.commander_order_status == self.OrderStatus.giving_order:
        
            self.decide_order = False
        
            if self.told_to_finish:
                pdb.set_trace()
                self.finished = True
        
            all_agents = [r[0] for r in robotState.get_all_robots()] # if r[0] != self.env.robot_id]
        
            print("COMPARISON", self.plan, self.previous_plan, self.carry_agents)
        
            #for agent_plan in plan:
            for agent_id in self.plan.keys():
                #agent_id = agent_plan[0]
                #first_order = agent_plan[1][0][0]
                #if agent_id not in self.plan or first_order != self.plan[agent_id]:
                try:
                    all_agents.remove(agent_id)
                except:
                    pdb.set_trace()
                
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
                
                if agent_id not in self.previous_plan or 'sense' in self.plan[agent_id][0] or self.plan[agent_id][0] != self.previous_plan[agent_id][0] or ('carry' in self.plan[agent_id][0] and weight > 1):
                
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
                            if agent_id == self.env.robot_id:
                                self.self_messages.append(self.self_message_create(MessagePattern.order_collect(agent_id, ob_id), robotState, info))
                            else:
                                self.message_text.insert(MessagePattern.item(robotState,idx,ob_id, info, self.env.robot_id, self.env.convert_to_real_coordinates),agent_id)
                                self.message_text.insert(MessagePattern.order_collect(agent_id, ob_id),"all")
                                self.message_text.insert("I'm collecting object " + str(ob_id),"all")
                            robo_idx = info['robot_key_to_index'][agent_id]
                            robotState.set("agents", "team", int(robo_idx), str([int(robo_idx)]), info["time"])
                        
                            
                    elif 'sense' in self.plan[agent_id][0]:
                        
                        if 'CLUSTER' in self.plan[agent_id][0]:
                            cluster_num = self.plan[agent_id][0].split("CLUSTER")[1]
                            ob_locations = []
                            for ob_id in self.planner.clusters[int(cluster_num)]['objects']:
                                idx = info['object_key_to_index'][ob_id]
                                location = robotState.get("objects", "last_seen_location", idx)
                                ob_locations.append(location)
                                #ob_locations.append(robotState.get("object_estimates", "last_seen_location", (idx ,robotState.get_num_robots())))
                                
                                if not ob_locations[-1]:
                                    pdb.set_trace()
                            try:
                                if agent_id == self.env.robot_id:
                                    self.self_messages.append(self.self_message_create(MessagePattern.order_sense_multiple(agent_id, self.planner.clusters[int(cluster_num)]['objects'], ob_locations, self.env.convert_to_real_coordinates), robotState, info))
                                else:
                                    self.message_text.insert(MessagePattern.order_sense_multiple(agent_id, self.planner.clusters[int(cluster_num)]['objects'], ob_locations, self.env.convert_to_real_coordinates),"all")
                            except:
                                pdb.set_trace()
                        
                        elif 'ROOM' in self.plan[agent_id][0]:
                            #room = self.plan[agent_id][0].split("_")[1]   
                            #location = self.planner.locations[room]
                            room = self.plan[agent_id][0].split("ROOM")[1]   
                            if agent_id == self.env.robot_id:
                                #self.self_messages.append(self.self_message_create(MessagePattern.order_sense(agent_id, 99, location, self.env.convert_to_real_coordinates), robotState, info))
                                self.self_messages.append(self.self_message_create(MessagePattern.order_sense_room(agent_id, str(room)), robotState, info))
                            else:
                                #self.message_text += MessagePattern.order_sense(agent_id, 99, location, self.env.convert_to_real_coordinates)
                                self.message_text.insert(MessagePattern.order_sense_room(agent_id, str(room)),"all")
                                self.message_text.insert("I'm sensing room " + str(room),"all")
                        else:
                            #pdb.set_trace()
                            ob_id = self.plan[agent_id][0].split("_")[1]
                            idx = info['object_key_to_index'][ob_id]
                            location = robotState.get("objects", "last_seen_location", idx) #robotState.get("object_estimates", "last_seen_location", (idx ,robotState.get_num_robots()))
                            if agent_id == self.env.robot_id:
                                self.self_messages.append(self.self_message_create(MessagePattern.order_sense(agent_id, ob_id, location, self.env.convert_to_real_coordinates), robotState, info))
                            else:
                                self.message_text.insert(MessagePattern.order_sense(agent_id, ob_id, location, self.env.convert_to_real_coordinates),"all")
                                self.message_text.insert("I'm sensing object " + str(ob_id),"all")
                        
                    elif 'REGION' in self.plan[agent_id][0]:
                        location = self.planner.locations[self.plan[agent_id][0]]
                        if agent_id == self.env.robot_id:
                            self.self_messages.append(self.self_message_create(MessagePattern.order_explore(agent_id, location, self.env.convert_to_real_coordinates), robotState, info))
                        else:
                            self.message_text.insert(MessagePattern.order_explore(agent_id, location, self.env.convert_to_real_coordinates),"all")
                        
              
            
            for r in robotState.get_all_robots():
                #if r[0] != self.env.robot_id:
                '''
                    if r[0] in all_agents:
                        self.agent_requesting_order[r[0]] = True
                    else:
                        self.agent_requesting_order[r[0]] = False
                '''
                self.agent_requesting_order[r[0]] = False
                    
                        
            print(self.agent_requesting_order, changed_carry, self.carry_agents)
                       
            for ob in changed_carry:
            
                robot_id = self.carry_agents[ob][0]
                robo_idx = info['robot_key_to_index'][robot_id]
                other_robots_ids = self.carry_agents[ob][1:]
                
                ob_id = self.plan[robot_id][0].split("_")[1]
                idx = info['object_key_to_index'][ob_id]
            
                if not other_robots_ids:
                    pdb.set_trace()
            
                self.message_text.insert(MessagePattern.agent(robot_id, int(robo_idx), robotState, self.env.convert_to_real_coordinates),other_robots_ids)
                self.message_text.insert(MessagePattern.item(robotState,idx,ob_id, info, self.env.robot_id, self.env.convert_to_real_coordinates),[robot_id,*other_robots_ids])
                self.message_text.insert(MessagePattern.order_collect_group(robot_id, other_robots_ids, ob),"all")
                
                if robot_id == self.env.robot_id or self.env.robot_id in other_robots_ids:
                    self.self_messages.append(self.self_message_create(MessagePattern.order_collect_group(robot_id, other_robots_ids, ob), robotState, info))
                
                robo_idxs = [info['robot_key_to_index'][rid] for rid in self.carry_agents[ob]]
                
                for p in robo_idxs:
                    robotState.set("agents", "team", int(p), str([int(p) for p in robo_idxs]), info["time"])       
            

            function_output = "sleep()"
            self.commander_order_status = self.OrderStatus.cancelling_order
        
        return function_output
            
    
