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
from process_text import Human2AIText
from cnl import MessagePattern
from movement import Movement



class HeuristicControl:

    def __init__(self, goal_coords, num_steps, robot_id, env, role, planning):
        self.goal_coords = goal_coords.copy()
        
        self.extended_goal_coords = goal_coords.copy()
        

        self.extended_goal_coords.extend([(g[0]+op[0],g[1]+op[1]) for g in self.goal_coords for op in [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]] if [g[0]+op[0],g[1]+op[1]] not in self.goal_coords])
        

        self.memory_replay = ReplayMemory(10000)
        
        self.num_steps = num_steps
        
        self.robot_id = robot_id
        
        self.env = env
        

        self.wait_free_limit = 10
        self.help_time_limit2 = 30

        
        self.ending_locations = [[x,y] for x in range(8,13) for y in range(15,19)] #ending locations
        self.ending_locations.remove([12,18]) #To ensure all locations are within communication range
        self.ending_locations.remove([8,18])
        
        self.other_agents = [self.Other_Agent() for r in range(env.action_space["robot"].n-1)]
        
        
        self.original_role = role
        
        
        self.original_planning = planning
        
        self.room_distance = 7
        
        if not all(robot[1] for robot in env.neighbors_info): #Check if there are human peers    
            self.human_to_ai_text = Human2AIText(self.robot_id)
            
            
        self.movement = Movement(env)
        
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
        explore = 19
        
    class Other_Agent:
        
        def __init__(self):
            self.my_location = {"ego_location": [], "goal_location": [], "next_location": []} #Location of agent being controlled here
            self.my_carrying = False
            self.my_team = ""
            
            self.other_location = {"ego_location": [], "goal_location": [], "next_location": []} 
            self.team = ""
            self.carrying = False
            self.items = {}
            self.assignment = "None"
    
        
        
    
            
        
    def start(self):
    
        self.action_index = self.State.get_closest_object
        
        self.last_action = -1
        self.retries = 0
        self.not_dangerous_objects = []
        self.item_index = -1
        self.sensed_items = []
        self.message_text = ""
        self.heavy_objects = {"index": [], "weight": []}
        

        
        self.target_location = []
        self.chosen_heavy_object = -1
        self.message_send_time = float('inf')
        self.next_loc = []
        

        self.previous_message = []
        self.stuck_time = 0
        self.previous_next_loc = []
        self.stuck_too_much = 0
        self.too_stuck = 0
        self.target_object_idx = -1
        self.assigned_target_location = []
        self.just_started = True
        self.planning = self.original_planning
        self.role = self.original_role
        self.finished_robots = []
        
        #self.other_agents = {n:Other_Agent() for n in agents_ids}
        

    def go_to_location(self, x, y, occMap, robotState, info, ego_location, end=False,checking=False): #Wrapper function
    
        action, path_to_follow, self.message_text, self.action_index = self.movement.go_to_location(x, y, occMap, robotState, info, ego_location, self.action_index,  end=end,checking=checking)
    
 
        return action,path_to_follow
        
    def modify_occMap(self,robotState, occMap, ego_location, info):
    
        self.movement.modify_occMap(robotState, occMap, ego_location, info, self.next_loc)
        
        if self.action_index != self.State.drop_object and robotState.object_held:
            for agent_id in self.movement.being_helped: #if you are being helped, ignore locations of your teammates

                agent_idx = info['robot_key_to_index'][agent_id]
                other_robot_location = robotState.robots[agent_idx]["neighbor_location"]
                
                if occMap[other_robot_location[0],other_robot_location[1]] != 5:
                    occMap[other_robot_location[0],other_robot_location[1]] = 3
                    

    
        
    def drop(self):
        return Action.drop_object.value
        
    def pick_up(self, occMap, item_location, ego_location):
        
    
        action = self.movement.position_to_action([ego_location[0][0],ego_location[1][0]],item_location,True)
        
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
        
    
    
    
    
    def message_processing(self,received_messages, robotState, info):
    
        action = -1
        last_move_request = ""
    
        for rm in received_messages:
            
            print("Received message:", rm)
            template_match = False
            
            if MessagePattern.carry_help_accept(self.robot_id) in rm[1]:
            
                template_match = True
                
                if self.chosen_heavy_object < len(self.heavy_objects["weight"]): 
                    
                    return_value,self.message_text,_ = self.movement.message_processing_carry_help_accept(rm, {"weight": self.heavy_objects["weight"][self.chosen_heavy_object], "index": self.heavy_objects["index"][self.chosen_heavy_object]}, self.message_text)
                    
                    
                    if return_value == 1:
                        
                        self.target_location = robotState.items[self.heavy_objects["index"][self.chosen_heavy_object]]['item_location']
                        self.target_object_idx = self.heavy_objects["index"][self.chosen_heavy_object]
                        
                        self.action_index = self.State.move_and_pickup
                        
                else: #Something happened with the object
                    self.message_text += MessagePattern.carry_help_reject(rm[0])


            if re.search(MessagePattern.carry_help_regex(),rm[1]) and self.role != "scout":
            
                template_match = True
            
                self.message_text,self.action_index,_ = self.movement.message_processing_carry_help(rm, robotState, self.action_index, self.message_text)
                
                
            if re.search(MessagePattern.follow_regex(),rm[1]) or MessagePattern.carry_help_cancel() in rm[1] or MessagePattern.carry_help_reject(self.env.robot_id) in rm[1] or MessagePattern.carry_help_finish() in rm[1] or MessagePattern.carry_help_complain() in rm[1]:
            
                template_match = True
            
                self.action_index,_ = self.movement.message_processing_help(rm, self.action_index, self.State.get_closest_object)
            
            
            if re.search(MessagePattern.location_regex(),rm[1]) and not (self.movement.being_helped and rm[0] in self.movement.being_helped and self.action_index == self.State.drop_object):
            
                template_match = True
                
                self.message_text,self.action_index,_ = self.movement.message_processing_location(rm, robotState, info, self.other_agents, self.target_location, self.action_index, self.message_text, self.State.get_closest_object, self.next_loc)
                
            
            if MessagePattern.wait(self.env.robot_id) in rm[1] or re.search(MessagePattern.move_order_regex(),rm[1]):
                template_match = True
                self.target_location, self.action_index, _ = self.movement.message_processing_wait(rm, info, self.target_location, self.action_index)
                
            if MessagePattern.move_request(self.env.robot_id) in rm[1]:
                template_match = True
                
                if not self.planning == "coordinator":
                    self.message_text,self.action_index,_ = self.movement.message_processing_move_request(rm, robotState, info, self.action_index, self.message_text)
                
            if re.search(MessagePattern.sensing_help_regex(),rm[1]): #"What do you know about object " in rm[1]:
                rematch = re.search(MessagePattern.sensing_help_regex(),rm[1])
                
                template_match = True
                
                object_id = rematch.group(1) #rm[1].strip().split()[-1] 
                object_idx = info['object_key_to_index'][object_id]
                
                self.message_text += MessagePattern.item(robotState.items,object_idx,object_id, self.env.convert_to_real_coordinates)
                
                if not self.message_text:
                     self.message_text += MessagePattern.sensing_help_negative_response(object_id)
            if re.search(MessagePattern.item_regex_full(),rm[1]):
            
                template_match = True
            
                for rematch in re.finditer(MessagePattern.item_regex_full(),rm[1]):
                
                    MessagePattern.parse_sensing_message(rematch, rm, robotState, info, self.other_agents, self.env.convert_to_grid_coordinates)
                    
                    object_id = rematch.group(1)
        
                    object_idx = info['object_key_to_index'][object_id]
                    
                    item_weight = int(rematch.group(2))
                    
                    if robotState.items[object_idx]["item_danger_level"] == 1:
                    
                        if object_idx not in self.not_dangerous_objects:
                    
                            self.not_dangerous_objects.append(object_idx)
                            
                            
                        if item_weight > 1 and object_idx in self.heavy_objects["index"]:
                            oi = self.heavy_objects["index"].index(object_idx)
                            del self.heavy_objects["index"][oi]
                            del self.heavy_objects["weight"][oi]
                        
                    elif robotState.items[object_idx]["item_danger_level"] == 2:
                    
                        if object_idx in self.not_dangerous_objects:
                            self.not_dangerous_objects.remove(object_idx)
                            
                        if item_weight > 1 and object_idx not in self.heavy_objects["index"]:
                            self.heavy_objects["index"].append(object_idx)
                            self.heavy_objects["weight"].append(item_weight)
                        
                                              
                                        
                    
            if MessagePattern.explanation_question(self.robot_id) in rm[1]:
                template_match = True
                
                self.message_text += MessagePattern.explanation_response(self.action_index)
                
            
                    
                    
            if re.search(MessagePattern.order_sense_regex(),rm[1]):
            
                template_match = True
            
                for rematch in re.finditer(MessagePattern.order_sense_regex(),rm[1]):
                    if rematch.group(1) == self.robot_id:
                        self.assigned_target_location = self.env.convert_to_grid_coordinates(eval(rematch.group(2)))
                        self.role = "scout"
                        self.action_index = self.State.get_closest_object
                        
            if re.search(MessagePattern.order_collect_regex(),rm[1]):
            
                template_match = True
            
                for rematch in re.finditer(MessagePattern.order_collect_regex(),rm[1]):
                    if rematch.group(1) == self.robot_id:
                        object_idx = info['object_key_to_index'][rematch.group(2)]
                        self.target_location = robotState.items[object_idx]["item_location"] #Check assignment
                        self.target_object_idx = object_idx
                        self.role = "lifter"
                        self.action_index = self.State.move_and_pickup
                        self.movement.being_helped_locations = []
                        
            if re.search(MessagePattern.order_collect_group_regex(),rm[1]):
            
                template_match = True
            
                for rematch in re.finditer(MessagePattern.order_collect_group_regex(),rm[1]):
                    if rematch.group(1) == self.robot_id:
                        object_idx = info['object_key_to_index'][rematch.group(4)]
                        self.role = "lifter"
                        self.movement.asked_time = time.time()
                        self.movement.being_helped.append(rematch.group(2))
                        
                        if rematch.group(3):
                            self.movement.being_helped.extend(rematch.group(3).split(",")[1:])
                    
                        
                        self.target_location = robotState.items[object_idx]['item_location']
                        self.target_object_idx = object_idx
                        
                        self.action_index = self.State.move_and_pickup
                        self.movement.being_helped_locations = []
                        
                    
                        match_pattern = re.search(MessagePattern.location_regex(),self.message_text)
                        
                        if match_pattern and not match_pattern.group(6):
                            self.message_text = self.message_text.replace(match_pattern.group(), match_pattern.group() + " Helping " + self.robot_id + ". ")
                            
                    elif rematch.group(2) == self.robot_id or (rematch.group(3) and self.robot_id in rematch.group(3).split(",")):
                    

                        self.role = "lifter"
                        
                        self.movement.helping = [rematch.group(1),0]
                        
                        self.action_index = self.movement.State.follow
                        
                        

            if MessagePattern.order_finished() in rm[1] and self.planning == "coordinator":  
            
                template_match = True
            
                robot_idx = info['robot_key_to_index'][rm[0]]
                
                self.other_agents[robot_idx].assignment = ""
                
            if MessagePattern.task_finished() in rm[1] and self.planning == "coordinated":
                
                template_match = True
                
                self.role = "lifter"
                self.planning = "equal"
                
                
            if MessagePattern.finish() in rm[1]:
            
                template_match = True
            
                if rm[0] not in self.finished_robots:
                    self.finished_robots.append(rm[0])
                    
                    
            if MessagePattern.finish_reject() in rm[1]:
            
                template_match = True
            
                if rm[0] in self.finished_robots:
                    self.finished_robots.remove(rm[0])
                
                
                    
            if not template_match and not robotState.robots[info['robot_key_to_index'][rm[0]]]["neighbor_type"]: #Human sent a message, we need to translate it. We put this condition at the end so that humans can also send messages that conform to the templates
                translated_message = self.human_to_ai_text.convert_to_ai(rm[1], True)
                
                if translated_message:
                    if translated_message == "Nothing":
                        self.message_text += self.human_to_ai_text.free_response(rm[1], True) + ". "
                    else:
                        received_messages.append((rm[0], translated_message, rm[2])) #Add it to the list of received messages
                else:
                    self.message_text += "I didn't understand you " + rm[0] + ". "
                
                
        return action  
        
    def check_team_arrangement(self, item_location, occMap, weight, ego_location): #We need to know if it's even possible for a team to arrange itself in order to carry the object
        

        sub_occMap = occMap[item_location[0]-1:item_location[0]+2,item_location[1]-1:item_location[1]+2]
   
        
        free_locs = np.where((sub_occMap == 0) | (sub_occMap == 3))
        num_possibles = 0
        
        for fl_idx in range(len(free_locs[0])):
            
            new_fl = [free_locs[0][fl_idx] + item_location[0]-1, free_locs[1][fl_idx] + item_location[1]-1]
            
        
            possible_path = self.movement.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([new_fl[0],new_fl[1]]),occMap)
            
            if possible_path:
                num_possibles += 1
                
        clean_occMap = np.copy(occMap)
        
        for sg in self.extended_goal_coords:
            clean_occMap[sg[0],sg[1]] = 0
            
        clean_occMap[item_location[0],item_location[1]] = 0
                
        possible_path = self.movement.findPath(np.array([item_location[0],item_location[1]]),np.array([sg[0],sg[1]]),clean_occMap)
        
        free_locs_path = []
        
        for p in possible_path: #Check if there is space through all the trajectory
            sub_occMap = clean_occMap[p[0]-1:p[0]+2,p[1]-1:p[1]+2]
        
            free_locs = len(np.where((sub_occMap == 0) | (sub_occMap == 3))[0])
            
            free_locs_path.append(free_locs)
                
        return num_possibles >= weight and all(f >= weight for f in free_locs_path)

     
    
        
        
    def get_neighboring_agents(self, robotState, ego_location):
    
        nearby_other_agents = []
        #Get number of neighboring robots at communication range
        for n_idx in range(len(robotState.robots)):
            if "neighbor_location" in robotState.robots[n_idx] and self.env.compute_real_distance([robotState.robots[n_idx]["neighbor_location"][0],robotState.robots[n_idx]["neighbor_location"][1]],[ego_location[0][0],ego_location[1][0]]) < self.env.map_config['communication_distance_limit']:
                nearby_other_agents.append(n_idx)
                
        return nearby_other_agents

    def carry_heavy_object(self, robotState, ego_location, nearby_other_agents, info):
    
        num_neighbors = len(nearby_other_agents)
    
        for ho_idx in reversed(range(len(self.heavy_objects["index"]))): #Eliminate heavy objects if they have already been taken care of
            if tuple(robotState.items[self.heavy_objects["index"][ho_idx]]['item_location']) in self.extended_goal_coords:
                del self.heavy_objects["index"][ho_idx]
                del self.heavy_objects["weight"][ho_idx]
                            
        print("Asked help", self.movement.asked_help)  
        if self.role != "scout" and self.heavy_objects["index"] and not robotState.object_held and not self.movement.accepted_help and not self.movement.being_helped and not self.movement.asked_help and time.time() - self.movement.asked_time > self.movement.help_time_limit: #Ask for help to move heavy objects 
            
            
                    
            order_heavy_objects_ind = np.argsort(self.heavy_objects['weight'])[::-1] #Depening on the number of neighbors, ask for help for a specific object
                    
            unavailable_robots = sum(1 if self.other_agents[oa_idx].team or self.other_agents[oa_idx].carrying else 0 for oa_idx in range(len(self.other_agents)) if oa_idx in nearby_other_agents)
                    
            chosen_object = -1
            
            for ho in order_heavy_objects_ind: #Select the object closest to ego
            
                try:    
                    if self.heavy_objects["index"][ho] in self.sensed_items and self.heavy_objects['weight'][ho] <= num_neighbors+1 - unavailable_robots and self.check_team_arrangement(robotState.items[self.heavy_objects["index"][ho]]['item_location'],robotState.latest_map, self.heavy_objects['weight'][ho], ego_location): #Heavy object should have been sensed by robot, there should be enough nearby robots, and it should be feasible
                    
                        if chosen_object == -1 or (chosen_object > -1 and self.env.compute_real_distance(robotState.items[self.heavy_objects["index"][ho]]['item_location'],[ego_location[0][0],ego_location[1][0]]) < self.env.compute_real_distance(robotState.items[self.heavy_objects["index"][chosen_object]]['item_location'],[ego_location[0][0],ego_location[1][0]])):
                            chosen_object = ho

                except:
                    pdb.set_trace()    
            
            if chosen_object > -1:
                object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.heavy_objects['index'][chosen_object])]
                self.message_text += MessagePattern.carry_help(object_id,self.heavy_objects['weight'][chosen_object]-1)
                self.movement.asked_help = True
                self.movement.asked_time = time.time()
                if not self.action_index == self.movement.State.wait_free and not self.action_index == self.movement.State.wait_random:
                    self.movement.last_action_index = self.action_index
                self.action_index = self.movement.State.wait_message
                self.chosen_heavy_object = chosen_object
                print("ASKING HELP")        

    

            
    def central_planning(self, robotState, info, occMap, ego_location, nearby_other_agents):
        
        sensing_agents_per_object = 1 #How many robots to sense each room
        
        if self.action_index == self.State.init_move:
        
            
            action = self.return_to_meeting_point(occMap, robotState, info, ego_location)
            
        elif self.action_index == self.State.init_move_complete:
        
            action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
        
            neighbor_locations = np.where(occMap == 3)
        
            print([1 if not rob.assignment else 0 for rob in self.other_agents])
        
            #all_robots_in_place = all([1 if not rob.assignment else 0 for rob in self.other_agents])
            all_robots_in_place = all([1 if [neighbor_locations[0][rob_idx],neighbor_locations[1][rob_idx]] in self.ending_locations else 0 for rob_idx in range(len(neighbor_locations[0]))])
                
        
            if not action and isinstance(action, list):
            
           
                if all_robots_in_place:
                
                    for rob in self.other_agents:
                        rob.assignment = ""
                
                    self.action_index = self.State.sense_compute
                    self.assigned_item_locations = []
            
                action = Action.get_occupancy_map.value
                
                
    
        elif self.action_index == self.State.sense_compute: #Calculate all clusters of object locations
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
            
            self.action_index = self.State.sense_order
            action = Action.get_occupancy_map.value
            self.target_object_idx = 0
            
        elif self.action_index == self.State.sense_order:
        
            for rob in nearby_other_agents:
                if not self.other_agents[rob].assignment:
                
                    self.other_agents[rob].assignment = "-1" #Sensing
                    robot_index_to_key = list(info['robot_key_to_index'].keys())
                    robot_id = robot_index_to_key[list(info['robot_key_to_index'].values()).index(rob)]
                    
                    self.message_text += MessagePattern.order_sense(robot_id, self.assigned_item_locations[self.target_object_idx], self.env.convert_to_real_coordinates)
                
                    self.target_object_idx += 1
                    
                    if self.target_object_idx == len(self.assigned_item_locations):
                        break
            
                        
            if self.target_object_idx == len(self.assigned_item_locations):
                self.action_index = self.State.collect_order
                self.target_object_idx = 0
                
            action = Action.get_occupancy_map.value
                
        elif self.action_index == self.State.collect_order:
        
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
                            self.message_text += MessagePattern.order_collect(robot_id,object_id)
                            
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
                                
                                
                            self.message_text += MessagePattern.order_collect_group(robot_id, other_robot_ids,object_id)
                            
                            del missing_objects[m_idx]
                            
                            break
                                    
             
            print([rob2.assignment for rob2 in self.other_agents])
            if all(1 if not rob2.assignment else 0 for rob2 in self.other_agents):
                self.action_index = self.State.end_meeting
                self.planning = "equal" #Ends role as coordinator
                self.role = "lifter"
                
            action = Action.get_occupancy_map.value                    
                                
        
        
        elif self.action_index == self.State.wait_random:
                

            other_robot_location = robotState.robots[self.wait_requester]["neighbor_location"]
            
            if self.env.compute_real_distance(other_robot_location,[ego_location[0][0],ego_location[1][0]]) >= self.env.map_config['communication_distance_limit'] or time.time() - self.movement.asked_time > self.movement.help_time_limit: #Until the other robot is out of range we can move
                self.action_index = self.movement.last_action_index
            
            if self.movement.pending_location and self.movement.pending_location != [ego_location[0][0],ego_location[1][0]]:
                action = self.movement.position_to_action([ego_location[0][0],ego_location[1][0]],self.movement.pending_location,False)
            else:
                action = Action.get_occupancy_map.value    
                self.movement.pending_location = []
            
            
        elif self.action_index == self.State.wait_free: 
                    
            for loc_wait_idx in reversed(range(len(self.movement.wait_locations))): #Wait for robots to move from location
                loc_wait = self.movement.wait_locations[loc_wait_idx]
                if occMap[loc_wait[0],loc_wait[1]] == 0:
                    del self.movement.wait_locations[loc_wait_idx]
            print(time.time() - self.movement.asked_time)
            if not self.movement.wait_locations or time.time() - self.movement.asked_time > self.movement.wait_time_limit:
                self.action_index = self.movement.last_action_index
                self.movement.wait_locations = []
                print("Last action", self.movement.last_action_index)
            else:
                action = Action.get_occupancy_map.value
                
                                   
                                
        return action            
        
    
    def return_to_meeting_point(self, occMap, robotState, info, ego_location):
    
    
        still_to_explore = np.where(occMap == -2)
        
        if still_to_explore[0].size > 0:
        
            closest_dist = float('inf')
            closest_idx = -1
        

            for se_idx in range(len(still_to_explore[0])):
                unknown_loc = [still_to_explore[0][se_idx],still_to_explore[1][se_idx]]
                
                unknown_dist = self.env.compute_real_distance(unknown_loc,[ego_location[0][0],ego_location[1][0]])
                
                if unknown_dist < closest_dist:
                    closest_dist = unknown_dist
                    closest_idx = se_idx
                    
            self.target_location = [still_to_explore[0][closest_idx],still_to_explore[1][closest_idx]]
            
            print("Point chosen is ", self.target_location)
                          
            self.action_index = self.State.explore                           
            
            action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)                
           
        else:
            self.action_index = self.State.end_meeting
                                
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
        
        ego_location = np.where(occMap == 5)
        

        self.modify_occMap(robotState, occMap, ego_location, info)

        
        
        
        #Get number of neighboring robots at communication range
        #for nl_idx in range(len(neighbors_location[0])):

        #    if self.env.compute_real_distance([neighbors_location[0][nl_idx],neighbors_location[1][nl_idx]],[ego_location[0][0],ego_location[1][0]]) < self.env.map_config['communication_distance_limit']:
            
            
        #        num_neighbors += 1
                
                
        
        nearby_other_agents = self.get_neighboring_agents(robotState, ego_location)
        
        if not self.movement.helping and not self.planning == "coordinator" and not self.planning == "coordinated": #not helping another robot

            self.carry_heavy_object(robotState, ego_location, nearby_other_agents, info)
                        


        
        if received_messages: #Process received messages
            self.pending_action = self.message_processing(received_messages, robotState, info)
            
            
            #for rm in received_messages:
            #    if MessagePattern.move_request(self.robot_id) in rm[1]:
            #        pdb.set_trace() 
            #        break
                                

        self.message_text += MessagePattern.exchange_sensing_info(robotState, info, nearby_other_agents, self.other_agents, self.env.convert_to_real_coordinates) #Exchange info about objects sensing measurements
            
            
        if not self.message_text: #if going to send message, skip normal execution of actions
        
        
            if self.planning != "coordinator":
            
                if self.action_index == self.State.get_closest_object:
                    print("New sequence")
                    item_locations = np.where(occMap == 2)
                    
                    
                    
                    
                    min_possible_path = float('inf')
                    item_location_idx = -1
                    min_path = []
                    
                    
                    
                    
                    if self.role == "general":
                        heavy_objects_location = [tuple(robotState.items[idx]['item_location']) for idx in self.heavy_objects["index"] if idx in self.sensed_items] #We only exclude objects if they were sensed by this robot
                        non_dangerous_objects_location = [tuple(robotState.items[idx]['item_location']) for idx in self.not_dangerous_objects if idx in self.sensed_items]
                        objects_to_ignore = [self.extended_goal_coords, self.movement.ignore_object, non_dangerous_objects_location, heavy_objects_location]
                    elif self.role == "scout":
                        sensed_objects_location = [tuple(robotState.items[idx]['item_location']) for idx in self.sensed_items]
                        objects_to_ignore = [self.extended_goal_coords, self.movement.ignore_object, sensed_objects_location]
                    elif self.role == "lifter":
                        non_sensed_objects = [tuple(robotState.items[idx]['item_location']) for idx in range(len(robotState.items)) if robotState.items[idx]["item_danger_level"] == 0]
                        non_dangerous_objects_location = [tuple(robotState.items[idx]['item_location']) for idx in self.not_dangerous_objects]
                        heavy_objects_location = [tuple(robotState.items[idx]['item_location']) for idx in self.heavy_objects["index"]]
                        objects_to_ignore = [self.extended_goal_coords, self.movement.ignore_object, non_dangerous_objects_location, heavy_objects_location, non_sensed_objects]
                    
                    
                    if not self.planning == "coordinated" or (self.planning == "coordinated" and self.role != "lifter" and not self.assigned_target_location and not self.just_started):
                    
                        for it_idx in range(len(item_locations[0])): #Check which object in the map to go to
                            loc = (item_locations[0][it_idx],item_locations[1][it_idx])
                            
                            if any(1 if loc in group else 0 for group in objects_to_ignore): 
                                continue
                                
                                
                            try:
                                _,possible_path = self.go_to_location(loc[0],loc[1],occMap,robotState,info,ego_location,checking=True)
                                #possible_path = self.movement.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([loc[0],loc[1]]),occMap,all_movements=(not robotState.object_held))
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
                                    self.action_index = self.State.move_and_pickup
                                    self.target_object_idx = obj_idx
                                    already_scanned = True
                                    
                                    print("Scanned", robotState.items[obj_idx],heavy_objects_location, self.heavy_objects)
      
                                    
                                    break


                        if not already_scanned:
                            if not action and isinstance(action, list): #If already near target, start sensing
                                action = Action.danger_sensing.value
                                print("Object:", self.target_location, robotState.items)
                                self.target_location = []
                                self.action_index = self.State.init_check_items 
                            elif self.role == "scout" and (previous_target_location and np.linalg.norm(np.array(previous_target_location) - np.array(self.target_location)) > self.room_distance): #Whenever it finishes sensing a room return to meeting location. CHECK THIS!!!
                                action = self.return_to_meeting_point(occMap, robotState, info, ego_location)
                            else:
                                self.action_index = self.State.sense_area
                        else:
                            if not action and isinstance(action, list) : #If already near target

                                wait_for_others,_,self.message_text = self.movement.wait_for_others_func(occMap, info, robotState, nearby_other_agents, [], ego_location, self.message_text)
                                self.movement.being_helped_locations = []
                                if not wait_for_others:
                                    action = self.pick_up(occMap, self.target_location, ego_location)
                                    self.action_index = self.State.pickup_and_move_to_goal
                                    
                                    if action < 0:
                                        action = Action.get_occupancy_map.value
          
                                else:
                                    action = Action.get_occupancy_map.value
                                    
                                self.movement.asked_time = time.time()
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
                                elif obj["item_weight"] == 1 and obj_idx not in self.not_dangerous_objects and self.role != "scout" and not (obj_idx in self.movement.ignore_object and self.role == "lifter"):
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
                           
                elif self.action_index == self.State.sense_area:
                
                    
                    action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
                    

                    if self.movement.stuck_moving > 10:
                        self.movement.stuck_moving = 0
                        self.action_index = self.State.get_closest_object
                        self.movement.ignore_object.append(self.target_location)
                        print("Getting stuck moving!", self.movement.ignore_object)
                    else:
                        self.movement.ignore_object = []
                
                    if occMap[self.target_location[0],self.target_location[1]] == 0: #The package was taken or something happened
                        self.action_index = self.State.get_closest_object

                 
                    if not action and isinstance(action, list):
                        action = Action.danger_sensing.value
                        self.target_location = []
                        self.action_index = self.State.init_check_items

                       
                        
                elif self.action_index == self.State.init_check_items:
                    self.movement.ignore_object = []
                    self.item_index = 0
                    self.target_location = []
                    action,item = self.process_sensor(robotState, next_observation)
                    if action < 0: #No new sensing measurements

                        self.action_index = self.State.get_closest_object
                        action = Action.get_occupancy_map.value
                    else:
                        self.action_index = self.State.check_items

                elif self.action_index == self.State.check_items:
                
                    
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
                            
                            #self.message_text += MessagePattern.item(robotState.items,self.item_index-1,object_id, self.env.convert_to_real_coordinates)
         

                
                    action,item = self.process_sensor(robotState, next_observation)
                    if action < 0: #finished processing sensor measurements
                    
                        try:
                            if not self.target_location or self.role == "scout": #in case there is no object sensed
                                self.action_index = self.State.get_closest_object
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
                                self.action_index = self.State.move_and_pickup
                                
                                self.message_text += MessagePattern.sensing_help(str(list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.target_object_idx)]))
                                
                                if not action and isinstance(action, list): #If already next to object, try to pick it up
                                
                                    wait_for_others,_,self.message_text = self.movement.wait_for_others_func(occMap, info, robotState, nearby_other_agents,[], ego_location,self.message_text)
                                    self.movement.being_helped_locations = []
                                    
                                    if not wait_for_others:
                                        action = self.pick_up(occMap, self.target_location, ego_location)
                                        #ego_location = np.where(occMap == 5)
                                        if action < 0:
                                            action = self.movement.position_to_action([ego_location[0][0],ego_location[1][0]],self.past_location,False) 
                                        #self.past_location = [ego_location[0][0],ego_location[1][0]]
                                        self.action_index = self.State.pickup_and_move_to_goal
                                        self.retries = 0
                                    else:
                                       action = Action.get_occupancy_map.value 
                        except:
                            pdb.set_trace() 
                    else: 
                        self.past_location = [ego_location[0][0],ego_location[1][0]]
                        
                        
                        
                elif self.action_index == self.State.move_and_pickup: #From here on, only lifter behavior
                    self.movement.ignore_object = []
                    action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
                    print(self.target_location)
                    
                    if occMap[self.target_location[0],self.target_location[1]] == 0 or robotState.items[self.target_object_idx]["item_danger_level"] == 1: #The package was taken or something happened
                        print("Something happened at move_and_pickup")
                        message = ""
                        if self.movement.being_helped:
                            message = MessagePattern.carry_help_finish()
                        _,self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.get_closest_object,self.message_text,message=message)
                 
                    if not action and isinstance(action, list):
                    
                        print("waiting for others!")                    
                        
                            
                        wait_for_others,_,self.message_text = self.movement.wait_for_others_func(occMap, info, robotState, nearby_other_agents, [], ego_location,self.message_text)
                        
                        if not wait_for_others and not robotState.object_held: #pickup if next to object already
                            action = self.pick_up(occMap, self.target_location, ego_location)
                            if action < 0:
                                action = self.movement.position_to_action([ego_location[0][0],ego_location[1][0]],self.past_location,False) 
                            
                        else:
                            action = Action.get_occupancy_map.value 
                            self.past_location = [ego_location[0][0],ego_location[1][0]]
                            
                        #self.past_location = [ego_location[0][0],ego_location[1][0]]
                        self.action_index = self.State.pickup_and_move_to_goal
                        self.retries = 0
                        self.movement.asked_time = time.time()
                        self.movement.being_helped_locations = []

                    else:
                        self.past_location = [ego_location[0][0],ego_location[1][0]]
                        
                    
                elif self.action_index == self.State.pickup_and_move_to_goal:
                    self.movement.ignore_object = []

                    if robotState.object_held:
                        g_coord = []
                        for g_coord in self.goal_coords:
                            if not occMap[g_coord[0],g_coord[1]]:
                                target = g_coord
                                break
                        
                        self.target_location = g_coord
                        
                        action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
                        self.action_index = self.State.drop_object
                        
                        self.movement.being_helped_locations = []
                        self.previous_next_loc = []
                        #self.movement.wait_for_others_func(occMap, info, robotState, self.next_loc)

                        self.movement.asked_time = time.time()
                        if not action and isinstance(action, list):
                            #pdb.set_trace()
                            action = self.drop()
                            self.target_location = self.past_location
                            self.action_index = self.State.move_end
                        else:
                            action = Action.get_occupancy_map.value #Wait for the next state in order to start moving
                    else:
                        
                        #ego_location = np.where(occMap == 5)
                        action = self.movement.position_to_action([ego_location[0][0],ego_location[1][0]],self.past_location,False)
                          
                        if action == -1:
                        
                            wait_for_others,combinations_found,self.message_text = self.movement.wait_for_others_func(occMap, info, robotState, nearby_other_agents, [], ego_location,self.message_text)
                            
                            print("Action move and pickup:", action, wait_for_others, self.movement.being_helped_locations) 
                            
                            if not wait_for_others:
                                
                                action = self.pick_up(occMap, self.target_location, ego_location)
                                
                                if self.retries == 3: #If can't pickup object just try with another

                                    self.movement.ignore_object.append(tuple(self.target_location))
                                    
                                    message = ""
                                    if self.movement.being_helped:
                                        message = MessagePattern.carry_help_finish()
                                    
                                    
                                    action,self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.get_closest_object,self.message_text,message=message)
                                    
                                self.retries += 1   
                                self.movement.asked_time = time.time()
                                
                                print("Pickup retries:", self.retries)
                            elif not combinations_found: #No way of moving                          
                                action,self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.get_closest_object,self.message_text,message=MessagePattern.carry_help_finish())
                            elif time.time() - self.movement.asked_time > self.help_time_limit2:                           
                                action,self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.get_closest_object,self.message_text,message=MessagePattern.carry_help_complain())
                            else:
                                action = Action.get_occupancy_map.value 
                                
                            
                        

                elif self.action_index == self.State.drop_object:
                                
                    if not robotState.object_held:            

                        action,self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.get_closest_object,self.message_text,message=MessagePattern.carry_help_complain())
                        
                    else:
                    
                        for agent_id in self.movement.being_helped: #remove locations with teammates

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
                                self.movement.being_helped_locations = []
                            
                                print("PEFIOUVS",self.movement.being_helped_locations, self.next_loc, self.previous_next_loc)
                                
                            
                            
                            
                        
                        if not loop_done:
                            wait_for_others,combinations_found,self.message_text = self.movement.wait_for_others_func(occMap, info, robotState, nearby_other_agents, self.previous_next_loc, ego_location,self.message_text)
                            
                            if not combinations_found: #No way of moving
                                action = self.drop()
                                
                                _,self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.get_closest_object,self.message_text,message=MessagePattern.carry_help_finish())
                        
                        
                                
                        if loop_done or not wait_for_others: #If carrying heavy objects, wait for others
                            
                            action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
                            
                            if self.next_loc and self.previous_next_loc and not self.previous_next_loc[0].tolist() == self.next_loc[0].tolist(): #location changed
                                self.previous_next_loc = []
                                
                            if occMap[self.target_location[0],self.target_location[1]] == 2: #A package is now there
                                self.action_index = self.State.pickup_and_move_to_goal
                                self.movement.being_helped_locations = []

                        
                            if not action and isinstance(action, list): #If already next to drop location
                                action = self.drop()
                                self.target_location = self.past_location
                                self.action_index = self.State.move_end
                            else:
                                self.past_location = [ego_location[0][0],ego_location[1][0]]
                                
                            self.movement.asked_time = time.time()
                        elif time.time() - self.movement.asked_time > self.help_time_limit2:
                            action = self.drop()
                            
                            _,self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.get_closest_object,self.message_text,message=MessagePattern.carry_help_complain())
                        elif action != Action.drop_object.value:
                            action = Action.get_occupancy_map.value
                            print("waiting for others...")
                            
                elif self.action_index == self.State.move_end:
                    action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
                
                    if not action and isinstance(action, list):
                        action = Action.get_occupancy_map.value
                        
                        self.action_index = self.State.get_closest_object
                        
                        if self.movement.being_helped:
                            self.message_text += MessagePattern.carry_help_finish()
                            self.movement.asked_time = time.time()
                        self.movement.being_helped = []
                        self.movement.being_helped_locations = []
                        

                elif self.action_index == self.State.end_meeting:
                
                    action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)
            
            
                    if [ego_location[0][0],ego_location[1][0]] in self.ending_locations:
                        self.just_started = False
            
                    if not action and isinstance(action, list):
                        
                        if self.role == "scout": #Scout should share information
                            
                            """
                            object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.item_index)]
                            
                            missing_objects = 0
                            while robotState.items[self.item_index]["item_danger_level"] == 0 and self.item_index < len(robotState.items):
                                self.message_text += MessagePattern.item(robotState.items,self.item_index,object_id, self.env.convert_to_real_coordinates)
                                
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
                                
                                self.action_index = self.State.get_closest_object
                            else:
                                self.message_text += MessagePattern.order_finished()
                                self.action_index = self.State.waiting_order
                                
                        elif self.role == "lifter": #If there are objects one can lift
                            if self.planning != "coordinated":
                                
                                for idx in range(len(robotState.items)):
                                    if robotState.items[idx]["item_danger_level"] == 2 and robotState.items[idx]["item_weight"] == 1:
                                        self.action_index = self.State.get_closest_object

                                if self.action_index == self.State.end_meeting and self.robot_id not in self.finished_robots: #Voluntarily finish
                                    self.message_text += MessagePattern.finish()
                                    self.finished_robots.append(self.robot_id)
                            else:
                                self.message_text += MessagePattern.order_finished()
                                self.action_index = self.State.waiting_order
                                
                        elif self.planning == "coordinated":
                            self.message_text += MessagePattern.order_finished()
                            self.action_index = self.State.waiting_order
                            
                            
                            

                    
                        action = Action.get_occupancy_map.value
                    print("Finished")
                    
                elif self.action_index == self.State.waiting_order:
                    action = Action.get_occupancy_map.value
                
                
                
                elif self.action_index == self.State.explore: #Explore the area
                
                    action,self.next_loc = self.go_to_location(self.target_location[0],self.target_location[1],occMap,robotState,info,ego_location)  
                    
                    if not action and isinstance(action, list):  
                        action = Action.get_occupancy_map.value
                        self.action_index = self.State.get_closest_object
                        
                    elif action == -1:
                        action = Action.get_occupancy_map.value
                        self.action_index = self.State.get_closest_object
                        
                        
                else:
                    self.message_text,self.action_index,self.target_location,self.next_loc, action = self.movement.movement_state_machine(occMap, info, robotState, self.action_index, self.message_text, self.target_location,self.State.get_closest_object, self.next_loc, ego_location, action)
                        
            else:
                action = self.central_planning(robotState, info, occMap, ego_location, nearby_other_agents)
                    
            if nearby_other_agents: #If there are nearby robots, announce next location and goal
            
            
                self.message_text, self.next_loc = self.movement.send_state_info(action, self.next_loc, self.target_location, self.message_text, self.other_agents, nearby_other_agents, ego_location, robotState)
                
            
            
            if self.message_text: #Send message first before doing action
                

                if re.search(MessagePattern.location_regex(),self.message_text):
                    self.message_send_time = info['time']
                    rematch = re.search(MessagePattern.location_regex(),self.message_text)
                    target_goal = eval(rematch.group(1))
                    target_loc = eval(rematch.group(2))
                    
                    #pdb.set_trace()
                    if target_goal != target_loc and not (self.previous_message and self.previous_message[0] == target_goal and self.previous_message[1] == target_loc): #Only if there was a change of location do we prioritize this message

                        self.previous_message = [target_goal,target_loc]

                        
                        action,message = self.send_message(self.message_text)
                        self.message_text = ""

                        print("SENDING MESSAGE", info['time'], message)
                        

                
            
        else:
        
            
        
            if re.search(MessagePattern.location_regex(),self.message_text):
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
        
        
        if self.action_index != self.State.end_meeting and self.robot_id in self.finished_robots: #Voluntarily finish
            self.message_text += MessagePattern.finish_reject()
            self.finished_robots.remove(self.robot_id)
        
        if action == -1 or action == "":
            
            action = Action.get_occupancy_map.value
            print("STUCK")
            
            
        
        print("action index:",self.action_index, "action:", Action(action), ego_location)
                
        if done: # or step_count == self.num_steps:
            action = -1
            
        if not action and isinstance(action, list):
            pdb.set_trace()



        
        return action,item,message,robot,len(self.finished_robots) == self.env.action_space["robot"].n
        
        
        
    
