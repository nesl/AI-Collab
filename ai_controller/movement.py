import pdb
import numpy as np
from cnl import MessagePattern
import itertools
from gym_collab.envs.action import Action
import time
import random
from enum import Enum
import re

class Movement:

    def __init__(self, env):
    
        self.potential_occupied_locations = []
        self.occupied_locations = []
        self.ignore_robots = []
        self.ignore_go_location = []
        self.stuck_retries = 0
        self.stuck_wait_moving = 0
        self.go_retries = 0
        self.previous_go_location = []
        self.stuck_moving = 0
        self.wait_locations = []
        self.env = env
        
        self.wait_requester = ""
        #self.asked_help = False
        #self.helping = []
        #self.accepted_help = ""
        #self.asked_time = 0
        self.help_status_info = [[],0,[],[],"",[],[],[]]
        self.help_status = self.HelpState.no_request
        self.last_action_index = -1
        self.wait_time_limit = 15#5
        self.help_time_limit = random.randrange(self.wait_time_limit,30)
        self.pending_location = []
        self.ignore_object = []
        #self.being_helped = []
        #self.being_helped_locations = []
        #self.being_helped_combinations = []
        self.follow_location = []


    class State(Enum):
        wait_message = 8
        follow = 10
        wait_random = 11
        wait_free = 12
        obey = 13
        wait_follow = 14
        
    class HelpState(Enum):
        no_request = 0
        asking = 1
        accepted = 2
        being_helped = 3
        helping = 4
        

    def calculateHValue(self,current,dest,all_movements):

        dx = abs(current[0] - dest[0])
        dy = abs(current[1] - dest[1])
     
        
        if all_movements:   
            D = 1
            D2 = np.sqrt(2)
     
            h = D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
    
        else:    
            h = dx + dy #For only four movements

        return h


    def tracePath(self,node_details,dest):
        path = []
        
        currentNode = dest

        while node_details[currentNode[0]][currentNode[1]]["parent"][0] != currentNode[0] or node_details[currentNode[0]][currentNode[1]]["parent"][1] != currentNode[1]:
            path.append(currentNode)
            currentNode = node_details[currentNode[0]][currentNode[1]]["parent"]
            
        path.reverse()
        
        return path
            

    def findPath(self,startNode,endNode,occMap,ignore=[],all_movements=True):

        all_movements = False

        if min(endNode) == -1 or any(endNode >= occMap.shape) or (endNode[0] == startNode[0] and endNode[1] == startNode[1]):
            return []
            
        '''
        if occMap[endNode[0],endNode[1]] != 0:
            possible_locations = np.array([[1,1],[-1,1],[1,-1],[-1,-1],[-1,0],[1,0],[0,1],[0,-1]])
            found_location = False
            for p in possible_locations:
                new_location = endNode + p
                
                if min(new_location) == -1 or any(new_location >= occMap.shape):
                    continue
                
                if occMap[new_location[0],new_location[1]] == 0:
                    endNode = new_location
                    found_location = True
                    break
            
            if not found_location:
                return []
            print("changed destination to",endNode)
        '''

        openSet = [startNode]
        closedSet = []
        
        highest_cost = float('inf') #2147483647
        
        node_details = {}
        
        for s0 in range(occMap.shape[0]):
            node_details[s0] = {}
            for s1 in range(occMap.shape[1]):
                if s0 == startNode[0] and s1 == startNode[1]:
                    node_details[s0][s1] = {"f":0, "g":0, "h":0, "parent":[startNode[0],startNode[1]]}
                else:
                    node_details[s0][s1] = {"f":highest_cost, "g":highest_cost, "h":highest_cost, "parent":[-1,-1]}
        

        
        for ig in ignore: #Remove ignore nodes
            closedSet.append(tuple(ig))
        
        if all_movements:
            next_nodes = np.array([[1,1],[-1,1],[1,-1],[-1,-1],[-1,0],[1,0],[0,1],[0,-1]]) #np.array([[-1,0],[1,0],[0,1],[0,-1]]) #np.array([[1,1],[-1,1],[1,-1],[-1,-1],[-1,0],[1,0],[0,1],[0,-1]])
        else:
            next_nodes = np.array([[-1,0],[1,0],[0,1],[0,-1]])

        while openSet:
        
            currentNode = openSet.pop(0)
            closedSet.append(tuple(currentNode))
            
     
                
            for nx in next_nodes:
                neighborNode = currentNode + nx
                
                if neighborNode[0] == endNode[0] and neighborNode[1] == endNode[1]:
                    node_details[neighborNode[0]][neighborNode[1]]["parent"] = currentNode
                    return self.tracePath(node_details, endNode)
                
                if min(neighborNode) == -1 or any(neighborNode >= occMap.shape) or not (occMap[neighborNode[0],neighborNode[1]] == 0 or occMap[neighborNode[0],neighborNode[1]] == 3 or occMap[neighborNode[0],neighborNode[1]] == -2) or tuple(neighborNode) in closedSet: #modified to allow a robot to step into another robot's place
                    continue

            
                gNew = node_details[currentNode[0]][currentNode[1]]["g"] + 1
                hNew = self.calculateHValue(neighborNode,endNode,all_movements)
                fNew = gNew + hNew
                
                if node_details[neighborNode[0]][neighborNode[1]]["f"] == highest_cost or node_details[neighborNode[0]][neighborNode[1]]["f"] > fNew:
                    openSet.append(neighborNode)
                    
                    node_details[neighborNode[0]][neighborNode[1]]["f"] = fNew
                    node_details[neighborNode[0]][neighborNode[1]]["g"] = gNew
                    node_details[neighborNode[0]][neighborNode[1]]["h"] = hNew
                    node_details[neighborNode[0]][neighborNode[1]]["parent"] = currentNode
                    

        return [] #No path


    def position_to_action(self,current_pos,dest,pickup):
        
        res = np.array(dest) - np.array(current_pos) 
        
        action = -1
        
        if int(res[0]) == 0 and res[1] > 0:
            if pickup:
                action = Action.grab_left.value
            else:
                action = Action.move_left.value
        elif int(res[0]) == 0 and res[1] < 0:
            if pickup:
                action = Action.grab_right.value
            else:
                action = Action.move_right.value
        elif res[0] > 0 and int(res[1]) == 0:
            if pickup:
                action = Action.grab_up.value
            else:
                action = Action.move_up.value
        elif res[0] < 0 and int(res[1]) == 0:
            if pickup:
                action = Action.grab_down.value
            else:
                action = Action.move_down.value
        elif res[0] > 0 and res[1] > 0:
            if pickup:
                action = Action.grab_up_left.value
            else:
                action = Action.move_up_left.value
        elif res[0] < 0 and res[1] > 0:
            if pickup:
                action = Action.grab_down_left.value
            else:
                action = Action.move_down_left.value
        elif res[0] < 0 and res[1] < 0:
            if pickup:
                action = Action.grab_down_right.value
            else:
                action = Action.move_down_right.value
        elif res[0] > 0 and res[1] < 0:
            if pickup:
                action = Action.grab_up_right.value
            else:
                action = Action.move_up_right.value
        else:
            #pdb.set_trace()
            pass
            

        
        return action
        

    def get_combinations(self,lst): # creating a user-defined method
        combination = [] # empty list 
        for r in range(1, len(lst) + 1):
            # to generate combination
            combination.extend(itertools.combinations(lst, r))
          
        return combination
        
    def check_safe_direction(self, location):
        
        for ax in location:
            if ax < 0:
                return False
            elif ax >= self.env.map_config['num_cells'][0]:
                return False
                
        return True

    def go_to_location(self, x, y, occMap, robotState, info, ego_location, action_index, end=False,checking=False):
        
        message_text = ""
        
        occMap_clean = np.copy(occMap)
        locations_to_test = [[1,0],[0,1],[1,1],[-1,0],[0,-1],[-1,-1],[-1,1],[1,-1]]

        all_movements = not robotState.object_held
        action = -1 #For checking
        
        path_to_follow = self.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),occMap,ignore=self.ignore_go_location, all_movements=all_movements)
        
        
        
        
        if x == ego_location[0][0] and y == ego_location[1][0]: #In case we are already in the destination
            action = []
            if not checking:
                self.stuck_retries = 0
        elif not path_to_follow: #This means there is no feasible path, maybe because robots are blocking the path
        

            if self.help_status == self.HelpState.helping: #self.helping:
                agent_idx = info['robot_key_to_index'][self.help_status_info[0][0]]
                helping_robot_location = robotState.robots[agent_idx]["neighbor_location"]
                
                if not (helping_robot_location[0] == -1 and helping_robot_location[1] == -1):
                    occMap_clean[helping_robot_location[0],helping_robot_location[1]] = 1
        
            if self.ignore_robots: #Modify the occMap by removing the robots, to see if that works
            
                nearby_robots = []
                for rb_idx in self.ignore_robots:
                    rb = robotState.robots[rb_idx]["neighbor_location"]
                    
                    if not (rb[0] == -1 and rb[1] == -1):
                        if self.help_status == self.HelpState.helping and rb == helping_robot_location: #Do not ignore the robot you are helping out
                            continue
                            
                        occMap_clean[rb[0],rb[1]] = 0
                        if self.env.compute_real_distance(rb,[ego_location[0][0],ego_location[1][0]]) < self.env.map_config['communication_distance_limit']:
                            nearby_robots.append(rb)
                    
                
                path_to_follow = self.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),occMap_clean,ignore=self.ignore_go_location, all_movements=all_movements)
                
                if path_to_follow and occMap[path_to_follow[0][0],path_to_follow[0][1]] != 0 and not checking: #Removing robots does work, make other robots move
                    robot_combinations = self.get_combinations(nearby_robots) 
                    allowed_robots_blocking = (-1,0)
                    for rc_idx,rc in enumerate(robot_combinations): #Check all possible combinations for moving robots (maybe moving one is enough or two are needed)
                        t_occMap = np.copy(occMap_clean)
                        for robot_loc in rc:
                            t_occMap[robot_loc[0],robot_loc[1]] = 1 #We create a mask with locations of robots
                            
                        temp_path_to_follow = self.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),t_occMap,ignore=self.ignore_go_location, all_movements=all_movements)
                        
                        if temp_path_to_follow and len(rc) >= allowed_robots_blocking[1]:
                            allowed_robots_blocking = (rc_idx,len(rc)) #Save combination index, move the least number of robots.
                    
                    order_robots = []
                            
                            
                    robot_index_to_key = list(info['robot_key_to_index'].keys())
                    action = -1
                    
                    for rb_idx in self.ignore_robots:
                        rb = robotState.robots[rb_idx]["neighbor_location"]
                        if not (rb[0] == -1 and rb[1] == -1) and (allowed_robots_blocking[0] == -1 or (allowed_robots_blocking[0] > -1 and rb not in robot_combinations[allowed_robots_blocking[0]])) and not (self.help_status == self.HelpState.helping and rb == helping_robot_location): #If the robot is not in the combination, move it
                            #order_robots.append(rb)
                            for nrobot_idx in range(len(robotState.robots)):

                                if not (robotState.robots[nrobot_idx]["neighbor_location"][0] == -1 and robotState.robots[nrobot_idx]["neighbor_location"][1] == -1) and robotState.robots[nrobot_idx]["neighbor_location"] == rb:
                                    robot_id = robot_index_to_key[list(info['robot_key_to_index'].values()).index(nrobot_idx)] #Get the id of the robot 
                                    break

                            
                            message_text += MessagePattern.move_request(robot_id)
                            self.wait_locations.append(rb)
                            
                            if not action_index == self.State.wait_free and not action_index == self.State.wait_random:
                                self.last_action_index = action_index
                            action_index = self.State.wait_free

                            #self.asked_time = time.time()
                            self.help_status_info[1] = time.time()


                    
                    print("Waiting: moving", x,y, path_to_follow, action_index, self.last_action_index)
                    
                    self.stuck_wait_moving += 1
                    
                    if self.stuck_wait_moving == 100:
                        #pdb.set_trace()
                        print("WAIT TOO MUCH!!!")
                        
                    
                    
                elif path_to_follow and occMap[path_to_follow[0][0],path_to_follow[0][1]] == 0: #If the next step has no robot move until you are next to a blocking robot
                    action = self.position_to_action([ego_location[0][0],ego_location[1][0]],path_to_follow[0],False)
                    
                    self.stuck_wait_moving = 0
                else: #We need to wait
                    action = -1
                    if not checking:
                        print("Waiting: Couldn't go to", x,y, path_to_follow)
                        #pdb.set_trace()
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
        #elif helping and (x == path_to_follow[1][0] and y == path_to_follow[1][1] and occMap[x,y]):
        #    action = []
        elif not checking:
            
            self.stuck_retries = 0
            current_location = [ego_location[0][0],ego_location[1][0]]
            
            if self.previous_go_location and path_to_follow[0][0] == self.previous_go_location[0] and path_to_follow[0][1] == self.previous_go_location[1]: #If it gets stuck at location
                if self.go_retries == 5:#2:

                    self.ignore_go_location.append(path_to_follow[0])
                    path_to_follow = self.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),occMap, ignore=self.ignore_go_location, all_movements=all_movements)
                    print("stuck1?", path_to_follow, self.ignore_go_location)
                    if not path_to_follow: #stuck
                        action = -1
                        print("I'm stuck!")
                        
                    self.go_retries = 0
                else:
                    self.go_retries += 1
                    print("stuck2?", path_to_follow, self.ignore_go_location, self.go_retries)
            else:
                self.go_retries = 0
                self.ignore_go_location = []
            
            if path_to_follow:
                self.previous_go_location = [path_to_follow[0][0],path_to_follow[0][1]]
                action = self.position_to_action(current_location,path_to_follow[0],False)
                       
        if not checking:
            #print("Retreis:", self.go_retries, self.previous_go_location)   
            
            if action == -1:
                self.stuck_moving += 1
            else:
                self.stuck_moving = 0
            
        return action,path_to_follow,message_text,action_index
        
        
        
    def modify_occMap(self,robotState, occMap, ego_location, info, next_loc):
    
        if robotState.object_held: #Eliminate all carried objects from the occupancy map if robot is carrying object
            carried_objects = np.where(occMap == 4)
            for c_idx in range(len(carried_objects[0])):
                occMap[carried_objects[0][c_idx],carried_objects[1][c_idx]] = 0
            

        #print(self.occupied_locations)

        for rob_loc_idx in reversed(range(len(self.occupied_locations))): #Make sure agents don't move to locations already occupied
        
            other_robot_location = self.occupied_locations[rob_loc_idx]
            if occMap[other_robot_location[0],other_robot_location[1]] != 5:
                if occMap[other_robot_location[0],other_robot_location[1]] == 0:
                    del self.occupied_locations[rob_loc_idx]  
                elif occMap[other_robot_location[0],other_robot_location[1]] == 3 and next_loc and next_loc[0][0] == other_robot_location[0] and next_loc[0][1] == other_robot_location[1]:
                    occMap[other_robot_location[0],other_robot_location[1]] = 1
                    print("modifying occmap 1", other_robot_location)
            else:
                del self.occupied_locations[rob_loc_idx]

        
        
        for rob_loc_idx in reversed(range(len(self.ignore_robots))): #We mark ignored robots as an object
        
            
            other_robot_location = robotState.robots[self.ignore_robots[rob_loc_idx]]["neighbor_location"]
      

            if (other_robot_location[0] == -1 and other_robot_location[1] == -1) or self.env.compute_real_distance([other_robot_location[0],other_robot_location[1]],[ego_location[0][0],ego_location[1][0]]) >= self.env.map_config['communication_distance_limit']:
                del self.ignore_robots[rob_loc_idx]
            elif occMap[other_robot_location[0],other_robot_location[1]] != 5:
                occMap[other_robot_location[0],other_robot_location[1]] = 1
                print("modifying occmap 2", other_robot_location)
        
        #Make sure possible directions are not blocked by other robots
        for direction in [[0,1],[1,0],[-1,0],[0,-1]]:
            new_direction = [ego_location[0][0] + direction[0],ego_location[1][0] + direction[1]]
            
            if not self.check_safe_direction(new_direction):
                continue
            
            if occMap[new_direction[0],new_direction[1]] == 3: 
                occMap[new_direction[0],new_direction[1]] = 1
                print("modifying occmap 3", new_direction)
                
                
        #print("Potential locations before", self.potential_occupied_locations)
        for pot_idx in reversed(range(len(self.potential_occupied_locations))): #Potentially occupied locations, eliminate after some time
            
            if time.time() - self.potential_occupied_locations[pot_idx][1] > 5 or occMap[self.potential_occupied_locations[pot_idx][0][0],self.potential_occupied_locations[pot_idx][0][1]]: #Seconds to eliminate
                del self.potential_occupied_locations[pot_idx]
            else:
                occMap[self.potential_occupied_locations[pot_idx][0][0],self.potential_occupied_locations[pot_idx][0][1]] = 1
                print("modifying occmap 4", self.potential_occupied_locations[pot_idx][0])
        
        #print("Potential locations", self.potential_occupied_locations)
        
                
        #Make sure the ego location is always there
        
        occMap[ego_location[0][0],ego_location[1][0]] = 5
        
        print(occMap)
     
    def wait_movement(self, agent_idx, agent, action_index):
    
        message_text = MessagePattern.wait(agent) #What happens when someone is carrying object
        if not action_index == self.State.wait_random and not action_index == self.State.wait_free:
            self.last_action_index = action_index
        action_index = self.State.wait_random
        self.wait_requester = agent_idx
        #self.asked_time = time.time()
        self.help_status_info[1] = time.time()
        
        return message_text, action_index
        
        
    def cancel_cooperation(self, initial_state, message_text, message=""):
        
        if message:
            message_text += message
        #self.asked_time = time.time()
        self.help_status_info[1] = time.time()
        self.help_status = self.HelpState.no_request
        #self.being_helped = []
        #self.being_helped_locations = []
        self.help_status_info[0] = []
        self.help_status_info[2] = []
        action = Action.get_occupancy_map.value
        
        return action,message_text,initial_state
        
        
    def send_state_info(self, action, next_loc, target_location, message_text, other_agents, nearby_other_agents, ego_location, robotState, object_of_interest, object_held):
    
        if action in [Action.move_up.value, Action.move_down.value, Action.move_left.value, Action.move_right.value, Action.move_up_right.value, Action.move_up_left.value, Action.move_down_right.value, Action.move_down_left.value]: #If it is going to move
                
            if not next_loc:
                next_loc = [np.array([ego_location[0][0],ego_location[1][0]])]
            
            if len(next_loc) < 2:
                next_loc.append(next_loc[0])
                
                
            if not target_location:
                target_loc = [ego_location[0][0],ego_location[1][0]]
            else:
                target_loc = target_location
                
        else: #It stays in a place
            next_loc = [np.array([ego_location[0][0],ego_location[1][0]]), np.array([ego_location[0][0],ego_location[1][0]])]
            target_loc = [ego_location[0][0],ego_location[1][0]]
            
        
        
        if self.help_status == self.HelpState.being_helped: #self.being_helped:
            helping = [self.env.robot_id]
        elif self.help_status == self.HelpState.helping:
            helping = self.help_status_info[0]
        else:
            helping = [] #self.helping
    
        #print("Helping", helping, self.help_status)
    
        changed = False
        
        
        for n_idx in nearby_other_agents: #If any piece of information changes, send message
        
            try:
                if other_agents[n_idx].my_location["goal_location"] != target_loc:
                    other_agents[n_idx].my_location["goal_location"] = target_loc
                    changed = True  
            except:
                pdb.set_trace()
            
            if other_agents[n_idx].my_carrying != robotState.object_held:
                other_agents[n_idx].my_carrying = robotState.object_held
                changed = True   
                
            if helping and other_agents[n_idx].my_team != helping[0]:
                other_agents[n_idx].my_team = helping[0]
                changed = True     
                
            if robotState.robots[n_idx]["neighbor_type"]: #Only other ai robots are interested in this information
                
                if other_agents[n_idx].my_location["ego_location"] != [ego_location[0][0],ego_location[1][0]]:
                    other_agents[n_idx].my_location["ego_location"] = [ego_location[0][0],ego_location[1][0]]
                    changed = True
                    
                if other_agents[n_idx].my_location["next_location"] != list(next_loc[0]):
                    other_agents[n_idx].my_location["next_location"] = list(next_loc[0])
                    changed = True
            
        if changed:
        
            if not object_of_interest:
                object_of_interest = "location"
        
            carrying_object = ""
            if robotState.object_held:
                carrying_object = object_held
                if not carrying_object: #Sometimes the agent doesn't know what it is carrying
                    carrying_object = "9999"
        
            message_text +=  MessagePattern.location(target_loc[0],target_loc[1],next_loc[0][0],next_loc[0][1], self.env.convert_to_real_coordinates, [ego_location[0][0],ego_location[1][0]], carrying_object, helping, object_of_interest)
            
            if not self.go_retries: #reset because of message
                self.previous_go_location = []
                print("reseting previous_go-location")
        
        return message_text,next_loc    

    def ask_carry_help(self, action_index, object_id, weight):
    

        message_text += MessagePattern.carry_help(object_id,weight)
        #self.asked_help = True
        #self.asked_time = time.time()
        
        self.help_status = self.HelpState.asking
        self.help_status_info[1] = time.time()
        
        if not action_index == State.wait_free and not action_index == State.wait_random:
            self.last_action_index = action_index
            
        action_index = self.State.wait_message

        return action_index, message_text

    def message_processing_carry_help_accept(self,rm, target_object, message_text, following):
    
        template_match = False
        return_value = 0
    
        if MessagePattern.carry_help_accept(self.env.robot_id) in rm[1]:
            
            template_match = True
            
            #self.asked_time = time.time()
            self.help_status_info[1] = time.time()
            
            
            if (self.help_status == self.HelpState.asking and not (following and rm[0] not in self.help_status_info[0])) or (self.help_status == self.HelpState.being_helped and len(self.help_status_info[0])+1 < target_object["weight"]):
                
                #teammate_number = len(self.being_helped)
                
                
                print("Being helped by ", rm[0])
                
                #self.being_helped.append(rm[0])
                if following:
                    self.help_status_info[0] = [rm[0]]
                else:
                    self.help_status_info[0].append(rm[0])
                
                
                #if len(self.help_status_info[0])+1 >= target_object["weight"]: #len(self.being_helped)+1 >= target_object["weight"]:
                #self.asked_help = False
                    
                self.help_status = self.HelpState.being_helped
                    
                    
                self.help_status_info[2] = []
                    
                #self.being_helped_locations = []
                return_value = 1
                    
                
                match_pattern = re.search(MessagePattern.location_regex(),message_text)
                
                if match_pattern and not match_pattern.group(7):
                    message_text = message_text.replace(match_pattern.group(), match_pattern.group() + " Helping " + self.env.robot_id + ". ")
                    
                if following:
                    message_text += MessagePattern.following(rm[0])
                else:
                    message_text += MessagePattern.follow(rm[0])
            else:
                message_text += MessagePattern.carry_help_reject(rm[0])
                
        return return_value,message_text,template_match
        
    def message_processing_carry_help(self, rm, robotState, action_index, message_text):
    
        template_match = False
    
        if re.search(MessagePattern.carry_help_regex(),rm[1]) or re.search(MessagePattern.sensing_ask_help_regex(),rm[1]): # "I need help" in rm[1]:
            
            sensing = False
            
            if re.search(MessagePattern.carry_help_regex(),rm[1]):
                rematch = re.search(MessagePattern.carry_help_regex(),rm[1])
            elif re.search(MessagePattern.sensing_ask_help_regex(),rm[1]):
                rematch = re.search(MessagePattern.sensing_ask_help_regex(),rm[1])
                sensing = True
            
            
            template_match = True
            
            if not sensing or (sensing and rematch.group(1) == self.env.robot_id):
            
                if re.search(MessagePattern.carry_help_regex(),message_text) or re.search(MessagePattern.sensing_ask_help_regex(),message_text): #This means the robot is preparing to ask for help and reject the help request, we shouldn't allow this
                
                    if re.search(MessagePattern.carry_help_regex(),message_text):
                        message_text = message_text.replace(re.search(MessagePattern.carry_help_regex(),message_text).group(), "")
                    if re.search(MessagePattern.sensing_ask_help_regex(),message_text):
                        message_text = message_text.replace(re.search(MessagePattern.sensing_ask_help_regex(),message_text).group(), "")
                        
                    #self.asked_help = False
                    #self.asked_time = time.time()
                    
                    self.help_status = self.HelpState.no_request
                    self.help_status_info[1] = time.time()
                    
                    action_index = self.last_action_index

                
                if not robotState.object_held and self.help_status == self.HelpState.no_request: #not self.helping and not self.being_helped and not self.accepted_help and not self.asked_help: # accept help request
                    message_text += MessagePattern.carry_help_accept(rm[0])
                    #self.accepted_help = rm[0]
                    
                    self.help_status = self.HelpState.accepted
                    self.help_status_info[1] = time.time()
                    self.help_status_info[0] = [rm[0]]
                    
                    if sensing == True:
                        self.help_status_info[4] = rematch.group(2)
                        self.help_status_info[5] = rematch.group(3)
                    else:
                        self.help_status_info[4] = rematch.group(2)
                        
                    #self.helping = rm[0]
                    #self.action_index = self.State.check_neighbors
                    
                else: #reject help request
                    message_text += MessagePattern.carry_help_participant_reject(rm[0])
                    print("Cannot help")
         
                
        return message_text,action_index,template_match
        
    def message_processing_help(self, rm, action_index, sensing, initial_state):
    
        template_match = False
        message = ""
                
        if re.search(MessagePattern.follow_regex(),rm[1]) or re.search(MessagePattern.following_regex(),rm[1]):
        
            template_match = True
        
            if self.help_status == self.HelpState.accepted:
                for rematch in itertools.chain(re.finditer(MessagePattern.follow_regex(),rm[1]),re.finditer(MessagePattern.following_regex(),rm[1])):
            
                    if rematch.group(1) == self.env.robot_id:
                
                        #teammate_number = int(rematch.group(2))
                        
                        #self.helping = [rm[0]]
                        
                        if sensing and rm[0] not in self.help_status_info[0]:
                            message += MessagePattern.sensing_ask_help_incorrect(rm[0])
                        else:
                            self.help_status_info[0] = [rm[0]]
                            self.help_status_info[6] = []
                            self.help_status = self.HelpState.helping
                            
                            if not sensing:
                                action_index = self.State.follow
                            
                            
                            print("HELPING")
                            break
        """            
        if re.search(MessagePattern.following_regex(),rm[1]):
        
            template_match = True
        
            for rematch in re.finditer(MessagePattern.following_regex(),rm[1]):
        
                if rematch.group(1) == self.env.robot_id:
            
                    #teammate_number = int(rematch.group(2))
                    
                    #self.helping = [rm[0]]
                    
                    self.help_status_info[0] = [rm[0]]
                    self.help_status_info[6] = []
                    self.help_status = self.HelpState.helping
                    
                    print("HELPING")
                    break      
        """    
        if MessagePattern.carry_help_cancel() in rm[1] or MessagePattern.carry_help_reject(self.env.robot_id) in rm[1] or MessagePattern.carry_help_finish() in rm[1] or MessagePattern.carry_help_complain() in rm[1]:
        
            template_match = True
            
            if self.help_status == self.HelpState.helping and self.help_status_info[0][0] == rm[0]: #self.helping and self.helping[0] == rm[0]:
                #self.accepted_help = ""
                action_index = initial_state
                print("Changed -3")
                self.help_status = self.HelpState.no_request
                self.help_status_info[0] = []
                #self.helping = []
            elif self.help_status == self.HelpState.accepted and self.help_status_info[0][0] == rm[0]: #self.accepted_help == rm[0]:
                #self.accepted_help = ""
                self.help_status = self.HelpState.no_request
                self.help_status_info[0] = []
                
                
            if MessagePattern.carry_help_reject(self.env.robot_id) in rm[1]:
                if self.help_status == self.HelpState.being_helped and rm[0] in self.help_status_info[0]:
                    self.help_status = self.HelpState.no_request
        
        if re.search(MessagePattern.object_not_found_regex(), rm[1]):
            template_match = True
            
            rematch = re.search(MessagePattern.object_not_found_regex(),rm[1])
              
            if rematch.group(1) == str(self.env.robot_id):
                if self.help_status == self.HelpState.being_helped and rm[0] in self.help_status_info[0]:
                    self.help_status = self.HelpState.no_request
        
        
                    
        #if MessagePattern.carry_help_participant_reject(self.env.robot_id) in rm[1]:
        #    #self.asked_help = False
        #    self.asked_time = time.time()
        
        
        return action_index,template_match,message
        
    def message_processing_location(self, rm, robotState, info, other_agents, target_location, action_index, message_text, initial_state, next_loc):
    

        template_match = False
           
      
        
        if re.search(MessagePattern.location_regex(),rm[1]) and not (self.help_status == self.HelpState.helping and self.help_status_info[0][0] == rm[0] and action_index == self.State.obey) and not action_index == self.State.wait_message:  #"Going towards location" in rm[1]: 
            match_pattern = re.search(MessagePattern.location_regex(),rm[1])
            
            template_match = True

            #print("location_regex", self.being_helped)

            #pdb.set_trace()
            other_target_location = self.env.convert_to_grid_coordinates(eval(match_pattern.group(2)))
            other_next_step = self.env.convert_to_grid_coordinates(eval(match_pattern.group(3)))

            agent_idx = info['robot_key_to_index'][rm[0]]
            
            if match_pattern.group(7): #Register whether other agents have already a team
                other_agents[agent_idx].team = match_pattern.group(8)
            else:
                other_agents[agent_idx].team = ""
                
            if match_pattern.group(5):
                other_agents[agent_idx].carrying = True
            else:
                other_agents[agent_idx].carrying = False
                
                
            if self.help_status == self.HelpState.helping and self.help_status_info[0][0] == rm[0] and not match_pattern.group(7): #This means the team leader disbanded the team without us knowing
                #self.helping = []
                #self.accepted_help = ""
                self.help_status = self.HelpState.no_request
                self.help_status_info[0] = []
                action_index = initial_state
                print("Changed -2")
                
                
            curr_loc = self.env.convert_to_grid_coordinates(eval(match_pattern.group(4)))
                
            other_agents[agent_idx].other_location["ego_location"] = curr_loc
            other_agents[agent_idx].other_location["goal_location"] = other_target_location
            other_agents[agent_idx].other_location["next_location"] = other_next_step
            
            curr_loc = tuple(curr_loc)
            
            if curr_loc not in self.occupied_locations:
                self.occupied_locations.append(curr_loc)
            
            if other_next_step == other_target_location: #robot stays there
                
                if agent_idx not in self.ignore_robots:
                    self.ignore_robots.append(agent_idx)
                    
                if target_location == other_target_location and not action_index == self.State.follow and not action_index == self.State.obey and not self.help_status == self.HelpState.helping: #Change destination
                    action_index = initial_state
                    print("Changed -1")
                    if self.help_status == self.HelpState.being_helped: #self.being_helped:
                        #self.being_helped = []
                        #self.being_helped_locations = []
                        self.help_status = self.HelpState.no_request
                        self.help_status_info[0] = []
                        self.help_status_info[2] = []
                        
                        message_text += MessagePattern.carry_help_finish()
            else:
                if target_location == other_target_location:
                
                
                    if not match_pattern.group(5) and not robotState.object_held and not match_pattern.group(7) and not self.help_status == self.HelpState.helping and not self.help_status == self.HelpState.being_helped: #not self.helping and not self.being_helped: #Possible change !!!!
                
                 
                
                        #if rm[2] <= self.message_send_time: #Message arrive at the same time or previous than this robot sent its message.  
                        
                        #if rm[2] == self.message_send_time: #rules to disambiguate are based on alphabetic order
                            
                        if ord(rm[0]) < ord(self.env.robot_id): #If sender's id appears first than receiver in alphabetic order
                            self.ignore_object.append(other_target_location)
                            print("Changed 0")
                            action_index = initial_state
                            
                            if self.help_status == self.HelpState.being_helped: #self.being_helped:
                                #self.being_helped = []
                                #self.being_helped_locations = []
                                self.help_status = self.HelpState.no_request
                                self.help_status_info[0] = []
                                self.help_status_info[2] = []
                                message_text += MessagePattern.carry_help_finish()
                                
                            
                            
                        
                            if re.search(MessagePattern.location_regex(),message_text):
                                message_text = message_text.replace(re.search(MessagePattern.location_regex(),message_text).group(), "")
                                
                                if message_text.isspace():    
                                    message_text = ""
                                print("changing going location!!!")
                                

                    elif (match_pattern.group(7) and (self.help_status == self.HelpState.helping or self.help_status == self.HelpState.being_helped)) or (match_pattern.group(5) and robotState.object_held) or (match_pattern.group(5) and (self.help_status == self.HelpState.helping or self.help_status == self.HelpState.being_helped)) or (match_pattern.group(7) and robotState.object_held):
                        if match_pattern.group(7):
                            other_id = match_pattern.group(8)
                        else:
                            other_id = rm[0]
                            
                        if self.help_status == self.HelpState.helping: #self.helping: 
                            our_id = self.help_status_info[0][0] #self.helping[0]
                        else:
                            our_id = self.env.robot_id
                                
                                
                        if ord(other_id) < ord(our_id): #If sender's id appears first than receiver in alphabetic order
                            #Move to free location
                            self.ignore_object.append(other_target_location)
                            print("Changed 2")
                            action_index = initial_state
                            if self.help_status == self.HelpState.being_helped: #self.being_helped:
                                #self.being_helped = []
                                #self.being_helped_locations = []
                                self.help_status = self.HelpState.no_request
                                self.help_status_info[0] = []
                                self.help_status_info[2] = []
                                message_text += MessagePattern.carry_help_finish()
                    
                    elif match_pattern.group(7) or match_pattern.group(5):
                        self.ignore_object.append(other_target_location)
                        action_index = initial_state
                        print("Changed 3")
                        if self.help_status == self.HelpState.being_helped: #self.being_helped:
                            #self.being_helped = []
                            #self.being_helped_locations = []
                            self.help_status = self.HelpState.no_request
                            self.help_status_info[0] = []
                            self.help_status_info[2] = []
                            message_text += MessagePattern.carry_help_finish()
                                         
                else: #If we are not going to same destination, just ignore temporarily the other location
                    self.ignore_object.append(other_target_location)
                    
                    
            
            
                if next_loc:

                
       
            
                    if (other_next_step == next_loc[0].tolist() or (len(next_loc) > 1 and other_next_step == next_loc[1].tolist())):
                    
                        if not match_pattern.group(5) and not robotState.object_held and not match_pattern.group(7) and not self.help_status == self.HelpState.helping and not self.help_status == self.HelpState.being_helped: #Message arrive at the same time or previous than this robot sent its message. This condition is true only when robots have no teams and are not carrying any object
                        
                            #if rm[2] == self.message_send_time: #If helping the one who send the message, automatically wait

                            other_id = rm[0]
                            our_id = self.env.robot_id
                            
                            if ord(other_id) < ord(our_id): #If sender's id appears first than receiver in alphabetic order
                                #Move to free location
                                #print(rm[2],self.message_send_time)
                                message_text_tmp, action_index = self.wait_movement(agent_idx,rm[0],action_index)
                                message_text += message_text_tmp
                            """
                            else:
                                print(rm[2],self.message_send_time)
                                self.wait_movement(agent_idx,rm[0])
                            """
                        elif (match_pattern.group(7) and (self.help_status == self.HelpState.helping or self.help_status == self.HelpState.being_helped)) or (match_pattern.group(5) and robotState.object_held) or (match_pattern.group(5) and (self.help_status == self.HelpState.helping or self.help_status == self.HelpState.being_helped)) or (match_pattern.group(7) and robotState.object_held): #Priority given to robot teamleader or robot carrying object with robot id that appears first in alphabetic order
                        
                            if match_pattern.group(7):
                                other_id = match_pattern.group(8)
                            else:
                                other_id = rm[0]
                                
                            if self.help_status == self.HelpState.helping: #self.helping: 
                                our_id = self.help_status_info[0][0] #self.helping[0]
                            else:
                                our_id = self.env.robot_id
                                    
                                    
                            if ord(other_id) < ord(our_id): #If sender's id appears first than receiver in alphabetic order
                                #Move to free location
                                #print(rm[2],self.message_send_time)
                                message_text_tmp, action_index = self.wait_movement(agent_idx,rm[0],action_index)
                                message_text += message_text_tmp

                        elif match_pattern.group(7) or match_pattern.group(5): #If we are not carrying an object while the other is, or we are not part of a team while the other is
                            #print(rm[2],self.message_send_time)
                            message_text_tmp, action_index = self.wait_movement(agent_idx,rm[0],action_index)
                            message_text += message_text_tmp
                            
                    else:

                        if match_pattern.group(7):
                            other_id = match_pattern.group(8)
                        else:
                            other_id = rm[0]
                            
                        if self.help_status == self.HelpState.helping: #self.helping: 
                            our_id = self.help_status_info[0][0] #self.helping[0]
                        else:
                            our_id = self.env.robot_id
                        
                        other_id = rm[0]
                        our_id = self.env.robot_id
                            
                        previous_index = -1
                        
                        if match_pattern.group(7) or match_pattern.group(5) or (ord(other_id) < ord(our_id) and (((match_pattern.group(7) and (self.help_status == self.HelpState.helping or self.help_status == self.HelpState.being_helped)) or (match_pattern.group(5) and robotState.object_held) or (match_pattern.group(5) and (self.help_status == self.HelpState.helping or self.help_status == self.HelpState.being_helped)) or (match_pattern.group(7) and robotState.object_held)) or not match_pattern.group(5) and not robotState.object_held and not match_pattern.group(7) and not self.help_status == self.HelpState.helping and not self.help_status == self.HelpState.being_helped)):
                        
                            for s_idx,s in enumerate(self.potential_occupied_locations):
                                if s[0] == other_next_step:
                                    previous_index = s_idx
                                
                            if previous_index == -1:
                                self.potential_occupied_locations.append([other_next_step,time.time()])
                                
                            else:
                                self.potential_occupied_locations[previous_index][1] = time.time()
        
        return message_text,action_index,template_match
        
                                
    def message_processing_wait(self, rm, info, target_location, action_index):
        
        template_match = False
                            
        if MessagePattern.wait(self.env.robot_id) in rm[1]:
            
            template_match = True
        
            agent_idx = info['robot_key_to_index'][rm[0]]
            #other_robot_location = robotState.robots[agent_idx]["neighbor_location"]
            #self.movement.ignore_robots.append(other_robot_location)
            if agent_idx not in self.ignore_robots:
                self.ignore_robots.append(agent_idx)
                
        if re.search(MessagePattern.move_order_regex(),rm[1]):
            rematch = re.search(MessagePattern.move_order_regex(),rm[1])
            
            template_match = True
            
            if rematch.group(1) == self.env.robot_id and self.help_status == self.HelpState.helping: #self.helping:
                target_location = self.env.convert_to_grid_coordinates(eval(rematch.group(2)))
                action_index = self.State.obey
                
        return target_location, action_index, template_match
        
    
            
            
    def message_processing_move_request(self, rm, robotState, info, action_index, message_text, other_agents):
    
        template_match = False
        
        if MessagePattern.move_request(self.env.robot_id) in rm[1]: # and not (last_move_request and last_move_request == rm[0]):
        
            template_match = True
            
            agent_idx = info['robot_key_to_index'][rm[0]]
            other_robot_location = robotState.robots[agent_idx]["neighbor_location"]
            
            if not (other_robot_location[0] == -1 and other_robot_location[1] == -1) and ((not robotState.object_held and not self.help_status == self.HelpState.being_helped and (not self.help_status == self.HelpState.helping or (self.help_status == self.HelpState.helping and (self.help_status_info[0][0] == rm[0] or rm[0] in self.help_status_info[6])))) or other_agents[agent_idx].carrying):#not self.being_helped and (not self.helping or (self.helping and self.helping[0] == rm[0])): #This condition is true only when robots have no teams and are not carrying any object
                        
                print("MOVING")
                
                #last_move_request = rm[0]
                
                possible_locations = [[1,0],[0,1],[-1,0],[0,-1]] #,[1,1],[1,-1],[-1,1],[-1,-1]]
                ego_location = np.where(robotState.latest_map == 5)
                
                maximum_distance = 0
                maximum_distance_with_robot = 0
                
                for p in possible_locations:
                    ego_location2 = [ego_location[0][0] + p[0],ego_location[1][0] + p[1]]

                    if robotState.latest_map[ego_location2[0],ego_location2[1]] == 0:
                        temp_distance = self.env.compute_real_distance([other_robot_location[0],other_robot_location[1]],ego_location2)
                        if temp_distance > maximum_distance:
                            next_location = p
                            maximum_distance = temp_distance
                    elif robotState.latest_map[ego_location2[0],ego_location2[1]] == 3 and ego_location2 != other_robot_location:
                        temp_distance = self.env.compute_real_distance([other_robot_location[0],other_robot_location[1]],ego_location2)
                        if temp_distance > maximum_distance_with_robot:
                            next_location_with_robot = p
                            maximum_distance_with_robot = temp_distance
                
                if maximum_distance:            
                    ego_location2 = [ego_location[0][0] + next_location[0],ego_location[1][0] + next_location[1]]
                    self.pending_location = ego_location2
                     
                    if not action_index == self.State.wait_random and not action_index == self.State.wait_free:
                        self.last_action_index = action_index
                    action_index = self.State.wait_random
                    self.wait_requester = agent_idx
                    #self.asked_time = time.time()
                    self.help_status_info[1] = time.time()
                    print("MOVING")
                elif maximum_distance_with_robot:
                    #object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.heavy_objects['index'][ho])]
                    robot_index_to_key = list(info['robot_key_to_index'].keys())
                    message_text += MessagePattern.move_request(robot_index_to_key[list(info['robot_key_to_index'].values()).index(agent_idx)])
                    if not action_index == self.State.wait_free and not action_index == self.State.wait_random:
                        self.last_action_index = action_index
                    action_index = self.State.wait_free
                    #self.asked_time = time.time()
                    self.help_status_info[1] = time.time()
                    self.wait_locations.append(next_location_with_robot)
                    print("MOVING robot")
                                    
        return message_text,action_index,template_match    
        
    
    def find_order_team_rec(self, help_ids, idx, robotState, info, goal_locations, limited_occ_map, num_agents):
    
        
        help_idx = help_ids[idx]
        agent_id = self.help_status_info[0][help_idx] #self.being_helped[help_idx]
        agent_idx = info['robot_key_to_index'][agent_id]
        agent_location = robotState.robots[agent_idx]["neighbor_location"]
            
        if (agent_location[0] == -1 and agent_location[1] == -1):
            return []
            
        limited_occ_map_copy = np.copy(limited_occ_map)
        limited_occ_map_copy[agent_location[0],agent_location[1]] = 3
        
        goal_distances = [float("inf")]*len(goal_locations)
        
        for gl_idx,gl in enumerate(goal_locations):
        
            if not limited_occ_map_copy[gl[0],gl[1]] == 1 and not limited_occ_map_copy[gl[0],gl[1]] == 2 and not limited_occ_map_copy[gl[0],gl[1]] == 4:
            
                if gl == agent_location:
                    goal_distances[gl_idx] = 0
                else:
            
                    possible_path = self.findPath(np.array(agent_location),np.array([gl[0],gl[1]]),limited_occ_map_copy,all_movements=False)
                
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
                        
    
    def wait_for_others_func(self,occMap, info, robotState, nearby_other_agents, next_locations, ego_location, message_text):
    
        wait_for_others = False    
        combinations_found = True
        within_comms_range = True
        previous_agent_location = []
        
        ego_location = [ego_location[0][0],ego_location[1][0]]
                                
        if self.help_status == self.HelpState.being_helped: #self.being_helped:
        
            ai_robots = []
            human_robots_idx = []
            for agent_id in self.help_status_info[0]:
                agent_idx = info['robot_key_to_index'][agent_id]
                        
                if robotState.robots[agent_idx]["neighbor_type"]:
                    ai_robots.append(agent_id)
                else:
                    human_robots_idx.append(agent_idx)
            
            
            humans_close = [self.env.compute_real_distance(robotState.robots[ag_idx]["neighbor_location"],ego_location) < self.env.map_config["strength_distance_limit"]-1 for ag_idx in human_robots_idx]
            wait_for_others = True
            
            #pdb.set_trace()

            if self.help_status_info[2] or self.help_status_info[7]: #self.being_helped_locations:
            
                if ai_robots:
                    if not next_locations:
                        help_idx = len(self.help_status_info[2])-1 #len(self.being_helped_locations)-1
                    else:
                        comb_idx = len(self.help_status_info[2])-1 #len(self.being_helped_locations)-1
                        help_idx = self.help_status_info[3][comb_idx][0] #self.being_helped_combinations[comb_idx][0]
                    
                    agent_id = ai_robots[help_idx] #self.being_helped[help_idx]
                    agent_idx = info['robot_key_to_index'][agent_id]
                    previous_agent_location = robotState.robots[agent_idx]["neighbor_location"]
                
            
            elif next_locations: #At the beginning choose the order of path assignment
            
                #According to next step, we have all the possible cells helping robots may move into
        
                if not all(humans_close):
                    for hc in range(len(humans_close)):
                        if not humans_close[hc]:
                            robot_index_to_key = list(info['robot_key_to_index'].keys())
                            robot_id = robot_index_to_key[list(info['robot_key_to_index'].values()).index(human_robots_idx[hc])]
                            message_text += MessagePattern.come_closer(robot_id)
                            self.help_status_info[7].append(robot_id)
        
                if ai_robots:
                
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
                    
                    for agent_id in ai_robots: #self.being_helped: #locations with teammates

                        agent_idx = info['robot_key_to_index'][agent_id]
                        other_robot_location = robotState.robots[agent_idx]["neighbor_location"]
                        if not (other_robot_location[0] == -1 and other_robot_location[1] == -1):
                            limited_occ_map[other_robot_location[0],other_robot_location[1]] = 1

                
                    
                    possible_permutations = list(itertools.permutations(list(range(len(ai_robots)))))
                    
                    solution_found = []
                    for perm in possible_permutations:

                        possible_perm = self.find_order_team_rec(perm,0,robotState,info,goal_locations,limited_occ_map, len(ai_robots))
                        
                        if possible_perm:
                            solution_found = perm
                            break
                            
                    if solution_found:
                        possible_perm.reverse()
                        
                        #self.being_helped_combinations = [[solution_found[p_idx],possible_perm[p_idx]] for p_idx in range(len(possible_perm))]
                        self.help_status_info[3] = [[solution_found[p_idx],possible_perm[p_idx]] for p_idx in range(len(possible_perm))]
                        print("Combinaionts", self.help_status_info[3])                    

                    else:
                        combinations_found = False
                        message_text += MessagePattern.carry_help_finish()
                        print("No possible combinations 1")
                
            else: #To wait for others to get into communication range
                agent_sum = 0
                robot_index_to_key = list(info['robot_key_to_index'].keys())
                for noa in nearby_other_agents:
                    robot_id = robot_index_to_key[list(info['robot_key_to_index'].values()).index(noa)]
                    if robot_id in self.help_status_info[0]: #self.being_helped:
                        agent_sum += 1
                            
                if agent_sum != len(self.help_status_info[0]):
                    within_comms_range = False
                    
                
                if within_comms_range and ai_robots: #Compute feasible paths if in communication range
                
        
                    limited_occ_map = np.copy(occMap)
                    
                    range1_1 = ego_location[0]-1
                    range1_2 = ego_location[0]+2
                    range2_1 = ego_location[1]-1
                    range2_2 = ego_location[1]+2
                    
                    goal_locations = [[x,y] for x in range(ego_location[0]-1,ego_location[0]+2,1) for y in range(ego_location[1]-1,ego_location[1]+2,1) if not (x == ego_location[0] and y == ego_location[1])]
                    limited_occ_map[ego_location[0],ego_location[1]] = 1
                    
                    
                    
                    for agent_id in ai_robots: #self.being_helped: #locations with teammates

                        agent_idx = info['robot_key_to_index'][agent_id]
                        other_robot_location = robotState.robots[agent_idx]["neighbor_location"]
                        if not (other_robot_location[0] == -1 and other_robot_location[1] == -1):
                            limited_occ_map[other_robot_location[0],other_robot_location[1]] = 1

            
                    possible_permutations = list(itertools.permutations(list(range(len(ai_robots)))))
                    
                    solution_found = []
                    for perm in possible_permutations:

                        possible_perm = self.find_order_team_rec(perm,0,robotState,info,goal_locations,limited_occ_map, len(ai_robots))
                        
                        if possible_perm:
                            solution_found = perm
                            break
                            
                    if solution_found:
                        possible_perm.reverse()
                        
                        #self.being_helped_combinations = [[solution_found[p_idx],possible_perm[p_idx]] for p_idx in range(len(possible_perm))]
                        self.help_status_info[3] = [[solution_found[p_idx],possible_perm[p_idx]] for p_idx in range(len(possible_perm))]
                        print("Combinaionts", self.help_status_info[3])                    

                    else:
                        combinations_found = False
                        message_text += MessagePattern.carry_help_finish()
                        print("No possible combinations 2")
                
            print("Expected locations:", self.help_status_info, previous_agent_location)
               
            
            
            if within_comms_range and combinations_found and ai_robots and (not self.help_status_info[2] or (self.help_status_info[2] and self.help_status_info[2][-1] == previous_agent_location and len(self.help_status_info[2]) != len(ai_robots))): #(not self.being_helped_locations or (self.being_helped_locations and self.being_helped_locations[-1] == previous_agent_location and len(self.being_helped_locations) != len(self.being_helped))):

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
                        message_text += MessagePattern.move_order(agent_id, new_location, self.env.convert_to_real_coordinates)
                    
                        print("NEW Location", occMap[new_location[0],new_location[1]])
                        
                        if occMap[new_location[0],new_location[1]] != 0 and occMap[new_location[0],new_location[1]] != 3:
                            pdb.set_trace()
                        
                    
                else:
                """
                
                
                comb_idx = len(self.help_status_info[2]) #len(self.being_helped_locations)
                new_location = self.help_status_info[3][comb_idx][1] #self.being_helped_combinations[comb_idx][1]

                #self.being_helped_locations.append(new_location)
                self.help_status_info[2].append(new_location)
                
                help_idx = self.help_status_info[3][comb_idx][0] #self.being_helped_combinations[comb_idx][0]
                
                agent_id = ai_robots[help_idx] #self.being_helped[help_idx]
                message_text += MessagePattern.move_order(agent_id, new_location, self.env.convert_to_real_coordinates)
                #self.asked_time = time.time()
                self.help_status_info[1] = time.time()
            
            elif len(self.help_status_info[2]) == len(ai_robots) and not (ai_robots and self.help_status_info[2][-1] != previous_agent_location) and all(humans_close): #len(self.being_helped_locations) == len(self.being_helped) and self.being_helped_locations[-1] == previous_agent_location: #When all agents have followed orders
                wait_for_others = False
                
            
            
            
            
                
        return wait_for_others,combinations_found,message_text
        
    
    def movement_state_machine(self, occMap, info, robotState, action_index, message_text, target_location, initial_state, next_loc, ego_location, action):
        
        
        if action_index == self.State.wait_message:
            if time.time() - self.help_status_info[1] > self.wait_time_limit: #time.time() - self.asked_time > self.wait_time_limit:

                #self.asked_help = False
                _,message_text,action_index = self.cancel_cooperation(initial_state, message_text, message=MessagePattern.carry_help_cancel())
                self.help_time_limit = random.randrange(self.wait_time_limit,30)
                print("end of waiting")
            action = Action.get_occupancy_map.value
            
        elif action_index == self.State.wait_follow:
            if time.time() - self.help_status_info[1] > self.wait_time_limit: #time.time() - self.asked_time > self.wait_time_limit:

                action_index = initial_state
                #self.accepted_help = ""
                self.help_status = self.HelpState.no_request
                self.help_status_info[0] = []
                print("end of waiting")
            action = Action.get_occupancy_map.value
            
        elif action_index == self.State.wait_random:
        
            #for rm in received_messages:
            #    if MessagePattern.move_request(self.env.robot_id) in rm[1]:
            #        pdb.set_trace() 
            #        break
          
        
            other_robot_location = robotState.robots[self.wait_requester]["neighbor_location"]
            #if not (self.next_loc and (occMap[self.next_loc[0][0],self.next_loc[0][1]] == 3 or (len(self.next_loc) > 1 and occMap[self.next_loc[1][0],self.next_loc[1][1]] == 3))): #Wait until there is no one in your next location
            
            if (other_robot_location[0] == -1 and other_robot_location[1] == -1) or self.env.compute_real_distance(other_robot_location,[ego_location[0][0],ego_location[1][0]]) >= self.env.map_config['communication_distance_limit'] or time.time() - self.help_status_info[1] > self.help_time_limit: #time.time() - self.asked_time > self.help_time_limit: #Until the other robot is out of range we can move
                action_index = self.last_action_index
            
            if self.pending_location and self.pending_location != [ego_location[0][0],ego_location[1][0]]:
                action = self.position_to_action([ego_location[0][0],ego_location[1][0]],self.pending_location,False)
                #action,next_loc,message_text,action_index = self.go_to_location(self.pending_location[0],self.pending_location[1],occMap,robotState,info,ego_location,action_index)
                if not action and isinstance(action, list):
                    action = Action.get_occupancy_map.value 
            else:
                action = Action.get_occupancy_map.value    
                self.pending_location = []
                
                
        elif action_index == self.State.wait_free: 
            
            for loc_wait_idx in reversed(range(len(self.wait_locations))): #Wait for robots to move from location
                loc_wait = self.wait_locations[loc_wait_idx]
                if occMap[loc_wait[0],loc_wait[1]] == 0:
                    del self.wait_locations[loc_wait_idx]
            #print(time.time() - self.asked_time)
            if not self.wait_locations or time.time() - self.help_status_info[1] > self.wait_time_limit: #time.time() - self.asked_time > self.wait_time_limit:
                action_index = self.last_action_index
                self.wait_locations = []
                print("Last action", self.last_action_index)
            else:
                action = Action.get_occupancy_map.value
                    
            
            

            """
            if self.action_index == self.State.check_neighbors:
                agent_idx = info['robot_key_to_index'][self.helping[0]]
                action = Action.check_robot.value
                robot = agent_idx
                self.action_index += 1
            """
        elif action_index == self.State.follow:
            
            agent_idx = info['robot_key_to_index'][self.help_status_info[0][0]]
            
            if not (robotState.robots[agent_idx]["neighbor_location"][0] == -1 and robotState.robots[agent_idx]["neighbor_location"][1] == -1) and robotState.robots[agent_idx]["neighbor_disabled"] != 1: 
                self.follow_location = robotState.robots[agent_idx]["neighbor_location"]
            
                target_location = self.follow_location
                

                action,next_loc,message_text,action_index = self.go_to_location(target_location[0],target_location[1],occMap,robotState,info,ego_location,action_index)
                
                
                real_distance = self.env.compute_real_distance([target_location[0],target_location[1]],[ego_location[0][0],ego_location[1][0]])
                
                if not robotState.robots[agent_idx]["neighbor_type"]: #depending on whether the robot is human controlled or not, we have different distances at which to maitain helping robots
                    distance_limit = self.env.map_config["strength_distance_limit"]-1
                else:
                    distance_limit = self.env.map_config['communication_distance_limit']-2
                
                
                
                if (not action and isinstance(action, list)) or real_distance < distance_limit:
                    action = Action.get_occupancy_map.value
            else:
                action = Action.get_occupancy_map.value
                self.help_status = self.HelpState.no_request
                action_index = initial_state
                print("Agent not found")
                #pdb.set_trace()
                    
        elif action_index == self.State.obey:
            print("TARGET LOCATION:", target_location)
            
           
            agent_idx = info['robot_key_to_index'][self.help_status_info[0][0]]
           
            helping_location = robotState.robots[agent_idx]["neighbor_location"]
            
            if not (helping_location[0] == -1 and helping_location[1] == -1):
                occMap[helping_location[0],helping_location[1]] = 1
                print("modifying occmap 10",helping_location )
            
            action,next_loc,message_text,action_index = self.go_to_location(target_location[0],target_location[1],occMap,robotState,info, ego_location, action_index,end=True)
            if (not action and isinstance(action, list)):
                action = Action.get_occupancy_map.value
        
            
            if action == -1:
                self.ignore_go_location = []
                #pdb.set_trace()
                
        
        return message_text,action_index,target_location,next_loc,action
                
