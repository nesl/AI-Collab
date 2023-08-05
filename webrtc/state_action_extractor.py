import glob
import pdb
import numpy as np
from gym_collab.envs.action import Action
import math
import json

def convert_to_grid_coords(x,y,scenario_size,cell_size):

    coordinate_x = cell_size*math.floor((scenario_size/2 + x)/cell_size)
    coordinate_y = cell_size*math.floor((scenario_size/2 + y)/cell_size)
    
    return coordinate_x,coordinate_y
    
def init_grid_map(wall_coords, scenario_size):
    
    grid_map = []

    for x in range(scenario_size):
        grid_map.append([])
        for y in range(scenario_size):
            if [x,y] in wall_coords:
                grid_map[x].append(1)
            else:
                grid_map[x].append(0)
                
    

    #grid_map = [[0 for y in range(scenario_size)] for y in range(scenario_size)]
    

    return grid_map
    
def print_map(occupancy_map): #Occupancy maps require special printing so that the orientation is correct
    new_occupancy_map = occupancy_map.copy()
    for row_id in range(occupancy_map.shape[0]):
        new_occupancy_map[row_id,:] = occupancy_map[occupancy_map.shape[0]-row_id-1,:]

    new_new_occupancy_map = new_occupancy_map.copy()
    for row_id in range(occupancy_map.shape[1]): 
        new_new_occupancy_map[:,row_id] = new_occupancy_map[:,occupancy_map.shape[1]-row_id-1]
    print(new_new_occupancy_map)
    
def determine_direction(origin,destination,pickup):

    diff = np.array(destination)-np.array(origin).tolist()
    
    diff = [int(diff[0]),int(diff[1])]

    
    if diff == [0,1]:
        if pickup:
            action = Action.grab_left.value
        else:
            action = Action.move_left.value    
    elif diff == [1,0]:
        if pickup:
            action = Action.grab_up.value
        else:
            action = Action.move_up.value
    elif diff == [1,1]:
        if pickup:
            action = Action.grab_up_left.value
        else:
            action = Action.move_up_left.value
    elif diff == [-1,0]:
        if pickup:
            action = Action.grab_down.value
        else:
            action = Action.move_down.value
    elif diff == [0,-1]:
        if pickup:
            action = Action.grab_right.value
        else:
            action = Action.move_right.value
    elif diff == [-1,-1]:
        if pickup:
            action = Action.grab_down_right.value
        else:
            action = Action.move_down_right.value
    elif diff == [-1,1]:
        if pickup:
            action = Action.grab_down_left.value
        else:
            action = Action.move_down_left.value
    elif diff == [1,-1]:
        if pickup:
            action = Action.grab_up_right.value
        else:
            action = Action.move_up_right.value
    else:
        action = -1
        
    return action

    
"""
def get_action_index(key):

    "Q","A","D"

    move_up = 0
move_down = 1
move_left = 2
move_right = 3
move_up_right = 4
move_up_left = 5
move_down_right = 6
move_down_left = 7
grab_up = 8
grab_right = 9
grab_down = 10
grab_left = 11
grab_up_right = 12
grab_up_left = 13
grab_down_right = 14
grab_down_left = 15
drop_object = 16

danger_sensing = 17
get_occupancy_map = 18
get_objects_held = 19
check_item = 20
check_robot = 21
get_messages = 22
send_message = 23
request_item_info = 24
request_agent_info = 25

"""


for d in glob.glob("*.txt"):
    d = "2023_08_04_15_08_34.txt"
    log_file = open(d)
    new_line = log_file.readline()
    messages_present = False
    log_state_file = open("../../simulator/log/" + d[:-4] + "_state.txt")
    num_objects = -1
    action_sample = {"action":-1,"item":-1,"robot":-1}
    state_sample = {"object_weight": 0, "object_danger_level": 0, "object_confidence": 0, "object_location": [-1,-1], "grid_map":[]}
    
    
    settings = eval(log_state_file.readline().strip())
    arguments = log_state_file.readline().strip()
    scenario_options = eval(log_state_file.readline().strip())
    
    agents_dict = {robot[0]:{"last_action":action_sample.copy(),"location":[],"last_grid_world":[], "changed":False, "object_carried":-1, "object_mapping":{}, "reward":0} for robot in scenario_options["robots_type"]}
    objects_dict = {}
    
    wall_coords = []
    for wall in scenario_options["walls"]:
        wall_p1 = np.array(convert_to_grid_coords(wall[0][0],wall[0][1],scenario_options["scenario_size"],settings["cell_size"]))
        wall_p2 = np.array(convert_to_grid_coords(wall[1][0],wall[1][1],scenario_options["scenario_size"],settings["cell_size"]))
        
        num_cells = wall_p2-wall_p1

        sign = [int(num_cells[0] < 0),int(num_cells[1] < 0)]

        num_cells = np.absolute(num_cells)        

        
        
        for x_idx in range(int(num_cells[0])+1):
            for y_idx in range(int(num_cells[1])+1):
                coord = wall_p1+np.array([x_idx*(-1)**sign[0],y_idx*(-1)**sign[1]]).tolist()
                coord = [int(coord[0]),int(coord[1])]
                wall_coords.append(coord)

    for env_object in scenario_options["env_objects"]:
        env_p = convert_to_grid_coords(env_object[0],env_object[1],scenario_options["scenario_size"],settings["cell_size"])
        env_p = [int(env_p[0]),int(env_p[1])]

    
        wall_coords.append(env_p)
        
        
    
    grid_map = init_grid_map(wall_coords, scenario_options["scenario_size"]) #[[0 for y in range(scenario_options["scenario_size"])] for y in range(scenario_options["scenario_size"])]
    
    log_memory = []
    
    while new_line:
        split_line = new_line.strip().split(',')
        
        log_line_type = int(split_line[1])

        
        if log_line_type == 0:
        
            m_idx1 = new_line.index('{')-1
            m_idx2 = new_line.index('}')+2
        
            
            metadata = json.loads(json.loads(new_line[m_idx1:m_idx2]))

            
            if num_objects == -1:
                num_objects = sum([1 if not m[0] else 0 for m in metadata["metadata"]])

                    
                
            
            
            new_grid_map = init_grid_map(wall_coords, scenario_options["scenario_size"]) #[[0 for y in range(scenario_options["scenario_size"])] for y in range(scenario_options["scenario_size"])]
            for m in metadata["metadata"]:
                if not m[0]: #object
                
                    object_id = int(m[1])
                
                    if object_id not in objects_dict:
                        objects_dict[object_id] = {"real_location":[],"location":[], "weight":m[4], "danger_level":m[5]}
                
                    coordinate_x,coordinate_y = convert_to_grid_coords(m[2],m[3],scenario_options["scenario_size"],settings["cell_size"])
                    new_grid_map[int(coordinate_x)][int(coordinate_y)] = 2
             
                    
                    if objects_dict[object_id]["location"]:
                        if objects_dict[object_id]["location"][-1] != [coordinate_x,coordinate_y]:
                            objects_dict[object_id]["location"].append([coordinate_x,coordinate_y])
                            objects_dict[object_id]["real_location"].append([m[2],m[3]])
                    else:
                        objects_dict[object_id]["location"].append([coordinate_x,coordinate_y])
                        objects_dict[object_id]["real_location"].append([m[2],m[3]])
                        
                    if len(objects_dict[object_id]["location"]) > 2:
                        objects_dict[object_id]["location"].pop(0)
                        objects_dict[object_id]["real_location"].pop(0)
                    
                else: #agent
                
                    disabled = m[4]
                    
                    if disabled:
                        continue
                    
                    agents_dict[m[1]]["last_action"] = action_sample.copy()
                    agents_dict[m[1]]["reward"] = 0
                    
                    coordinate_x,coordinate_y = convert_to_grid_coords(m[2],m[3],scenario_options["scenario_size"],settings["cell_size"])
                    
                    strength = m[7]
                    
                    carrying_objects = [int(m[5]),int(m[6])]
                    carrying_objects_bool = any(True if ob >= 0 else False for ob in carrying_objects)
                    
                    if m[1] == 'A':
                        new_grid_map[int(coordinate_x)][int(coordinate_y)] = 5
                    else:
                        new_grid_map[int(coordinate_x)][int(coordinate_y)] = 3
                    
                    if agents_dict[m[1]]["object_carried"] == -1 and carrying_objects_bool:

                        
                        if carrying_objects[0] > -1:
                            object_id = carrying_objects[0]
                        else:
                            object_id = carrying_objects[1]
                            
                        agents_dict[m[1]]["object_carried"] = object_id
                        
                        if agents_dict[m[1]]["location"][-1] != objects_dict[object_id]["location"][0]:
                            agent_location = agents_dict[m[1]]["location"][-1]
                        else:
                            agent_location = agents_dict[m[1]]["location"][0]
                        
                        agents_dict[m[1]]["last_action"]["action"] = determine_direction(agent_location,objects_dict[object_id]["location"][0],True)
                        agents_dict[m[1]]["changed"] = True
                    
                    elif agents_dict[m[1]]["object_carried"] > -1 and agents_dict[m[1]]["object_carried"] not in carrying_objects:
                        object_id = agents_dict[m[1]]["object_carried"]
                        #Check weight
                        if strength >= objects_dict[object_id]["weight"]:
                            agents_dict[m[1]]["last_action"]["action"] = Action.drop_object.value
                            
                            if np.linalg.norm(objects_dict[object_id]["real_location"][-1]) < settings['goal_radius']:
                                agents_dict[m[1]]["reward"] = 1
                            
                        agents_dict[m[1]]["changed"] = True
                        agents_dict[m[1]]["object_carried"] = -1
                            
                    
                    if agents_dict[m[1]]["location"]:
                        if agents_dict[m[1]]["location"][-1] != [coordinate_x,coordinate_y]:

                            action_result = determine_direction(agents_dict[m[1]]["location"][-1],[coordinate_x,coordinate_y],False)
                            
                            if action_result == -1: #If a movement skips through cells, restart
                                agents_dict[m[1]]["location"] = [[coordinate_x,coordinate_y]]
                                
                                continue
                                
                            agents_dict[m[1]]["last_action"]["action"] = action_result
                       
                            agents_dict[m[1]]["location"].append([coordinate_x,coordinate_y])
                            agents_dict[m[1]]["changed"] = True
                    else:
                        agents_dict[m[1]]["location"].append([coordinate_x,coordinate_y])
                        #agents_dict[m[1]]["changed"] = True
                        
                    if len(agents_dict[m[1]]["location"]) > 2:
                        agents_dict[m[1]]["location"].pop(0)
                    
                    #If nothing happened, wait
                    if agents_dict[m[1]]["last_action"]["action"] == -1:
                        agents_dict[m[1]]["last_action"]["action"] = Action.wait.value
                    
            for a_key in agents_dict.keys():
                if agents_dict[a_key]["changed"]:
                    agents_dict[a_key]["changed"] = False
                    
                    current_state = state_sample.copy()
                    next_state = state_sample.copy()
                    current_state["grid_map"] = grid_map
                    next_state["grid_map"] = new_grid_map
                    log_memory.append([a_key,current_state,agents_dict[a_key]["last_action"],next_state,agents_dict[a_key]["reward"]])
                    #print(a_key,grid_map,Action(agents_dict[a_key]["last_action"]),new_grid_map,0)
                    print(a_key,Action(agents_dict[a_key]["last_action"]["action"]), "REWARD", agents_dict[a_key]["reward"])
                    #print_map(np.array(grid_map))
                    print_map(np.array(new_grid_map))
                    
            grid_map = new_grid_map
                    
            
                    
        
        elif log_line_type == 1:
            a_key = split_line[2]
            key = split_line[3]

            if key == "Q" or key == "B":
                agent_action = action_sample.copy()
                agent_action["action"] = Action.danger_sensing.value
                
                current_state = state_sample.copy()
                current_state["grid_map"] = grid_map
                
                log_memory.append([a_key,current_state,agent_action,current_state,0])
                print(Action.danger_sensing)
                #agents_dict[a_key]["activate_sensor"] = True

        
            
        elif log_line_type == 2:
            if not messages_present:
                print(d)
                messages_present = True
                
        elif log_line_type == 3:
            a_key = split_line[2]
            m_idx1 = new_line.index('{')
            m_idx2 = new_line.rfind('}')+1

            object_results = json.loads(new_line[m_idx1:m_idx2])
            agent_action = action_sample.copy()
            agent_action["action"] = Action.check_item.value
            
            current_state = state_sample.copy()
            next_state = state_sample.copy()
            current_state["grid_map"] = grid_map
            next_state["grid_map"] = grid_map
            
            for ob_key in object_results.keys():
            
                new_measurement = False
            
                weight = object_results[ob_key]["weight"]
                time = object_results[ob_key]["time"]
                location = object_results[ob_key]["location"]

                if object_results[ob_key]["sensor"]:              
                    danger_level = object_results[ob_key]["sensor"][a_key]["value"]
                    confidence = object_results[ob_key]["sensor"][a_key]["confidence"]

                
                if int(ob_key) not in agents_dict[a_key]["object_mapping"].keys():
                    new_idx = len(agents_dict[a_key]["object_mapping"].keys())
                    agents_dict[a_key]["object_mapping"][int(ob_key)] = {"index":new_idx,"time":time,"location":location}
                    new_measurement = True
                else:
                    if agents_dict[a_key]["object_mapping"][int(ob_key)]["time"] < time and location[0] != agents_dict[a_key]["object_mapping"][int(ob_key)]["location"][0] and location[1] != agents_dict[a_key]["object_mapping"][int(ob_key)]["location"][1]:
                        agents_dict[a_key]["object_mapping"][int(ob_key)]["time"] = time
                        agents_dict[a_key]["object_mapping"][int(ob_key)]["location"] = location
                        new_measurement = True
                
                if new_measurement:
                    agent_action["item"] = agents_dict[a_key]["object_mapping"][int(ob_key)]["index"]
                    next_state["object_weight"] = weight
                    next_state["location"] = location
                    
                    if object_results[ob_key]["sensor"]:
                        next_state["object_danger_level"] = danger_level
                        next_state["object_confidence"] = confidence
                    
                    log_memory.append([a_key,current_state,agent_action,agent_action,next_state,0])
                    print(agent_action, weight,danger_level,confidence,location)
                    
                    current_state = next_state
                
             
        new_line = log_file.readline()
    break

