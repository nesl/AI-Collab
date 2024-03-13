import glob
import pdb
import numpy as np
from gym_collab.envs.action import Action
import math
import json
import re
import matplotlib.pyplot as plt
import time
import matplotlib.colors as mcolors

def convert_to_grid_coords(x,y,scenario_size,cell_size):

    if isinstance(scenario_size,list):
        coordinate_x = math.floor((scenario_size[0]/2 + x)/cell_size)
        coordinate_y = math.floor((scenario_size[1]/2 + y)/cell_size)
    else:
        coordinate_x = math.floor((scenario_size/2 + x)/cell_size)
        coordinate_y = math.floor((scenario_size/2 + y)/cell_size)
    
    return coordinate_x,coordinate_y
    
def convert_to_real_coords(x,y,scenario_size,cell_size):

    if isinstance(scenario_size,list):
        coordinate_x = x*cell_size+cell_size/2 - scenario_size[0]/2
        coordinate_y = y*cell_size+cell_size/2 - scenario_size[1]/2
    else:
        coordinate_x = x*cell_size+cell_size/2 - scenario_size/2
        coordinate_y = y*cell_size+cell_size/2 - scenario_size/2
    
    return coordinate_x,coordinate_y
    
def init_grid_map(wall_coords, scenario_size):
    
    grid_map = []
    
    if isinstance(scenario_size,list):
        size_x = scenario_size[0]
        size_y = scenario_size[1]
    else:
        size_x = scenario_size
        size_y = scenario_size

    for x in range(size_x):
        grid_map.append([])
        for y in range(size_y):
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

    """
    out_file = open("map.json", "w")
    json.dump({"data":new_new_occupancy_map.tolist()},out_file)
    out_file.close()
    exit()
    """
def print_results(scenario_options, agents_dict, objects_dict):

    #plt.axis([-10, 10, -10, 10])  # Adjust axis limits for various functions
    #plt.ion()
    #plt.show()
    plt.ion()
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(111)
    ax.axis([-10, 10, -10, 10])

    keys = list(agents_dict.keys())

    if keys and "stats" in agents_dict[keys[0]]:
        printing_stats = True
    else:
        printing_stats = False

    if printing_stats:
        print("{:^50}".format(""),end=" ")
                        
    for w2 in scenario_options["walls"]:
        wall = np.array(w2)
        plt.plot(wall[:,0], wall[:,1], color="black")
      
    for w2 in scenario_options["env_objects"]:
        plt.scatter(w2[0], w2[1], color="black")
         
    objects_to_carry = {}
    
    ranges = [[],[]]
    
    for ob_key in objects_dict.keys():
        locs = objects_dict[ob_key]["stored_locations"]
        
        if len(locs):
            if objects_dict[ob_key]["danger_level"] == 1:
                color_danger = "blue" 
            elif objects_dict[ob_key]["danger_level"] == 2:
                color_danger = "red"
                
            objects_to_carry[ob_key] = plt.scatter(locs[0][0], locs[0][1], s=500, color=color_danger, marker="$" + str(ob_key) + "\_" + str(objects_dict[ob_key]["weight"]) + "$")
            
        #if len(objects_dict[ob_key]["drop_times"]):             

    locs = {}
    messages = {}
    #pdb.set_trace()
    colors = list(mcolors.TABLEAU_COLORS.keys())
    color_assigned = {}
    objects_carried = {}
    for b_idx,b_key in enumerate(agents_dict.keys()):
        color_assigned[b_key] = colors[b_idx]
        locs[b_key] = np.array(agents_dict[b_key]["locations"])
        messages[b_key] = agents_dict[b_key]["sent_messages"]
        
        for ob_key in agents_dict[b_key]["history_objects_carried"]:
            if ob_key not in objects_carried.keys():
                objects_carried[ob_key] = np.array(objects_dict[ob_key]["stored_locations"])
                
               
        if printing_stats:
            print("{:^20}".format(b_key), end=" ")
            
        """    
        locs = np.array(agents_dict[b_key]["locations"])
        #pdb.set_trace()
        if locs.size > 0:
            plt.plot(locs[:,0], locs[:,1])
        """
        

    lines = {}
    
    skip_frames = 10
    max_history = 1000
    texts = []
    for l_idx in range(0,len(agents_dict[b_key]["locations"]),skip_frames):
    
        if texts and l_idx - texts[0][1] > 100:
            texts[0][0].remove()
            texts.pop(0)
    
        for a_key in agents_dict.keys():

            if not l_idx:
                #pdb.set_trace()
                tmp_line, = ax.plot(locs[a_key][l_idx,0], locs[a_key][l_idx,1], color=color_assigned[a_key], label=a_key)
                plt.legend(loc="upper left")
                lines[a_key] = tmp_line
            else:
            
                s_idx = 0
                if l_idx > max_history:
                    s_idx = l_idx - max_history
            
                lines[a_key].set_xdata(locs[a_key][s_idx:l_idx,0])
                lines[a_key].set_ydata(locs[a_key][s_idx:l_idx,1])
                
            for l_sub_idx in range(l_idx-skip_frames+1,l_idx+1):
                
                if len(messages[a_key]) > l_sub_idx:
                    for m in messages[a_key][l_sub_idx]:
                        tmp_text = ax.text(locs[a_key][l_sub_idx,0], locs[a_key][l_sub_idx,1], m, fontsize = 10, bbox ={'facecolor':color_assigned[a_key], 'alpha':0.9, 'pad':1})
                        texts.append([tmp_text,l_sub_idx])
                        #print(m, locs[a_key][l_idx,0], locs[a_key][l_idx,1])
            #plt.scatter(locs[l_idx][0], locs[l_idx][1])
            #print(locs[l_idx])

        for ob_key in objects_carried.keys():
            if not l_idx:
                #pdb.set_trace()
                tmp_line, = ax.plot(objects_carried[ob_key][l_idx,0], objects_carried[ob_key][l_idx,1], color="yellow")
                lines[ob_key] = tmp_line
            else:
            
                s_idx = 0
                if l_idx > max_history:
                    s_idx = l_idx - max_history
            
                lines[ob_key].set_xdata(objects_carried[ob_key][s_idx:l_idx,0])
                lines[ob_key].set_ydata(objects_carried[ob_key][s_idx:l_idx,1])
            
                
            for dt in objects_dict[ob_key]["pickup_times"]:
                if dt <= l_idx:
                    
                    objects_dict[ob_key]["pickup_times"].pop(0)
                
                    if ob_key in objects_to_carry and objects_to_carry[ob_key]:
                        objects_to_carry[ob_key].remove()
                        objects_to_carry[ob_key] = []

                        
            for dt in objects_dict[ob_key]["drop_times"]:
                if dt <= l_idx:
                    plt.scatter(objects_carried[ob_key][l_idx,0],objects_carried[ob_key][l_idx,1], marker="x", color="red")
                    objects_dict[ob_key]["drop_times"].pop(0)
                    
                    if objects_dict[ob_key]["danger_level"] == 1:
                        color_danger = "blue" 
                    elif objects_dict[ob_key]["danger_level"] == 2:
                        color_danger = "red"
                        
                    objects_to_carry[ob_key] = plt.scatter(objects_carried[ob_key][l_idx,0],objects_carried[ob_key][l_idx,1], s=500, color=color_danger, marker="$" + str(ob_key) + "\_" + str(objects_dict[ob_key]["weight"]) + "$")


            
                    
        #plt.draw()
        #plt.pause(0.001)
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        plt.savefig("matplots/matplots_output_" + str(l_idx).zfill(5) + ".jpg")
        #time.sleep(0.0001)
        
        
            
    if printing_stats:
        print("")                 
    
    if printing_stats:
        for s in agents_dict[b_key]["stats"].keys():
            print("{:^50}".format(s),end=" ")
            for b_idx,b_key in enumerate(agents_dict.keys()):
                                    
                print("{:^20}".format(str(agents_dict[b_key]["stats"][s])), end=" ")
            print("")
    """
    for ob_key in objects_dict.keys():
        locs = np.array(objects_dict[ob_key]["stored_locations"])
        if locs.size > 0:
            plt.plot(locs[:,0], locs[:,1], color="yellow")
    """                        
    #plt.show()  
    

    
    if printing_stats:
        print("")   
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

    
def get_settings(log_state_file, target_cell_size):


    settings = log_state_file.readline()
    
    if not settings:
        return None,None,None,None
        
    settings = eval(settings.strip())
    arguments = log_state_file.readline().strip()
    scenario_options_line = log_state_file.readline()
    if scenario_options_line:
        scenario_options = eval(scenario_options_line.strip())
        
    else: #For old files
    
        wall_width = settings["cell_size"]/2
        scenario_size = 20
        
        scenario_options = {"scenario_size":scenario_size, "env_objects":[]}
        
        wall1_1 =  [{"x": 6, "y": 1}, {"x": 6, "y": 2},{"x": 6, "y": 3},{"x": 6, "y": 4}]
        wall1_2 = [{"x": 1, "y": 6},{"x": 2, "y": 6},{"x": 3, "y": 6},{"x": 4, "y": 6}]
        wall2_1 = [{"x": 14, "y": 1}, {"x": 14, "y": 2},{"x": 14, "y": 3},{"x": 14, "y": 4}]
        wall2_2 = [{"x": 19, "y": 6},{"x": 18, "y": 6},{"x": 17, "y": 6},{"x": 16, "y": 6}]
        wall3_1 = [{"x": 6, "y": 19}, {"x": 6, "y": 18},{"x": 6, "y": 17},{"x": 6, "y": 16}]
        wall3_2 = [{"x": 1, "y": 14},{"x": 2, "y": 14},{"x": 3, "y": 14},{"x": 4, "y": 14}]
        wall4_1 = [{"x": 14, "y": 19}, {"x": 14, "y": 18},{"x": 14, "y": 17},{"x": 14, "y": 16}]
        wall4_2 = [{"x": 19, "y": 14},{"x": 18, "y": 14},{"x": 17, "y": 14},{"x": 16, "y": 14}]

        scenario_options["walls"] = [[[wall[0]['x']+wall_width-scenario_size/2,wall[0]['y']+wall_width-scenario_size/2],[wall[-1]['x']+wall_width-scenario_size/2,wall[-1]['y']+wall_width-scenario_size/2]] for wall in [wall1_1,wall1_2,wall2_1,wall2_2,wall3_1,wall3_2,wall4_1,wall4_2]]

        
        
        
    
    
    wall_coords = []
    for wall in scenario_options["walls"]:
        wall_p1 = np.array(convert_to_grid_coords(wall[0][0],wall[0][1],scenario_options["scenario_size"],target_cell_size))
        wall_p2 = np.array(convert_to_grid_coords(wall[1][0],wall[1][1],scenario_options["scenario_size"],target_cell_size))
        
        num_cells = wall_p2-wall_p1

        sign = [int(num_cells[0] < 0),int(num_cells[1] < 0)]

        num_cells = np.absolute(num_cells)        

        
        
        for x_idx in range(int(num_cells[0])+1):
            for y_idx in range(int(num_cells[1])+1):
                coord = wall_p1+np.array([x_idx*(-1)**sign[0],y_idx*(-1)**sign[1]]).tolist()
                coord = [int(coord[0]),int(coord[1])]
                wall_coords.append(coord)

    for env_object in scenario_options["env_objects"]:
        env_p = convert_to_grid_coords(env_object[0],env_object[1],scenario_options["scenario_size"],target_cell_size)
        env_p = [int(env_p[0]),int(env_p[1])]

    
        wall_coords.append(env_p)
        
    return scenario_options,wall_coords,settings,arguments


def init_agents_dict(agents_dict, scenario_options):

    for r in scenario_options["robots_type"]:
        agent_id = r[0]
        if agent_id not in agents_dict:
            agents_dict[agent_id] = {"last_action":action_sample.copy(),"location":[],"last_grid_world":[], "changed":False, "object_carried":-1, "object_mapping":{}, "reward":0, "disabled":False, "object_manipulation":{"dangerous_dropped": 0, "non_dangerous_collected": 0, "dangerous_collected": 0, "dangerous_not_collected": 0}, "locations":[], "message_context":[], "sent_messages":[], "history_objects_carried":[]}

    
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

good_files = ["2023_11_16_14_46_50.txt","2024_01_24_16_42_47.txt","2024_01_25_10_00_40.txt","2024_01_30_15_47_09.txt","2024_01_18_16_48_46.txt","2024_01_19_06_37_37.txt","2024_01_30_15_07_42.txt","2024_01_26_15_19_22.txt","2024_02_01_17_29_04.txt","2024_01_26_10_27_40.txt","2024_01_23_13_34_24.txt","2024_01_26_15_20_28.txt","2024_01_18_14_03_02.txt","2024_01_23_14_33_58.txt","2024_01_04_14_41_25.txt","2024_01_15_10_02_48.txt","2024_01_10_15_39_25.txt","2024_01_23_13_33_59.txt","2024_01_25_15_15_34.txt","2024_01_03_14_40_35.txt","2023_11_03_14_51_11.txt","2024_01_15_16_19_21.txt","2024_01_22_14_47_18.txt"]

target_cell_size = 1
saved_results = []
all_agents_stats = {}


#plt.axis([-10, 10, -10, 10])  # Adjust axis limits for various functions
#plt.ion()
#plt.show()

for d in good_files: #glob.glob("*.txt"):
    #d = "2023_11_02_16_05_09.txt" #"2023_05_12_14_00_00.txt" #"2023_08_04_15_08_34.txt"
    
    #d = "2024_01_22_14_47_18.txt"
    
    if not re.match("\d{4}_\d{2}",d):
        continue
    
    print(d)
    
    log_file = open(d)
    new_line = log_file.readline()
    messages_present = False
    log_state_file = open("../../simulator/log/" + d[:-4] + "_state.txt")
    num_objects = -1
    action_sample = {"action":-1,"item":-1,"robot":-1, "message":""}
    state_sample = {"object_weight": 0, "object_danger_level": 0, "object_confidence": 0, "object_location": [-1,-1], "grid_map":[]}
    
    
    sensed_objects = {}
    team_collected_objects = 0
    individual_stats_count = 0

    scenario_options,wall_coords,settings,arguments = get_settings(log_state_file, target_cell_size)
    agents_dict = {}
    
    
    if "robots_type" in scenario_options:
        init_agents_dict(agents_dict, scenario_options)
                    
    grid_map = init_grid_map(wall_coords, scenario_options["scenario_size"]) #[[0 for y in range(scenario_options["scenario_size"])] for y in range(scenario_options["scenario_size"])]
    
    #agents_dict = {robot[0]:{"last_action":action_sample.copy(),"location":[],"last_grid_world":[], "changed":False, "object_carried":-1, "object_mapping":{}, "reward":0, "disabled":False} for robot in scenario_options["robots_type"]}
    
    objects_dict = {}
    rewarded_objects = []
    legacy_objects = {}
    
    log_memory = []
    
    legacy = False
    legacy_keys = {}
    legacy_list_carrying_objects = {}
    
    just_reset = []
    session_num = 0
    
    while new_line:
        split_line = new_line.strip().split(',')
        
        log_line_type = int(split_line[1])

        
        if log_line_type == 0:
        
            m_idx1 = new_line.index('{')-1
            m_idx2 = new_line.index('}')+2
        
            
            metadata = json.loads(json.loads(new_line[m_idx1:m_idx2]))

            
            if num_objects == -1:
                
                num_objects = sum([1 if not m[0] else 0 for m in metadata["metadata"]]) #Check with legacy

                    
                
            
            
            new_grid_map = init_grid_map(wall_coords, scenario_options["scenario_size"]) #[[0 for y in range(scenario_options["scenario_size"])] for y in range(scenario_options["scenario_size"])]
            agent_number = 0
            
            legacy_change = []
            for m in metadata["metadata"]:
            
                if isinstance(m[3],list):
                    legacy = True
            
                if (not m[0] and not legacy) or (legacy and isinstance(m[3][0],list)): #object
                
                    if legacy:
                    
                        object_id = int(m[3][0][0])
       
                        
                        real_x,real_y = convert_to_real_coords(m[0],m[1],scenario_options["scenario_size"],settings["cell_size"])
                        coordinate_x,coordinate_y = convert_to_grid_coords(real_x,real_y,scenario_options["scenario_size"],target_cell_size)


                        weight = m[3][0][1]
                        danger_level = -1
                    else:
                        object_id = int(m[1])
                        real_x = m[2]
                        real_y = m[3]
                        weight = m[4]
                        danger_level = m[5] 
                        coordinate_x,coordinate_y = convert_to_grid_coords(real_x,real_y,scenario_options["scenario_size"],target_cell_size)
                
                    if object_id not in objects_dict:
                        objects_dict[object_id] = {"real_location":[],"location":[], "weight":weight, "danger_level":danger_level, "stored_locations":[], "drop_times":[], "pickup_times":[]}
                
                    objects_dict[object_id]["stored_locations"].append([real_x,real_y])
                    
                    new_grid_map[int(coordinate_x)][int(coordinate_y)] = 2
                    
             
                    
                    if objects_dict[object_id]["location"]:
                        if objects_dict[object_id]["location"][-1] != [coordinate_x,coordinate_y]:
                            objects_dict[object_id]["location"].append([coordinate_x,coordinate_y])
                            objects_dict[object_id]["real_location"].append([real_x,real_y])
                    else:
                        objects_dict[object_id]["location"].append([coordinate_x,coordinate_y])
                        objects_dict[object_id]["real_location"].append([real_x,real_y])
                        
                    if len(objects_dict[object_id]["location"]) > 2:
                        objects_dict[object_id]["location"].pop(0)
                        objects_dict[object_id]["real_location"].pop(0)
                    
                else: #agent
                
                    agent_number += 1
                    
                
                    if not legacy:
                        disabled = m[4]
                        
                        if disabled:
                            continue
                        
                    if legacy:
                        agent_id = m[3][0]
                    else:
                        agent_id = m[1]
                        
                    #if d == "2023_07_26_11_32_05.txt":
                    #    agent_id = scenario_options["robots_type"][agent_number-1][0]
                        
                    if agent_id not in agents_dict:
                        init_agents_dict(agents_dict,{"robots_type":[[agent_id]]})
                        #agents_dict[agent_id] = {"last_action":action_sample.copy(),"location":[],"last_grid_world":[], "changed":False, "object_carried":-1, "object_mapping":{}, "reward":0, "disabled":False, "object_manipulation":{"dangerous_dropped": 0, "non_dangerous_collected": 0, "dangerous_collected": 0, "dangerous_not_collected": 0}}
                    
                    agents_dict[agent_id]["last_action"] = action_sample.copy()
                    agents_dict[agent_id]["reward"] = 0
                    agents_dict[agent_id]["locations"].append([m[2],m[3]])
                    agents_dict[agent_id]["sent_messages"].append([])
                    if legacy:
                        x_orig,y_orig = convert_to_real_coords(m[0],m[1],scenario_options["scenario_size"],settings["cell_size"])
                    else:
                        x_orig = m[2]
                        y_orig = m[3]
                    

                    coordinate_x,coordinate_y = convert_to_grid_coords(x_orig,y_orig,scenario_options["scenario_size"],target_cell_size)
                    
                    if len(m) > 7:
                        strength = m[7]
                    else: #To allow for legacy format
                        strength = 0
                        new_agent_number = 0
                        for new_m in metadata["metadata"]:
                            if new_m[0]:
                            
                                if legacy:
                                    x_new_orig,y_new_orig = convert_to_real_coords(new_m[0],new_m[1],scenario_options["scenario_size"],settings["cell_size"])
                                else:
                                    x_new_orig = new_m[2]
                                    y_new_orig = new_m[3]
                                    
                                new_agent_number += 1
                                other_agent_id = new_m[1]
                                #if d == "2023_07_26_11_32_05.txt":
                                #    other_agent_id = scenario_options["robots_type"][new_agent_number-1][0]
                                if other_agent_id != agent_id and np.linalg.norm(np.array([x_orig,y_orig])-np.array([x_new_orig,y_new_orig])) < settings['strength_distance_limit']:
                                    strength += 1
                    
                    if not legacy:
                        carrying_objects = [int(m[5]),int(m[6])]
                    else:
                        if agent_id not in legacy_list_carrying_objects:
                            legacy_list_carrying_objects[agent_id] = [-1,-1]
                        carrying_objects = legacy_list_carrying_objects[agent_id]
                            
                    carrying_objects_bool = any(True if ob >= 0 else False for ob in carrying_objects)
                    
                    if agent_id == 'A':
                        new_grid_map[int(coordinate_x)][int(coordinate_y)] = 5
                    else:
                        new_grid_map[int(coordinate_x)][int(coordinate_y)] = 3
                    
                    
                    if carrying_objects_bool: #Don't show the object being carried
                        for c_object_id in carrying_objects:
                            if c_object_id > -1:
                                object_location = objects_dict[c_object_id]["location"][-1]

                                if new_grid_map[int(object_location[0])][int(object_location[1])] == 2:
                                    new_grid_map[int(object_location[0])][int(object_location[1])] = 0
                        
                    if agents_dict[agent_id]["object_carried"] == -1 and carrying_objects_bool:


                        if carrying_objects[0] > -1:
                            object_id = carrying_objects[0]
                        else:
                            object_id = carrying_objects[1]
                            
                        agents_dict[agent_id]["object_carried"] = object_id
                        objects_dict[object_id]["pickup_times"].append(len(agents_dict[agent_id]["locations"]))
                        
                        if object_id not in agents_dict[agent_id]["history_objects_carried"]:
                            agents_dict[agent_id]["history_objects_carried"].append(object_id)
                        
                        if agents_dict[agent_id]["location"][-1] != objects_dict[object_id]["location"][0]:
                            agent_location = agents_dict[agent_id]["location"][-1]
                        else:
                            agent_location = agents_dict[agent_id]["location"][0]
                        

                        agents_dict[agent_id]["last_action"]["action"] = determine_direction(agent_location,objects_dict[object_id]["location"][0],True)
                        agents_dict[agent_id]["changed"] = True
                        
                    
                    elif agents_dict[agent_id]["object_carried"] > -1 and agents_dict[agent_id]["object_carried"] not in carrying_objects:
                        object_id = agents_dict[agent_id]["object_carried"]
                        #print(object_id,agent_id)
                        

                        #If object is dropped consciously
                        if (strength >= objects_dict[object_id]["weight"] and not legacy) or (legacy and "dropped" in legacy_keys[agent_id]):
                            agents_dict[agent_id]["last_action"]["action"] = Action.drop_object.value
                            
                            if legacy:

                                d_idx = legacy_keys[agent_id].index("dropped")
                                legacy_keys[agent_id][d_idx] = ""
                            
                            if object_id not in rewarded_objects and (objects_dict[object_id]["danger_level"] == 2 or objects_dict[object_id]["danger_level"] == -1) and np.linalg.norm(objects_dict[object_id]["real_location"][-1]) < settings['goal_radius']: #Give reward of 1 if in goal area. We assume at first all legacy objects are dangerous until the end were we check for each case
                                agents_dict[agent_id]["reward"] = 1
                                rewarded_objects.append(object_id)
                                
                                if objects_dict[object_id]["danger_level"] == -1: #This is for legacy
                                    legacy_change.append(agent_id)
                                    
                                    if agent_id not in legacy_objects:
                                        legacy_objects[agent_id] = []
                                        
                                    legacy_objects[agent_id].append({"index":-1,"object_id":object_id})
                                    
                                #agents_dict[agent_id]["object_manipulation"]["dangerous_collected"] += 1
                                
                            #elif objects_dict[object_id]["danger_level"] == 1 and np.linalg.norm(objects_dict[object_id]["real_location"][-1]) < settings['goal_radius']:
                            #    agents_dict[agent_id]["object_manipulation"]["non_dangerous_collected"] += 1
                           

                        elif not legacy: #Object was dropped unintentionally
                            agents_dict[agent_id]["object_manipulation"]["dangerous_dropped"] += 1
                               
                                        
                                     
                            
                        agents_dict[agent_id]["changed"] = True
                        agents_dict[agent_id]["object_carried"] = -1
                        objects_dict[object_id]["drop_times"].append(len(agents_dict[agent_id]["locations"]))
                            
                    
                    if agents_dict[agent_id]["location"]:
                        if agents_dict[agent_id]["location"][-1] != [coordinate_x,coordinate_y]:

                            action_result = determine_direction(agents_dict[agent_id]["location"][-1],[coordinate_x,coordinate_y],False)
                            
                            if action_result == -1: #If a movement skips through cells, restart
                                agents_dict[agent_id]["location"] = [[coordinate_x,coordinate_y]]
                                if agents_dict[agent_id]["last_action"]["action"] == -1:
                                    agents_dict[agent_id]["last_action"]["action"] = Action.wait.value
                                continue
                                
                            agents_dict[agent_id]["last_action"]["action"] = action_result
                       
                            agents_dict[agent_id]["location"].append([coordinate_x,coordinate_y])
                            agents_dict[agent_id]["changed"] = True

                    else:
                        agents_dict[agent_id]["location"].append([coordinate_x,coordinate_y])
                        #agents_dict[agent_id]["changed"] = True
                        
                    if len(agents_dict[agent_id]["location"]) > 2:
                        agents_dict[agent_id]["location"].pop(0)
                    
                    #If nothing happened, wait
                    if agents_dict[agent_id]["last_action"]["action"] == -1:
                        agents_dict[agent_id]["last_action"]["action"] = Action.wait.value
                    
            for a_key in agents_dict.keys():
                if agents_dict[a_key]["changed"]:
                    agents_dict[a_key]["changed"] = False
                    
                    if a_key in legacy_change:
                        legacy_objects[a_key][-1]["index"] = len(log_memory)
                    
                    current_state = state_sample.copy()
                    next_state = state_sample.copy()
                    current_state["grid_map"] = grid_map
                    next_state["grid_map"] = new_grid_map
                    object_carried = agents_dict[a_key]["object_carried"] > -1
                    log_memory.append([a_key,current_state,agents_dict[a_key]["last_action"],next_state,agents_dict[a_key]["reward"],agents_dict[a_key]["message_context"]])
                    agents_dict[a_key]["message_context"] = []
                    #print(a_key,grid_map,Action(agents_dict[a_key]["last_action"]),new_grid_map,0)
                    
                    
                    """
                    if a_key == "A":
                        print(a_key,Action(agents_dict[a_key]["last_action"]["action"]), "REWARD", agents_dict[a_key]["reward"], "CARRYING", object_carried)

                        #print_map(np.array(grid_map))
                        print_map(np.array(new_grid_map))
                    """

            grid_map = new_grid_map
            

            

                    
            
                    
        
        elif log_line_type == 1: #Keys pressed
            a_key = split_line[2]
            key = split_line[3]

            if key == "Q" or key == "B":
                agent_action = action_sample.copy()
                agent_action["action"] = Action.danger_sensing.value
                
                current_state = state_sample.copy()
                current_state["grid_map"] = grid_map
                
                log_memory.append([a_key,current_state,agent_action,current_state,0,agents_dict[a_key]["message_context"]])
                agents_dict[a_key]["message_context"] = []
                #print(Action.danger_sensing)
                #agents_dict[a_key]["activate_sensor"] = True
                
                if a_key in legacy_keys:
                    for arm_idx in range(len(legacy_keys[a_key])):
                        if legacy_keys[a_key][arm_idx] == "pick_up":
                            legacy_keys[a_key][arm_idx] = ""
            elif key == "N" or key == "E":
                if a_key in legacy_keys:
                    for arm_idx in range(len(legacy_keys[a_key])):
                        if legacy_keys[a_key][arm_idx] == "pick_up":
                            legacy_keys[a_key][arm_idx] = ""
            elif key == "Z" or key == "A":
                if legacy:
                    if a_key not in legacy_keys:
                        legacy_keys[a_key] = ["",""]
                        
                    if legacy_keys[a_key][0] == "carrying":
                        legacy_keys[a_key][0] = "dropping"
                    else:
                        legacy_keys[a_key][0] = "pick_up"
            elif key == "X" or key == "D":
                if legacy:
                    if a_key not in legacy_keys:
                        legacy_keys[a_key] = ["",""]
                    if legacy_keys[a_key][1] == "carrying":
                        legacy_keys[a_key][1] = "dropping"

                    else:
                        legacy_keys[a_key][1] = "pick_up"
                    

        
            
        elif log_line_type == 2: #messages sent
            if not messages_present:
                messages_present = True
                session_num += 1
            
            
            last_q = new_line.rfind('"')
            second_last_q = new_line[:last_q].rfind('"')
            third_last_q = new_line[:second_last_q].rfind('"')
            first_q = new_line.index('"')
            
            strings_extract = [first_q+1,third_last_q,second_last_q+1,last_q]
            
            agent_action = action_sample.copy()
            agent_action["action"] = Action.send_message.value
            sent_message = new_line[strings_extract[0]:strings_extract[1]]
            agent_action["message"] = sent_message
            a_key = split_line[2]
            
            agents_dict[a_key]["sent_messages"].append([sent_message])
            
            if new_line[strings_extract[2]:strings_extract[3]]: #get recipients of message
                recipients = new_line[strings_extract[2]:strings_extract[3]].split(',')[:-1]
                
                for r in recipients:
                    agents_dict[r]["message_context"].append([a_key,agent_action["message"]])

                
            
            current_state = state_sample.copy()
            current_state["grid_map"] = grid_map



            log_memory.append([a_key,current_state,agent_action,current_state,0,agents_dict[a_key]["message_context"]])
            agents_dict[a_key]["message_context"] = []
            #print(agent_action)
                
        elif log_line_type == 3 and agents_dict: #objects and teammates
            a_key = split_line[2]
            
            if legacy:
                m_idx1 = new_line.index('{')
                m_idx2 = new_line.rfind('{')-1
            else:
                m_idx1 = new_line.index('{')
                m_idx2 = new_line.rfind('}')+1

            object_results = json.loads(new_line[m_idx1:m_idx2])
            agent_action = action_sample.copy()
            agent_action["action"] = Action.check_item.value
            
            current_state = state_sample.copy()
            next_state = state_sample.copy()
            current_state["grid_map"] = grid_map
            next_state["grid_map"] = grid_map
            
            legacy_carried_objects = [False,False]
            
            if legacy: #Check if an object was dropped
                if a_key in legacy_list_carrying_objects:
                    for ob_idx,ob_key in enumerate(legacy_list_carrying_objects[a_key]):
                        if ob_key > -1 and str(ob_key) not in object_results.keys() and legacy_keys[a_key][ob_idx]:
                            if legacy_keys[a_key][ob_idx] == "dropping":
                                legacy_keys[a_key][ob_idx] = "dropped"
                            else: #Object was dropped accidentaly
                                legacy_keys[a_key][ob_idx] = ""
                                agents_dict[a_key]["object_manipulation"]["dangerous_dropped"] += 1
                                
                            #print(ob_key,a_key, legacy_keys[a_key][ob_idx], "going to drop")

                            
                            legacy_list_carrying_objects[a_key][ob_idx] = -1
            
            for ob_key in object_results.keys():
            
            
                new_measurement = False
            
                if legacy: #Check whether an object was picked up
                        
                    if a_key in legacy_keys:
                        for arm_idx in range(len(legacy_keys[a_key])):
                            if legacy_keys[a_key][arm_idx] == "pick_up" and int(ob_key) not in legacy_list_carrying_objects[a_key]:
                                #print(ob_key,a_key, legacy_keys[a_key][arm_idx], "going to carry")
                                legacy_keys[a_key][arm_idx] = "carrying"
                                legacy_list_carrying_objects[a_key][arm_idx] = int(ob_key)
                                
            
                weight = object_results[ob_key]["weight"]
                time_object = object_results[ob_key]["time"]
                location = object_results[ob_key]["location"]

                if object_results[ob_key]["sensor"] and a_key in object_results[ob_key]["sensor"]:   
                
                    danger_level = object_results[ob_key]["sensor"][a_key]["value"]
                    confidence = object_results[ob_key]["sensor"][a_key]["confidence"]

                

                if int(ob_key) not in agents_dict[a_key]["object_mapping"].keys():
                    new_idx = len(agents_dict[a_key]["object_mapping"].keys())
                    agents_dict[a_key]["object_mapping"][int(ob_key)] = {"index":new_idx,"time":time_object,"location":location}
                    new_measurement = True
                else:
                    if agents_dict[a_key]["object_mapping"][int(ob_key)]["time"] < time_object and location[0] != agents_dict[a_key]["object_mapping"][int(ob_key)]["location"][0] and location[1] != agents_dict[a_key]["object_mapping"][int(ob_key)]["location"][1]:
                        agents_dict[a_key]["object_mapping"][int(ob_key)]["time"] = time_object
                        agents_dict[a_key]["object_mapping"][int(ob_key)]["location"] = location
                        new_measurement = True
                
                if new_measurement:
                    agent_action["item"] = agents_dict[a_key]["object_mapping"][int(ob_key)]["index"]
                    next_state["object_weight"] = weight
                    next_state["location"] = location
                    
                    if object_results[ob_key]["sensor"] and a_key in object_results[ob_key]["sensor"]:
                        next_state["object_danger_level"] = danger_level
                        next_state["object_confidence"] = confidence
                    
                    log_memory.append([a_key,current_state,agent_action,next_state,0,agents_dict[a_key]["message_context"]])
                    agents_dict[a_key]["message_context"] = []
                    #print(agent_action, weight,danger_level,confidence,location)
                    
                    current_state = next_state
                    
                    
                    if ob_key not in sensed_objects.keys(): #For score calculation
                        sensed_objects[ob_key] = []                        
                    if a_key not in sensed_objects[ob_key]:
                        sensed_objects[ob_key].append(a_key)
                
        elif log_line_type == 4: #Reset
            
            log_line_agent = split_line[2]
            
            just_reset.append(log_line_agent)
            
            #print(just_reset)
            if all(robot[0] in just_reset for robot in scenario_options["robots_type"] if robot[1] == "human"): #Check robot is real
            
                print("Resetting")
                scenario_options_tmp,wall_coords_tmp,settings_tmp,arguments_tmp = get_settings(log_state_file, target_cell_size)
                    
                if scenario_options_tmp:
                    scenario_options = scenario_options_tmp
                    wall_coords = wall_coords_tmp
                    settings = settings_tmp
                    arguments = arguments_tmp
                    
                grid_map = init_grid_map(wall_coords, scenario_options["scenario_size"])
                agents_dict = {}
                if "robots_type" in scenario_options:
                    init_agents_dict(agents_dict, scenario_options)
                objects_dict = {}
                rewarded_objects = []
                legacy_objects = {}
                legacy_keys = {}
                legacy_list_carrying_objects = {}
                log_memory = []
                individual_stats_count = 0
                
                just_reset = [] #multiple resets
            
        elif log_line_type == 5: #statistics
            a_key = split_line[2]

            #print(a_key)
            if '{' in new_line: #else it is a legacy format
                m_idx1 = new_line.index('{')
                m_idx2 = new_line.rfind('}')+1

                stats = json.loads(new_line[m_idx1:m_idx2])
                
                #print("finish", a_key, individual_stats_count)
                
                if "total_dangerous_objects" in stats and stats["total_dangerous_objects"]:
                
                    
                    individual_stats_count += 1
                    
                    """
                    individual_effort = 0
                    objects_sensed_by_agent = []
                    total_objects = 20

                    for s in sensed_objects.keys():
                        if int(s) >= 20:
                            total_objects = 25
                        
                        if a_key in sensed_objects[s]:
                        
                            individual_effort += 1/len(sensed_objects[s])
                            
                            
                    individual_effort /= total_objects
                    

                    individual_quality = len(stats["dangerous_objects_in_goal"]) - (len(stats["objects_in_goal"])-len(stats["dangerous_objects_in_goal"])) - agents_dict[a_key]["object_manipulation"]["dangerous_dropped"] # - (stats["total_dangerous_objects"]-len(stats["dangerous_objects_in_goal"]))
                    
                    individual_quality /= stats["total_dangerous_objects"]
                    
                    individual_quality = max(0, individual_quality)
                    
                    team_collected_objects += len(stats["objects_in_goal"])
                    
                    agents_dict[a_key]["effort"] = individual_effort
                    agents_dict[a_key]["quality"] = individual_quality
                    """
                    
                    
                    agents_dict[a_key]["effort"] = stats["effort"]
                    agents_dict[a_key]["quality"] = stats["quality_work"]
                    agents_dict[a_key]["payment"] = stats["individual_payment"]
                    agents_dict[a_key]["stats"] = stats.copy()
                                       
                
                if individual_stats_count == len(agents_dict.keys()):
                
                    """
                    team_objects_dropped = sum([agents_dict[a_key]["object_manipulation"]["dangerous_dropped"] for a_key in agents_dict.keys()])
                
                    team_quality = stats["team_objects_in_goal"] - (team_collected_objects-stats["team_objects_in_goal"]) - team_objects_dropped # - (stats["total_dangerous_objects"]-stats["team_objects_in_goal"])
                    

                    
                    team_quality /= stats["total_dangerous_objects"]
                    
                    team_quality = max(0, team_quality)
                    
                    maxPayment = 7*len(agents_dict.keys())
                    
                    totalEffort = sum([agents_dict[a_key]["effort"] for a_key in agents_dict.keys()])
                    
                    actualPayment = maxPayment * (team_quality + totalEffort) / 2
                    
                    if team_quality + totalEffort == 0:
                        continue
                    
                    acc_individual_payment = 0
                    for b_idx,b_key in enumerate(agents_dict.keys()):
                    
                            

                        
                        individual_contribution = (agents_dict[b_key]["quality"] + agents_dict[b_key]["effort"])/(team_quality + totalEffort)
                        individual_payment = actualPayment * individual_contribution
                        acc_individual_payment += individual_payment
                        print(b_key, "Effort:", agents_dict[b_key]["effort"], "Quality:", agents_dict[b_key]["quality"], "Payment:", individual_payment)
                        
                    
                    print("Team Quality:", team_quality, "Dangerous objects in goal:", stats["team_objects_in_goal"], "Total objects in goal:", team_collected_objects, "Objects dropped:", team_objects_dropped, "Total dangerous objects:", stats["total_dangerous_objects"])   
                    print("Total Effort:", totalEffort)
                    print('Maximum Payment:', maxPayment, "Total Payment:", acc_individual_payment)
                    """
                    
                    #for b_idx,b_key in enumerate(agents_dict.keys()):
                    #    print(b_key, "Effort:", agents_dict[b_key]["effort"], "Quality:", agents_dict[b_key]["quality"], "Payment:", agents_dict[a_key]["payment"])
                    #    #print(b_key, 
                    
                    
                    objects_in_goal = []
                    total_distance_traveled = 0
                    total_messages_exchanged = 0
                    total_pickup_times = 0
                    total_dropped_times = 0
                    
                    time_in_company = {}
                    
                    lifter_role = {}
                    sensing_role = {}
                    
                    
                    
                    if not ("token" in stats and not stats["token"]):  
                    
                              
                        for b_idx,b_key in enumerate(agents_dict.keys()):        
                            agents_dict[b_key]["stats"]["quality_choosing"] = 0
                            agents_dict[b_key]["stats"]["quality_moving"] = 0
                            agents_dict[b_key]["stats"]["quality_communication"] = 0
                            agents_dict[b_key]["stats"]["quality_carrying"] = 0
                            agents_dict[b_key]["stats"]["task_completion"] = len(agents_dict[b_key]["stats"]["dangerous_objects_in_goal"]) / agents_dict[b_key]["stats"]["total_dangerous_objects"]
                            agents_dict[b_key]["stats"]["team_completion"] = agents_dict[b_key]["stats"]["team_dangerous_objects_in_goal"]/agents_dict[b_key]["stats"]["total_dangerous_objects"]
                            agents_dict[b_key]["stats"]["team_quality_choosing"] = 0
                            agents_dict[b_key]["stats"]["team_quality_moving"] = float("inf")
                            agents_dict[b_key]["stats"]["team_quality_carrying"] = float("inf")
                            agents_dict[b_key]["stats"]["team_quality_communication"] = float("inf")
                            agents_dict[b_key]["stats"]["team_task_interdependence"] = 0
                            agents_dict[b_key]["stats"]["team_role_specialization"] = 0
                            
                            
                            if not (agents_dict[b_key]["stats"]["grabbed_objects"] == 0 and agents_dict[b_key]["stats"]["objects_sensed"] == 0):
                                lifter_role[b_key] = agents_dict[b_key]["stats"]["grabbed_objects"]
                                sensing_role[b_key] = agents_dict[b_key]["stats"]["objects_sensed"]
                            
                            for c_key in agents_dict[b_key]["stats"]["time_with_teammates"].keys():
                                if c_key + "_" + b_key not in time_in_company.keys() and b_key + "_" + c_key not in time_in_company.keys():
                                    time_in_company[c_key + "_" + b_key] = agents_dict[b_key]["stats"]["time_with_teammates"][c_key]
                            
                            
                            total_distance_traveled += agents_dict[b_key]["stats"]["distance_traveled"]
                            total_messages_exchanged += agents_dict[b_key]["stats"]["num_messages_sent"]
                            if len(agents_dict[b_key]["stats"]["objects_in_goal"]) > 0:
                            
                                for ob in agents_dict[b_key]["stats"]["objects_in_goal"]:
                                    if ob not in objects_in_goal:
                                        objects_in_goal.append(ob)
                            
                                agents_dict[b_key]["stats"]["quality_choosing"] = len(agents_dict[b_key]["stats"]["dangerous_objects_in_goal"])/len(agents_dict[b_key]["stats"]["objects_in_goal"])
                                agents_dict[b_key]["stats"]["quality_moving"] = agents_dict[b_key]["stats"]["distance_traveled"]/len(agents_dict[b_key]["stats"]["objects_in_goal"])
                            
                            if len(agents_dict[b_key]["stats"]["dangerous_objects_in_goal"]) > 0:
                                agents_dict[b_key]["stats"]["quality_communication"] = agents_dict[b_key]["stats"]["num_messages_sent"]/len(agents_dict[b_key]["stats"]["dangerous_objects_in_goal"])
                            
                            pickup_times = len(agents_dict[b_key]["stats"]["dropped_outside_goal"]) + len(agents_dict[b_key]["stats"]["objects_in_goal"])
                            total_pickup_times += pickup_times
                            total_dropped_times += len(agents_dict[b_key]["stats"]["dropped_outside_goal"])
                            if pickup_times > 0:
                                agents_dict[b_key]["stats"]["quality_carrying"] = len(agents_dict[b_key]["stats"]["dropped_outside_goal"])/pickup_times
             
                        time_in_company_avg = np.average(list(time_in_company.values()))/stats["team_end_time"]
             
                        """
                        n = len(lifter_role.values())
                        out_diff = np.abs(np.sum(np.array(list(lifter_role.values())) * np.arange(n-1, -n, -2) ) / (n*(n-1) / 2)) #Mean difference
                        
                        sum_diff = np.sum(list(lifter_role.values()))
                        if sum_diff > 0:
                            lifter_diff = out_diff/sum_diff
                        else:
                            lifter_diff = 0
                        
                        
                        out_diff = np.abs(np.sum(np.array(list(sensing_role.values())) * np.arange(n-1, -n, -2) ) / (n*(n-1) / 2))
                        sum_diff = np.sum(list(sensing_role.values()))
                        if sum_diff > 0:
                            sensing_diff = out_diff/sum_diff
                        else:
                            sensing_diff = 0
                        """
                        if list(sensing_role.values()) and max(list(lifter_role.values())) and max(list(sensing_role.values())):
                            role_specialization = ((max(list(lifter_role.values())) - np.average(list(lifter_role.values())))/max(list(lifter_role.values())) + (max(list(sensing_role.values())) - np.average(list(sensing_role.values())))/max(list(sensing_role.values())))/2#(sensing_diff + lifter_diff)/2
                        else:
                            role_specialization = 0
                 
             
                        agents_stats = {}
                        for b_idx,b_key in enumerate(agents_dict.keys()):
                        
                            agents_dict[b_key]["stats"]["team_objects_in_goal"] = objects_in_goal
                            agents_dict[b_key]["stats"]["total_distance_traveled"] = total_distance_traveled
                            if total_distance_traveled > 0:
                                agents_dict[b_key]["stats"]["productivity"] = agents_dict[b_key]["stats"]["team_dangerous_objects_in_goal"]/total_distance_traveled
                            else:
                                agents_dict[b_key]["stats"]["productivity"] = 0
                            
                            agents_dict[b_key]["stats"]["team_other_quality_work"] = (agents_dict[b_key]["stats"]["team_dangerous_objects_in_goal"]-(len(objects_in_goal)-agents_dict[b_key]["stats"]["team_dangerous_objects_in_goal"]))/agents_dict[b_key]["stats"]["total_dangerous_objects"]
                            
                            
                            agents_dict[b_key]["stats"]["team_task_interdependence"] = time_in_company_avg
                            agents_dict[b_key]["stats"]["team_role_specialization"] = role_specialization
                            
                            if objects_in_goal:    
                                agents_dict[b_key]["stats"]["team_quality_choosing"] = agents_dict[b_key]["stats"]["team_dangerous_objects_in_goal"]/len(objects_in_goal)
                                
                                agents_dict[b_key]["stats"]["team_quality_moving"] = total_distance_traveled/len(objects_in_goal)
                                
                            if agents_dict[b_key]["stats"]["team_dangerous_objects_in_goal"] > 0:
                                agents_dict[b_key]["stats"]["team_quality_communication"] = total_messages_exchanged/agents_dict[b_key]["stats"]["team_dangerous_objects_in_goal"]
                            
                            if total_pickup_times > 0:    
                                agents_dict[b_key]["stats"]["team_quality_carrying"] = total_dropped_times/total_pickup_times
                                
                            agents_stats[b_key] = agents_dict[b_key]["stats"]
                            
                            
                        if d not in agents_stats:
                            all_agents_stats[d] = {}
                        all_agents_stats[d][session_num] = agents_stats.copy()
                                    
                        action_stats = {}
                        num_actions = 27
                        for lm in log_memory:
                            if lm[0] not in action_stats.keys():
                                action_stats[lm[0]] = [0]*num_actions
                            action_stats[lm[0]][lm[2]["action"]] += 1
                   
                        
                        print_session = False
                        """
                        if len(list(agents_dict.keys())) > 2:
                            for object_id in objects_in_goal:
                                if objects_dict[int(object_id)]["weight"] > 1:
                                    print_session = True
                                    break
                                
                        if print_session:
                            print_results(scenario_options, agents_dict, objects_dict)  
                            quit()  
                        """
                        
                        if agents_dict[b_key]["stats"]["team_end_time"] > 900 and agents_dict[b_key]["stats"]["total_distance_traveled"] > 0 and all(so[1] == "human" for so in scenario_options["robots_type"]): #No tutorials
                            saved_results.append([scenario_options.copy(),agents_dict.copy(), objects_dict.copy(), action_stats.copy(), d])
                        
                        
                        
                        #print("Team Quality:", stats["team_quality_work"], "Dangerous objects in goal:", stats["team_dangerous_objects_in_goal"], "Total dangerous objects:", stats["total_dangerous_objects"])   
                        #print("Total Effort:", stats["human_team_effort"])
                        #print("Total Team Payment:", stats["team_payment"])
                        
                        
                    
                if a_key in legacy_objects:
                    for lo in legacy_objects[a_key]: #Rewrite log file for legacy cases were we only know the dangerous objects until the end
            
                    
                        if lo["object_id"] not in stats["dangerous_objects_in_goal"]:
                            log_memory[lo["index"]][4] = 0
   
        
        new_line = log_file.readline()
    
    #break

"""
scores = {"team_score":{2:[],3:[],4:[]},"frac_dangerous":{2:[],3:[],4:[]},"frac_collected":{2:[],3:[],4:[]},"sensor_activations":{2:[],3:[],4:[]},"distance_traveled":{2:[],3:[],4:[]},"productivity":{2:[],3:[],4:[]}}
for s in saved_results:
    if len(s[1].keys()) >= 2:
        a_key = list(s[1].keys())[0]
        
        scores["team_score"][len(s[1].keys())].append(s[1][a_key]["stats"]["team_other_quality_work"])
        if s[1][a_key]["stats"]["total_dangerous_objects"]:
            scores["frac_dangerous"][len(s[1].keys())].append(s[1][a_key]["stats"]["team_dangerous_objects_in_goal"]/s[1][a_key]["stats"]["total_dangerous_objects"])
        else:
            scores["frac_dangerous"][len(s[1].keys())].append(0)
            
        if s[1][a_key]["stats"]["team_objects_in_goal"]:
            scores["frac_collected"][len(s[1].keys())].append(s[1][a_key]["stats"]["team_dangerous_objects_in_goal"]/len(s[1][a_key]["stats"]["team_objects_in_goal"]))
        else:
            scores["frac_collected"][len(s[1].keys())].append(0)
            
        scores["sensor_activations"][len(s[1].keys())].append(s[1][a_key]["stats"]["sensor_activation"])
        scores["distance_traveled"][len(s[1].keys())].append(s[1][a_key]["stats"]["total_distance_traveled"])
        scores["productivity"][len(s[1].keys())].append(s[1][a_key]["stats"]["team_dangerous_objects_in_goal"]/s[1][a_key]["stats"]["total_distance_traveled"])


print(scores)
for t_key in scores["team_score"].keys():
    print(t_key, len(scores["team_score"][t_key]))

for s_key in scores.keys():
    for t_key in scores[s_key].keys():
            
        print(s_key, t_key, np.average(scores[s_key][t_key]))
"""


'''
best_team_metrics = [0,0,float("inf"),float("inf"),float("inf")]
best_team = [[],[],[],[],[]]
colors = list(mcolors.TABLEAU_COLORS.keys())
legends_labels = []
series = {}
for s in saved_results:
    if len(s[1].keys()) == 4:
        for k in s[1].keys():
            team_completion = s[1][k]["stats"]["team_completion"]
            team_quality_choosing = s[1][k]["stats"]["team_quality_choosing"]
            team_quality_moving = s[1][k]["stats"]["team_quality_moving"]
            team_quality_carrying = s[1][k]["stats"]["team_quality_carrying"]
            team_quality_communication = s[1][k]["stats"]["team_quality_communication"]
            team_interdependece = s[1][k]["stats"]["team_task_interdependence"]
            team_other_quality = s[1][k]["stats"]["team_other_quality_work"]
            productivity = s[1][k]["stats"]["productivity"]
            break
        
        label = str(len(s[1].keys()))
        if label not in legends_labels:
            legends_labels.append(label)
            
            series[label] = {"x":[],"y":[]}
            
            
        series[label]["x"].append(team_interdependece)
        series[label]["y"].append(productivity)
        
        #plt.scatter(team_interdependece, productivity, color=colors[len(s[1].keys())], label=label)
        
        print(team_interdependece, productivity, s[4])
        if team_interdependece > best_team_metrics[0]:
            best_team_metrics[0] = team_interdependece
            best_team[0] = s
        if team_interdependece < best_team_metrics[3]:
            best_team_metrics[3] = 1-team_interdependece
            best_team[3] = s
        """
        if team_completion > best_team_metrics[0]:
            best_team_metrics[0] = team_completion
            best_team[0] = s
        if team_quality_choosing > best_team_metrics[1]:
            best_team_metrics[1] = team_quality_choosing
            best_team[1] = s
        if team_quality_moving < best_team_metrics[2]:
            best_team_metrics[2] = team_quality_moving
            best_team[2] = s
        if team_quality_carrying < best_team_metrics[3]:
            best_team_metrics[3] = 1-team_quality_carrying
            best_team[3] = s
        if team_quality_communication < best_team_metrics[4]:
            best_team_metrics[4] = team_quality_communication
            best_team[4] = s
        """
        
"""
for s_key in series.keys():
    indices = np.argsort(series[s_key]["x"])
    x = []
    y = []
    for i in indices:
        x.append(series[s_key]["x"][i])
        y.append(series[s_key]["y"][i])
        
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    
    plt.plot(x, p(x), color=colors[int(s_key)])


plt.xlabel("Task Interdependence", fontsize=20)
plt.ylabel("Productivity (objects/m)", fontsize=20)
plt.ylim([0, 0.02])
plt.axhline(y=0.0193, color='r', linestyle='--')
plt.show()  
"""     
print_results(best_team[3][0], best_team[3][1], best_team[3][2])
#for b in best_team:
#    print_results(b[0], b[1], b[2])
'''
out_file = open('agents_stats.json', "w") 
json.dump(all_agents_stats, out_file)
