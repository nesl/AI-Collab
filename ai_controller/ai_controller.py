import gym_collab
import gymnasium as gym
import time
import argparse
from collections import defaultdict
import numpy as np
import pdb
import sys
import random
import cv2
import yaml


from magnebot import ActionStatus
from gym_collab.envs.action import Action

from llm_control import LLMControl
from deepq_control import DeepQControl
from heuristic_control import HeuristicControl
from tutorial_control import TutorialControl
from decision_control import DecisionControl

parser = argparse.ArgumentParser(
    description="AI Controller"
)
parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
parser.add_argument(
    "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
)
parser.add_argument(
    "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
)
parser.add_argument("--record-to", help="Write received media to a file."),
parser.add_argument("--verbose", "-v", action="count")
parser.add_argument("--use-occupancy", action='store_true', help="Use occupancy maps instead of images")
parser.add_argument("--address", default='https://172.17.15.69:4000', help="Address where our simulation is running")
parser.add_argument("--robot-number", default=1, help="Robot number to control")
parser.add_argument("--view-radius", default=0, help="When using occupancy maps, the view radius")
parser.add_argument("--control", default="heuristic", type=str, help="Type of control to apply: heuristic,llm,openai,deepq,q,manual,decision")
parser.add_argument("--message-loop", action="store_true", help="Use to allow messages to be sent back to sender")
parser.add_argument("--role", default="general", help="Choose a role for the agent: general, scout, lifter")
parser.add_argument("--planning", default="equal", help="Choose a planning role for the agent: equal, coordinator, coordinated")
parser.add_argument('--webcam', action="store_true", help="Use images from virtual webcam")
parser.add_argument('--video-index', type=int, default=0, help='index of the first /dev/video device to capture frames from')
parser.add_argument('--config', type=str, default='team_structure.yaml', help='Path to team structure configuration file')
#parser.add_argument("--openai", action='store_true', help="Use openai.")
#parser.add_argument("--llm", action='store_true', help="Use LLM.")

args = parser.parse_args()


import torch


'''        
# hyperparameters
learning_rate = 0.01
n_episodes = 100
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1


num_reward_states = 2

agents = []

for n in range(num_reward_states):
    agents.append(QLearningAgent(
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        max_action_number=25, #19,
        final_epsilon=final_epsilon,
    ))
'''

'''
ACTIONS

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

wait = 26

Action space
    
                "action": spaces.Discrete(len(Action)),
                "item": spaces.Discrete(self.map_config['num_objects']),
                # Allow for 0
                "robot": spaces.Discrete(len(self.map_config['all_robots']) + 1),
                "message" : spaces.Text(min_length=0,max_length=100),
                "num_cells_move": spaces.Discrete(map_size), #ignore
            }




Observation space
    {
                "frame": spaces.Box(low=-2, high=5, shape=(map_size, map_size), dtype=np.int16),
                "objects_held": spaces.Discrete(3, start=-1),
                "action_status": spaces.MultiDiscrete(np.array([2] * 4), dtype=np.int16),

                "item_output": spaces.Dict(
                    {
                        "item_weight": spaces.Discrete(len(self.map_config['all_robots'])+1),
                        "item_danger_level": spaces.Discrete(3),
                        "item_danger_confidence": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
                        "item_location": spaces.Box(low=-np.infty, high=np.infty, shape=(2,), dtype=np.int16),
                        "item_time": spaces.Box(low=0, high=np.infty, shape=(1,), dtype=np.int16)
                    }
                ),
                "num_items": spaces.Discrete(self.map_config['num_objects'] + 1),

                "neighbors_output": spaces.Dict(
                    {
                        "neighbor_type": spaces.Discrete(3, start=-1),
                        "neighbor_location": spaces.Box(low=-np.infty, high=np.infty, shape=(2,), dtype=np.int16)
                    }

                ),
                # Strength starts from zero
                "strength": spaces.Discrete(len(self.map_config['all_robots']) + 2),
                "num_messages": spaces.Discrete(100)

            }

'''





def print_map(occupancy_map): #Occupancy maps require special printing so that the orientation is correct
    new_occupancy_map = occupancy_map.copy()
    for row_id in range(occupancy_map.shape[0]):
        new_occupancy_map[row_id,:] = occupancy_map[occupancy_map.shape[0]-row_id-1,:]

    new_new_occupancy_map = new_occupancy_map.copy()
    for row_id in range(occupancy_map.shape[1]): 
        new_new_occupancy_map[:,row_id] = new_occupancy_map[:,occupancy_map.shape[1]-row_id-1]
    print(new_new_occupancy_map)


                        

device = "cuda"

env = gym.make('gym_collab/AICollabWorld-v0', use_occupancy=args.use_occupancy, view_radius=args.view_radius, skip_frames=10, client_number=int(args.robot_number), host=args.host, port=args.port, address=args.address, cert_file=args.cert_file, key_file=args.key_file, webcam=args.webcam, video_index=args.video_index+1)



#processed_observation = (tuple(map(tuple, observation['frame'])), bool(observation['objects_held']))

#print_map(observation["frame"])

next_observation = []

actions_to_take = [*[1]*2,*[2]*3,11,19,*[3]*4,*[0]*5,16,19]



class RobotState:
    def __init__(self, latest_map, object_held, env):
        self.latest_map = latest_map
        self.object_held = object_held
        self.items = []
        self.item_estimates = {}
        self.robots = [{"neighbor_type": env.neighbors_info[n][1], "neighbor_location": [-1,-1], "neighbor_time": [0.0], "neighbor_disabled": -1} for n in range(len(env.neighbors_info))] # 0 if human, 1 if ai
        self.strength = 1
        self.map_metadata = {}
        self.sensor_parameters = env.sensor_parameters
        self.neighbors_sensor_parameters = env.neighbors_sensor_parameters
        self.possible_estimates = {}
        
    def average_fusion(self, item_idx):
    
        benign = 0
        dangerous = 0
        num_samples = 0
        for ie in self.item_estimates[item_idx]:
            if ie["item_danger_level"] == 2:
                benign += 1-ie["item_danger_confidence"]
                dangerous += ie["item_danger_confidence"]
                num_samples += 1
            elif ie["item_danger_level"] == 1:
                benign += ie["item_danger_confidence"]
                dangerous += 1-ie["item_danger_confidence"]
                num_samples += 1
                
        if num_samples:
            benign /= num_samples
            dangerous /= num_samples
            
            if dangerous > benign:
                self.items[item_idx]["item_danger_level"] = 2
                self.items[item_idx]["item_danger_confidence"] = [dangerous]
            else:
                self.items[item_idx]["item_danger_level"] = 1
                self.items[item_idx]["item_danger_confidence"] = [benign]
                
    def bayesian_fusion(self, item_idx):
    
        prior_benign = 0.5#0.7
        prior_dangerous = 0.5#0.3
        samples = False
        
        self.possible_estimates[item_idx] = {}
    
        for ie_idx,ie in enumerate(self.item_estimates[item_idx]):
        
            if ie["item_danger_level"]:
            
                samples = True
                
                if ie_idx == len(self.item_estimates[item_idx])-1:
                    if ie["item_danger_level"] == 2:
                        benign = 1-self.sensor_parameters[0]
                        dangerous = self.sensor_parameters[1]
                    elif ie["item_danger_level"] == 1:
                        benign = self.sensor_parameters[0]
                        dangerous = 1-self.sensor_parameters[1]
                        
                else:
                    if ie["item_danger_level"] == 2:
                        benign = 1-self.neighbors_sensor_parameters[ie_idx][0]
                        dangerous = self.neighbors_sensor_parameters[ie_idx][1]
                    elif ie["item_danger_level"] == 1:
                        benign = self.neighbors_sensor_parameters[ie_idx][0]
                        dangerous = 1-self.neighbors_sensor_parameters[ie_idx][1]
            
                prob_evidence = (prior_benign*benign + prior_dangerous*dangerous)
                
                prior_benign = benign*prior_benign/prob_evidence
                prior_dangerous = dangerous*prior_dangerous/prob_evidence
                
                
        for ie_idx,ie in enumerate(self.item_estimates[item_idx]):
        
            if not ie["item_danger_level"]:
            
                
                for item_danger_level in [1,2]:
                
                    if ie_idx == len(self.item_estimates[item_idx])-1:
                        if item_danger_level == 2:
                            benign = 1-self.sensor_parameters[0]
                            dangerous = self.sensor_parameters[1]
                        elif item_danger_level == 1:
                            benign = self.sensor_parameters[0]
                            dangerous = 1-self.sensor_parameters[1]
                            
                    else:
                        if item_danger_level == 2:
                            benign = 1-self.neighbors_sensor_parameters[ie_idx][0]
                            dangerous = self.neighbors_sensor_parameters[ie_idx][1]
                        elif item_danger_level == 1:
                            benign = self.neighbors_sensor_parameters[ie_idx][0]
                            dangerous = 1-self.neighbors_sensor_parameters[ie_idx][1]
                
                    prob_evidence = (prior_benign*benign + prior_dangerous*dangerous)
                    
                    prior_benign_temp = benign*prior_benign/prob_evidence
                    prior_dangerous_temp = dangerous*prior_dangerous/prob_evidence
                
                    prior_list = [prior_benign_temp,prior_dangerous_temp]
                    dangerous_level = np.argmax(prior_list)
                    
                    if ie_idx not in self.possible_estimates[item_idx]:
                        self.possible_estimates[item_idx][ie_idx] = []
                    
                    self.possible_estimates[item_idx][ie_idx].append(prior_list[dangerous_level])
                
                
                
                
                    
                
        if samples:
            prior_list = [prior_benign,prior_dangerous]
            dangerous_level = np.argmax(prior_list)
            self.items[item_idx]["item_danger_level"] = dangerous_level + 1
            self.items[item_idx]["item_danger_confidence"] = [prior_list[dangerous_level]]
        
    def update_robots(self, neighbor_output, robot_idx):
    
        if neighbor_output["neighbor_type"] >= 0:
            self.robots[robot_idx]["neighbor_type"] = neighbor_output["neighbor_type"]
    
        if neighbor_output["neighbor_disabled"] >= 0:
            self.robots[robot_idx]["neighbor_disabled"] = neighbor_output["neighbor_disabled"]
    
        if self.robots[robot_idx]["neighbor_time"][0] <= neighbor_output["neighbor_time"][0]:
        
            self.robots[robot_idx]["neighbor_location"] = [int(neighbor_output["neighbor_location"][0]),int(neighbor_output["neighbor_location"][1])]
            self.robots[robot_idx]["neighbor_time"] = neighbor_output["neighbor_time"]
            
            if self.latest_map[self.robots[robot_idx]["neighbor_location"][0], self.robots[robot_idx]["neighbor_location"][1]] != 5:
                self.latest_map[self.robots[robot_idx]["neighbor_location"][0], self.robots[robot_idx]["neighbor_location"][1]] = 3
    
    def update_items(self,item_output, item_idx, robot_idx): #Updates items

        information_change = False
        #We save estimates from all robots
        if item_idx >= len(self.items):
            
            diff_len = item_idx+1 - len(self.items)
            print("item_change", item_idx, len(self.items), diff_len)
            self.items.extend([{'item_weight': 0, 'item_danger_level': 0, 'item_danger_confidence': np.array([0.]), 'item_location': np.array([-1, -1], dtype=np.int16), 'item_time': np.array([0], dtype=np.int16)} for d in range(diff_len)])
            information_change = True

        if item_idx not in self.item_estimates:
            self.item_estimates[item_idx] = [{"item_danger_level": 0, "item_danger_confidence": 0, "item_location": [-1,-1], "item_time": 0} for n in range(len(self.robots)+1)]
            information_change = True
            
        self.item_estimates[item_idx][robot_idx]["item_location"] = [int(item_output["item_location"][0]),int(item_output["item_location"][1])]
        self.item_estimates[item_idx][robot_idx]["item_time"] = item_output["item_time"]
        
        if item_output["item_danger_level"]:
            self.item_estimates[item_idx][robot_idx]["item_danger_level"] = item_output["item_danger_level"]
            self.item_estimates[item_idx][robot_idx]["item_danger_confidence"] = item_output["item_danger_confidence"][0]
            information_change = True


            
        """
        if not self.items[item_idx]["item_danger_level"] or  (item_output["item_danger_level"] and round(self.items[item_idx]["item_danger_confidence"][0],3) == round(item_output["item_danger_confidence"][0],3) and self.items[item_idx]["item_time"][0] < item_output["item_time"][0]) or (item_output["item_danger_level"] and self.items[item_idx]["item_danger_confidence"][0] < item_output["item_danger_confidence"][0]):
            
            self.items[item_idx] = item_output
            self.items[item_idx]["item_location"] = [int(item_output["item_location"][0]),int(item_output["item_location"][1])]
        """
           
        try: 
            if not self.items[item_idx]["item_danger_level"]:
                self.items[item_idx] = item_output
                self.items[item_idx]["item_location"] = [int(item_output["item_location"][0]),int(item_output["item_location"][1])]
        except:
            pdb.set_trace()
        if self.items[item_idx]["item_time"][0] <= item_output["item_time"][0]:
        
            self.items[item_idx]["item_location"] = [int(item_output["item_location"][0]),int(item_output["item_location"][1])]
            self.items[item_idx]["item_time"] = item_output["item_time"]
            
            if self.latest_map[self.items[item_idx]["item_location"][0], self.items[item_idx]["item_location"][1]] != 5 and self.latest_map[self.items[item_idx]["item_location"][0], self.items[item_idx]["item_location"][1]] != 3 and self.latest_map[self.items[item_idx]["item_location"][0], self.items[item_idx]["item_location"][1]] != 4:
                self.latest_map[self.items[item_idx]["item_location"][0], self.items[item_idx]["item_location"][1]] = 2
                #print("changing object location")
            
        if item_output["item_weight"]:
            self.items[item_idx]["item_weight"] = item_output["item_weight"]
        
        
        
        if information_change:
            print(item_output)
            #self.average_fusion(item_idx)
            self.bayesian_fusion(item_idx)
        
        
'''
reward_machine_state = 0
action["action"] = agents[reward_machine_state].get_action(processed_observation)

process_reward = 0
process_last_action = action["action"]
reward_machine_state = 0
'''

num_steps = 3000 #600 #200#600

num_episodes = 600



# Get number of actions from gym action space

# Get the number of state observations

team_structure = {}
with open(args.config, 'r') as file:
    team_structure = yaml.safe_load(file)
    

just_starting = True

while True:

    observation, info = env.reset()
    
    if just_starting: #Initialized only once
        if args.control == 'heuristic':
            h_control = HeuristicControl(env.goal_coords, num_steps, env.robot_id, env, args.role, args.planning)
            print("ROLE:", args.role, "PLANNING:", args.planning)
        elif args.control == 'deepq':
            deepq_control = DeepQControl(observation,device,num_steps)
        elif args.control == 'tutorial':
            t_control = TutorialControl(num_steps, env.robot_id, env)
        just_starting = False
    
    
    print(env.neighbors_info)
    robotState = RobotState(observation['frame'].copy(), 0, env)
    #observation, reward, terminated, truncated, info = env.step(17)
    done = False

    action_issued = [False,False]
    last_action = [0,0]

    messages = []
    message_queue = []
    
    action = env.action_space.sample()
    action["action"] = Action.get_occupancy_map.value #actions_to_take.pop(0)

    action['num_cells_move'] = 1

    high_level_action_finished = True

    action_function = ""
    function_output = []

    step_count = 0
    
    process_reward = 0
    
    disabled = False
    

    if args.control == 'llm' or args.control == 'openai':
        obs_sample = env.observation_space.sample()

        llm_control = LLMControl(args.control == 'openai',env, device, robotState)
    elif args.control == "decision":
        decision_control = DecisionControl(env, robotState, team_structure)
    elif args.control == 'heuristic':
        h_control.start()
    elif args.control == 'deepq':
        action["action"] = deepq_control.start(observation)
    elif args.control == 'tutorial':
        t_control.start(robotState)
        print("INITIALIAZED")

    last_high_action = action["action"]
    
    fell_down = 0

    while not done:

        #action = env.action_space.sample()
        
        #Make sure to issue concurrent actions but not of the same type. Else, wait.
        if action["action"] < Action.danger_sensing.value and not action_issued[0]:
            action_issued[0] = True
            last_action[0] = action["action"]
            #print("Locomotion", Action(action["action"]))
        elif action["action"] != Action.wait.value and action["action"] >= Action.danger_sensing.value and not action_issued[1]:
            last_action_arguments = [action["item"],action["robot"],action["message"]]
            action_issued[1] = True
            last_action[1] = action["action"]
            
            #print("Sensing", Action(action["action"]))
        else:
            action["action"] = Action.wait.value


        #action = agent.get_action(processed_observation)
        
        
            
        #print(action, next_observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        

        if args.webcam:
            pass
            #cv2.imwrite("frame.jpg",info["frame"])
            #cv2.waitKey(100) 


        if args.message_loop:
            info["robot_key_to_index"][env.robot_id] = len(robotState.robots)-1
        
        
        if reward != 0:
            print('Reward', reward)
            process_reward += reward
            
        #print(next_observation["num_items"])
        
        if info["status"] == ActionStatus.tipping:
            fell_down += 1
            if fell_down >= 1000:
                print("FELL DOWN")
                done = True
        else:
            fell_down = 0
            
        if next_observation["num_items"] > len(robotState.items):
            diff_len = next_observation["num_items"] - len(robotState.items)
            robotState.items.extend([{'item_weight': 0, 'item_danger_level': 0, 'item_danger_confidence': np.array([0.]), 'item_location': np.array([-1, -1], dtype=np.int16), 'item_time': np.array([0], dtype=np.int16)} for d in range(diff_len)])
            
        robotState.strength = next_observation["strength"]
            
        #When any action has completed
        if next_observation and any(next_observation['action_status']) and not disabled:
            
            
            
            ego_location = np.where(next_observation['frame'] == 5)
            previous_ego_location = np.where(robotState.latest_map == 5)
            
            robotState.latest_map[previous_ego_location[0][0],previous_ego_location[1][0]] = 0 #If there was an agent there it will eliminate it from the map
            
            for ob_key in range(len(robotState.robots)): #If the agent is not where it was last seen, mark it
                robo_location = robotState.latest_map[robotState.robots[ob_key]["neighbor_location"][0],robotState.robots[ob_key]["neighbor_location"][1]]
                if robotState.robots[ob_key]["neighbor_location"][0] != -1 and robotState.robots[ob_key]["neighbor_location"][1] != -1 and previous_ego_location[0][0] == robotState.robots[ob_key]["neighbor_location"][0] and previous_ego_location[1][0] == robotState.robots[ob_key]["neighbor_location"][1]:
                    robotState.latest_map[previous_ego_location[0][0],previous_ego_location[1][0]] = 3
                    
                if robotState.robots[ob_key]["neighbor_disabled"] == 1 and robotState.robots[ob_key]["neighbor_location"][0] != -1 and robotState.robots[ob_key]["neighbor_location"][1] != -1:
                    robotState.latest_map[robotState.robots[ob_key]["neighbor_location"][0],robotState.robots[ob_key]["neighbor_location"][1]] = 1
            
            

            
            if next_observation['action_status'][2]: #If sensing action was succesful
                if Action(last_action[1]) == Action.get_occupancy_map: #Maintain the state of the occupancy map and update it whenever needed
                
                    view_radius = int(args.view_radius)
                    
                    max_x = ego_location[0][0] + view_radius
                    max_y = ego_location[1][0] + view_radius
                    min_x = max(ego_location[0][0] - view_radius, 0)
                    min_y = max(ego_location[1][0] - view_radius, 0)
                    robotState.latest_map[min_x:max_x+1,min_y:max_y+1]= next_observation["frame"][min_x:max_x+1,min_y:max_y+1]
                    
                    robotState.map_metadata = info['map_metadata']

                    for m_key in info['map_metadata'].keys(): #We get the ids of objects/robots present in the current view and update locations

                        for map_object in info['map_metadata'][m_key]:
                            m_key_xy = m_key.split('_')
                            if not map_object[0]: #Object information


                                ob_key = info["object_key_to_index"][map_object[1]]
                             
                                template_item_info = {'item_weight': 0, 'item_danger_level': 0, 'item_danger_confidence': np.array([0.]), 'item_location': np.array([int(m_key_xy[0]), int(m_key_xy[1])], dtype=np.int16), 'item_time': np.array([info["time"]], dtype=np.int16)}
                                robotState.update_items(template_item_info, ob_key, -1)
                                #robotState.items[ob_key]["item_location"] = [int(m_key_xy[0]), int(m_key_xy[1])]
                    

                                
                            elif map_object[1] != env.robot_id: #map_object not in info['map_metadata'][str(ego_location[0][0])+'_'+str(ego_location[1][0])]: #Robot information
                                
                                robot_idx = info["robot_key_to_index"][map_object[1]]
                                
                                template_robot_info = {"neighbor_type": -1, "neighbor_location": np.array([int(m_key_xy[0]), int(m_key_xy[1])], dtype=np.int16), "neighbor_time": np.array([info["time"]], dtype=np.int16), "neighbor_disabled": map_object[2]}
                                print("Disabled:", map_object)
                                robotState.update_robots(template_robot_info, robot_idx)

                                
                                
                            
                elif Action(last_action[1]) == Action.get_objects_held:

                    robotState.object_held = next_observation['objects_held']
            
                elif Action(last_action[1]) == Action.check_item:
                
                    #robotState.items[last_action_arguments[0]] = next_observation["item_output"]
                    robotState.update_items(next_observation["item_output"],last_action_arguments[0], -1)
                    
                elif Action(last_action[1]) == Action.check_robot: #Make sure to update estimates and take the one with the highest confidence
                
                    robot_idx = last_action_arguments[1]-1
       
                    robotState.update_robots(next_observation["neighbors_output"], robot_idx)
                        

                        

                    
                elif Action(last_action[1]) == Action.get_messages:
                    #print("Message arrived", info['messages'])
                    messages = info['messages']
                    
                
            '''
            #Fixed set of actions
            if next_observation['action_status'][0] or next_observation['action_status'][2]:
                action["action"] = actions_to_take.pop(0)
            elif next_observation['action_status'][1]:
                action["action"] = last_action[0]
            elif next_observation['action_status'][3]:
                action["action"] = last_action[1]
            '''
            

            #action["action"] = int(input("Next action > "))
            
            
            robotState.latest_map[ego_location[0][0],ego_location[1][0]] = 5 #Set ego robot in map
            
            
            for ob_key in range(len(robotState.robots)): #If the agent is not where it was last seen, mark it
                robo_location = robotState.latest_map[robotState.robots[ob_key]["neighbor_location"][0],robotState.robots[ob_key]["neighbor_location"][1]]
                if robo_location != 5 and robo_location != 3 and robotState.robots[ob_key]["neighbor_location"][0] != -1 and robotState.robots[ob_key]["neighbor_location"][1] != -1 and robotState.robots[ob_key]["neighbor_disabled"] != 1:
                    #pdb.set_trace()
                    print("ROBOT NOT FOUND", ob_key, robotState.robots[ob_key]["neighbor_location"])
                    robotState.robots[ob_key]["neighbor_location"] = [-1,-1]
                    
                    #robotState.robots[ob_key]["neighbor_time"] = info["time"]
                    
            for ob_key in range(len(robotState.items)): #If the agent is not where it was last seen, mark it
                try:
                    item_location = robotState.latest_map[robotState.items[ob_key]["item_location"][0],robotState.items[ob_key]["item_location"][1]]
                    if item_location == 0:
                        robotState.items[ob_key]["item_location"] = [-1,-1]
                except:
                    pdb.set_trace()
            	
            
            
            if high_level_action_finished: #When a high level action finishes, we sense the environment
                if last_action[1] == Action.get_messages.value: #Action.get_occupancy_map.value:
                
                    print_map(robotState.latest_map)
                    print("Held:",robotState.object_held)
                
                    last_action[1] = 0 #Reset last sensing action
                    step_count += 1
                    
                    #print("Messages", messages)
                    if args.control == 'llm' or args.control == 'openai':
                        action,terminated_tmp = llm_control.control(messages, robotState, info, next_observation)
                        
                        if terminated_tmp:
                            disabled = True
                            env.sio.emit("disable")
                        #high_level_action_finished = False
                        
                    elif args.control == "decision":
                        action,terminated_tmp = decision_control.control(messages, robotState, info, next_observation)
                        
                        if action["action"] == Action.send_message.value and "message_ai" in action:
                            if action["message_ai"]:
                                if action["message"]:
                                    high_level_action_finished = False
                                    message_queue.append(action["message_ai"])
                                else:
                                    action["message"] = action["message_ai"]
                                    action["robot"] = -1
                            
                            del action["message_ai"]
                        
                        if terminated or truncated:
                            break
                        
                        if terminated_tmp:
                            disabled = True
                            env.sio.emit("disable")
                            
                            
                        
                        
                    elif args.control == 'heuristic':
                        #action["action"] = h_control.planner(robotState, process_reward, step_count, terminated or truncated)
                        

                        action["action"],action["item"],action["message"],action["robot"],terminated_tmp = h_control.planner_sensing(robotState, process_reward, step_count, terminated or truncated, next_observation, info, messages)

                        
                        if terminated_tmp:
                            disabled = True
                            env.sio.emit("disable")
                        
                        print("STEP", step_count, action["action"])
                        
                        if action["action"] < 0:
                            break
                        
                    elif args.control == 'deepq':
                    
                        last_high_action = action["action"]
                        action["action"] = deepq_control.control(reward, terminated, truncated, robotState, action, step_count, ego_location)

                        if action["action"] < 0:
                            break
                            
                    elif args.control == 'tutorial':
                    
                    
                        action["action"],action["message"],terminated_tmp = t_control.planner_sensing(robotState, process_reward, step_count, terminated or truncated, next_observation, info, messages)
                        action["robot"] = 0
                        
                        if action["action"] < 0:
                            break
                            
                    elif args.control == 'manual':
                        print("Messages", messages)
                        print("Total reward", process_reward)
                        action["action"] = int(input(">> "))
                        
                    messages = []
                    process_reward = 0

                    
                    
                    

                    
                elif last_action[1] == Action.get_occupancy_map.value:
                    action["action"] = Action.get_objects_held.value

                elif last_action[1] == Action.get_objects_held.value:
                    action["action"] = Action.get_messages.value

                else:       

                    action["action"] = Action.get_occupancy_map.value
                 
            elif message_queue:
                action["message"] = message_queue.pop(0)
                action["action"] = Action.send_message.value
                action["robot"] = -1
                
                if not message_queue:
                    high_level_action_finished = True
                            

            '''
            if not high_level_action_finished:
            
                
                if args.control == 'llm' or args.control == 'openai':
                    action, high_level_action_finished,function_output = eval(action_function) #go_to_location(x,y, action_sequence, robotState, next_observation)
                
                #print(function_output, high_level_action_finished)
            '''    
            
            #print_map(robotState.latest_map)
            #print("Held:",robotState.object_held)
            
         
            #print(next_observation['item_output'], next_observation['objects_held'], next_observation['neighbors_output'], next_observation['strength'], next_observation['num_messages'], next_observation['num_items'], next_observation['action_status'], last_action)
            #print("Object held", robotState.object_held)

            if any(next_observation['action_status'][:2]):
                action_issued[0] = False
            if any(next_observation['action_status'][2:4]):
                action_issued[1] = False
                
            '''
            #For Q Learning
            processed_next_observation = (tuple(map(tuple, robotState.latest_map)), robotState.object_held)
        

            agents[reward_machine_state].update(processed_observation, process_last_action, process_reward, terminated, processed_next_observation)
            
            action["action"] = agents[reward_machine_state].get_action(processed_next_observation)
            process_last_action = action["action"]
            
            processed_observation = processed_next_observation
            
            
            #TODO object_held not working
            if reward_machine_state == 0 and robotState.object_held: #Transition to next state in reward machine
                print("Change to state 1")

                reward_machine_state = 1
            elif reward_machine_state == 1 and not robotState.object_held:
                print("Change to state 0")
                reward_machine_state = 0
            
            process_reward = 0
            '''

        #processed_observation = processed_next_observation

        if terminated or truncated:
            done = True
            
            objects_to_report = []
            for ob_key in range(len(robotState.items)):
                if robotState.items[ob_key]["item_weight"] >= len(env.map_config['all_robots'])+1 and robotState.items[ob_key]["item_danger_level"] == 2:
                    object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(ob_key)]
                    objects_to_report.append(object_id)

            if objects_to_report:
                print("Reporting", objects_to_report)
                env.sio.emit("report", (objects_to_report))
                    
                
            



print("Closing environment")
env.close()



