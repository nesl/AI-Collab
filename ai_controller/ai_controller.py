import gym_collab
import gymnasium as gym
import time
import argparse
from collections import defaultdict
import numpy as np
import pdb
import sys
import random


from gym_collab.envs.action import Action

from llm_control import LLMControl
from deepq_control import DeepQControl
from heuristic_control import HeuristicControl

parser = argparse.ArgumentParser(
    description="WebRTC audio / video / data-channels demo"
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
parser.add_argument("--control", default="heuristic", type=str, help="Type of control to apply: heuristic,llm,openai,deepq,q,manual")
parser.add_argument("--message-loop", action="store_true", help="Use to allow messages to be sent back to sender")
parser.add_argument("--role", default="general", help="Choose a role for the agent: general, scout, lifter")
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

env = gym.make('gym_collab/AICollabWorld-v0', use_occupancy=args.use_occupancy, view_radius=args.view_radius, skip_frames=10, client_number=int(args.robot_number), host=args.host, port=args.port, address=args.address, cert_file=args.cert_file, key_file=args.key_file)



#processed_observation = (tuple(map(tuple, observation['frame'])), bool(observation['objects_held']))

#print_map(observation["frame"])

next_observation = []

actions_to_take = [*[1]*2,*[2]*3,11,19,*[3]*4,*[0]*5,16,19]



class RobotState:
    def __init__(self, latest_map, object_held, num_robots):
        self.latest_map = latest_map
        self.object_held = object_held
        self.items = []
        self.robots = [{} for n in range(num_robots)]
        self.strength = 1
        self.map_metadata = {}
        
    def update_items(self,item_output, item_idx): #Updates items

        print(item_output)
        if not self.items[item_idx]["item_danger_level"] or  (item_output["item_danger_level"] and round(self.items[item_idx]["item_danger_confidence"][0],3) == round(item_output["item_danger_confidence"][0],3) and self.items[item_idx]["item_time"][0] < item_output["item_time"][0]) or (item_output["item_danger_level"] and self.items[item_idx]["item_danger_confidence"][0] < item_output["item_danger_confidence"][0]):
            
            self.items[item_idx] = item_output
            
        elif self.items[item_idx]["item_time"][0] < item_output["item_time"][0]:
        
            self.items[item_idx]["item_location"] = item_output["item_location"]
            self.items[item_idx]["item_time"] = item_output["item_time"]
            
        
        
'''
reward_machine_state = 0
action["action"] = agents[reward_machine_state].get_action(processed_observation)

process_reward = 0
process_last_action = action["action"]
reward_machine_state = 0
'''

num_steps = 3000 #600 #200#600

num_episodes = 600

process_reward = 0

# Get number of actions from gym action space

# Get the number of state observations
observation, info = env.reset()


if args.control == 'heuristic':
    h_control = HeuristicControl(env.goal_coords, num_steps, env.robot_id, env, args.role)
    print("ROLE:", args.role)
elif args.control == 'deepq':
    deepq_control = DeepQControl(observation,device,num_steps)



while True:

    observation, info = env.reset()
    
    if args.message_loop:
        num_robots = env.action_space["robot"].n
    else:
        num_robots = env.action_space["robot"].n-1
    
    robotState = RobotState(observation['frame'].copy(), 0, num_robots)
    #observation, reward, terminated, truncated, info = env.step(17)
    done = False

    action_issued = [False,False]
    last_action = [0,0]

    messages = []
    
    action = env.action_space.sample()
    action["action"] = Action.get_occupancy_map.value #actions_to_take.pop(0)

    action['num_cells_move'] = 1

    high_level_action_finished = True

    action_function = ""
    function_output = []

    step_count = 0
    
    

    if args.control == 'llm' or args.control == 'openai':
        obs_sample = env.observation_space.sample()

        room_size = str(obs_sample['frame'].shape[0])

        llm_control = LLMControl(args.control == 'openai',room_size, env.action_space.sample(), device)
    elif args.control == 'heuristic':
        h_control.start()
    elif args.control == 'deepq':
        action["action"] = deepq_control.start(observation)

    last_high_action = action["action"]

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
        
        
            
        
        next_observation, reward, terminated, truncated, info = env.step(action)


        if args.message_loop:
            info["robot_key_to_index"][env.robot_id] = len(robotState.robots)-1
        
        
        if reward != 0:
            print('Reward', reward)
            process_reward += reward
            
        #print(next_observation["num_items"])
            
        if next_observation["num_items"] > len(robotState.items):
            diff_len = next_observation["num_items"] - len(robotState.items)
            robotState.items.extend([{'item_weight': 0, 'item_danger_level': 0, 'item_danger_confidence': np.array([0.]), 'item_location': np.array([-1, -1], dtype=np.int16), 'item_time': np.array([0], dtype=np.int16)} for d in range(diff_len)])
            
        robotState.strength = next_observation["strength"]
            
        #When any action has completed
        if next_observation and any(next_observation['action_status']):
            
            
            
            
            ego_location = np.where(next_observation['frame'] == 5)
            previous_ego_location = np.where(robotState.latest_map == 5)
            robotState.latest_map[previous_ego_location[0][0],previous_ego_location[1][0]] = 0
            

            
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
                            if isinstance(map_object, list): #Object information

                                ob_key = info["object_key_to_index"][map_object[0]]
                                
                                
                                robotState.items[ob_key]["item_location"] = [int(m_key_xy[0]), int(m_key_xy[1])]

                                
                            elif map_object not in info['map_metadata'][str(ego_location[0][0])+'_'+str(ego_location[1][0])]: #Robot information
                                
                                ob_key = info["robot_key_to_index"][map_object]
                                robotState.robots[ob_key]["neighbor_location"] = [int(m_key_xy[0]), int(m_key_xy[1])]

                                
                                
                            
                elif Action(last_action[1]) == Action.get_objects_held:

                    robotState.object_held = next_observation['objects_held']
            
                elif Action(last_action[1]) == Action.check_item:
                
                    #robotState.items[last_action_arguments[0]] = next_observation["item_output"]
                    robotState.update_items(next_observation["item_output"],last_action_arguments[0])
                    
                elif Action(last_action[1]) == Action.check_robot: #Make sure to update estimates and take the one with the highest confidence
                
                    item_idx = last_action_arguments[1]-1
                    robotState.robots[item_idx] = next_observation["neighbors_output"]
                        

                        

                    
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
            
            if high_level_action_finished: #When a high level action finishes, we sense the environment
                if last_action[1] == Action.get_messages.value: #Action.get_occupancy_map.value:
                
                    print_map(robotState.latest_map)
                    print("Held:",robotState.object_held)
                
                    last_action[1] = 0 #Reset last sensing action
                    step_count += 1
                    
                    #print("Messages", messages)
                    if args.control == 'llm' or args.control == 'openai':
                        action_function = llm_control.control(messages, robotState, action_function, function_output)
                        high_level_action_finished = False
                    elif args.control == 'heuristic':
                        #action["action"] = h_control.planner(robotState, process_reward, step_count, terminated or truncated)
                        

                        action["action"],action["item"],action["message"],action["robot"] = h_control.planner_sensing(robotState, process_reward, step_count, terminated or truncated, next_observation, info, messages)

                        
                        print("STEP", step_count, action["action"])
                        
                        if action["action"] < 0:
                            break
                        
                    elif args.control == 'deepq':
                    
                        last_high_action = action["action"]
                        action["action"] = deepq_control.control(reward, terminated, truncated, robotState, action, step_count, ego_location)

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
                 
                        

            
            if not high_level_action_finished:
            
                
                if args.control == 'llm' or args.control == 'openai':
                    action, high_level_action_finished,function_output = eval(action_function) #go_to_location(x,y, action_sequence, robotState, next_observation)
                
                #print(function_output, high_level_action_finished)
                
            
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



print("Closing environment")
env.close()



