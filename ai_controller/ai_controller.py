import gym_collab
import gymnasium as gym
import time
import argparse
from collections import defaultdict
import numpy as np
import pdb
import sys
import random
import torch
import re
import openai
import os
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from gym_collab.envs.action import Action

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
parser.add_argument("--openai", action='store_true', help="Use openai.")

args = parser.parse_args()


class QLearningAgent:
    def __init__(
        self,
        learning_rate,
        initial_epsilon,
        epsilon_decay,
        final_epsilon,
        max_action_number,
        discount_factor = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(max_action_number+1))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
        
        self.max_action_number = max_action_number

    def get_action(self, obs):
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return random.randint(0, self.max_action_number) #env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs,
        action,
        reward,
        terminated,
        next_obs,
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)
        
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
    {
        "action" : spaces.Discrete(len(self.Action)),
        "item" : spaces.Discrete(self.map_config['num_objects']),
        "robot" : spaces.Discrete(len(self.map_config['all_robots'])+1), #Allow for 0
        "message" : spaces.Text(min_length=0,max_length=100)
    }




Observation space
    {
        "frame" : spaces.Box(low=0, high=5, shape=(map_size, map_size), dtype=int),
        "objects_held" : spaces.Discrete(2),
        "action_status" : spaces.MultiDiscrete([2]*4)
        "item_output" : spaces.Dict(
            {
                "item_weight" : spaces.Discrete(10),
                "item_danger_level" : spaces.Discrete(3),
                "item_location" : spaces.MultiDiscrete([map_size, map_size])
            }
        ),
        "num_items" : spaces.Discrete(self.map_config['num_objects']),
        "neighbors_output" : spaces.Dict(
            {
                "neighbor_type" : spaces.Discrete(2),
                "neighbor_location" : spaces.MultiDiscrete([map_size, map_size])
            }
        
        ),
        "strength" : spaces.Discrete(len(self.map_config['all_robots'])+1), #Strength starts from zero
        "num_messages" : spaces.Discrete(100)
        
        #"objects_danger_level" : spaces.Box(low=1,high=2,shape=(self.map_config['num_objects'],), dtype=int)
    }

'''

def calculateHValue(current,dest):

    dx = abs(current[0] - dest[0])
    dy = abs(current[1] - dest[1])
    
    D = 1
    D2 = np.sqrt(2)
 
    h = D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
    
    h = dx + dy

    return h


def tracePath(node_details,dest):
    path = []
    
    currentNode = dest

    while node_details[currentNode[0]][currentNode[1]]["parent"][0] != currentNode[0] or node_details[currentNode[0]][currentNode[1]]["parent"][1] != currentNode[1]:
        path.append(currentNode)
        currentNode = node_details[currentNode[0]][currentNode[1]]["parent"]
        
    path.reverse()
    
    return path
        

def findPath(startNode,endNode,occMap):


    if min(endNode) == -1 or any(endNode >= occMap.shape) or (endNode[0] == startNode[0] and endNode[1] == startNode[1]):
        return []
        
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
    

    
    
    
    next_nodes = np.array([[-1,0],[1,0],[0,1],[0,-1]]) #np.array([[1,1],[-1,1],[1,-1],[-1,-1],[-1,0],[1,0],[0,1],[0,-1]])
    
    while openSet:
    
        currentNode = openSet.pop(0)
        closedSet.append(tuple(currentNode))
        
 
            
        for nx in next_nodes:
            neighborNode = currentNode + nx
            
            if min(neighborNode) == -1 or any(neighborNode >= occMap.shape) or occMap[neighborNode[0],neighborNode[1]] != 0 or tuple(neighborNode) in closedSet:
                continue
        
            if neighborNode[0] == endNode[0] and neighborNode[1] == endNode[1]:
                node_details[neighborNode[0]][neighborNode[1]]["parent"] = currentNode
                return tracePath(node_details, endNode)
        
            gNew = node_details[currentNode[0]][currentNode[1]]["g"] + 1
            hNew = calculateHValue(neighborNode,endNode)
            fNew = gNew + hNew
            
            if node_details[neighborNode[0]][neighborNode[1]]["f"] == highest_cost or node_details[neighborNode[0]][neighborNode[1]]["f"] > fNew:
                openSet.append(neighborNode)
                
                node_details[neighborNode[0]][neighborNode[1]]["f"] = fNew
                node_details[neighborNode[0]][neighborNode[1]]["g"] = gNew
                node_details[neighborNode[0]][neighborNode[1]]["h"] = hNew
                node_details[neighborNode[0]][neighborNode[1]]["parent"] = currentNode
                

    return
    
    
def position_to_action(current_pos,dest,pickup):
    
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
        pdb.set_trace()
        

    
    return action
    
    
def go_to_location(x,y, action_sequence, robotState, next_observation):
            
    global path_to_follow
            
    ego_location = np.where(robotState.latest_map == 5)
    
    finished = False
    action = env.action_space.sample()
    action["action"] = -1
    action["num_cells_move"] = 1
    
    output = []
    
    """
    if action_sequence == 0:
        action_sequence += 1
        action = Action.get_occupancy_map.value
    """
    if action_sequence == 0:
        path_to_follow = findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),robotState.latest_map)
        
        if not path_to_follow:
            action["action"] = Action.get_occupancy_map.value
            finished = True
            output = -1
        else:
        
            next_location = [ego_location[0][0],ego_location[1][0]]
            action["action"] = position_to_action(next_location,path_to_follow[0],False)
        
            previous_action = ""
            repetition = 1
            action["num_cells_move"] = repetition 
            
            """
            previous_action = ""
            repetition = 1
            next_location = [ego_location[0][0],ego_location[1][0]]
            for p_idx in range(len(path_to_follow)):
                action["action"] = position_to_action(next_location,path_to_follow[p_idx],False)
                
                if not p_idx:
                    previous_action = action["action"]
                    next_location = path_to_follow[p_idx]
                else:
                    if previous_action == action["action"]:
                        repetition += 1
                        next_location = path_to_follow[p_idx]
                    else:
                        break
                        
            for r in range(repetition-1):
                path_to_follow.pop(0)
                
            action["num_cells_move"] = repetition   
            """ 
            action_sequence += 1
            print(path_to_follow, ego_location)

    else:
        if any(next_observation['action_status'][:2]):
            if ego_location[0][0] == path_to_follow[0][0] and ego_location[1][0] == path_to_follow[0][1]:
                if path_to_follow:
                    path_to_follow.pop(0)
                    
   
            if path_to_follow:    
                next_location = [ego_location[0][0],ego_location[1][0]]
                action["action"] = position_to_action(next_location,path_to_follow[0],False)
            
                previous_action = ""
                repetition = 1
                action["num_cells_move"] = repetition 
                
                """
                for p_idx in range(len(path_to_follow)):
                    action["action"] = position_to_action(next_location,path_to_follow[p_idx],False)
                    
                    if not p_idx:
                        previous_action = action["action"]
                        next_location = path_to_follow[p_idx]
                    else:
                        if previous_action == action["action"]:
                            repetition += 1
                            next_location = path_to_follow[p_idx]
                        else:
                            break
                for r in range(repetition-1):
                    path_to_follow.pop(0)
                action["num_cells_move"] = repetition  
                """
            else:
                action["action"] = Action.get_occupancy_map.value
                finished = True
                
    return action,action_sequence,finished,output
    
def activate_sensor(action_sequence,robotState, next_observation):
    global item_list, item_list_dup

    action = env.action_space.sample()
    action["action"] = -1
    finished = False
    output = []

    if action_sequence == 0:
        action_sequence += 1
        action["action"] = Action.danger_sensing.value
        
    elif action_sequence == 1:
        item_list = info["last_sensed"]
        print(item_list)
        item_list_dup = item_list.copy()
        
        if not item_list: #No items scanned
            action["action"] = Action.get_occupancy_map.value
            finished = True
        else:
        
            object_key = item_list.pop(0)
            
            action["action"] = Action.check_item.value    
            action["item"] = info["object_key_to_index"][object_key]
            
            if not item_list: #item list finished
                action_sequence += 2
            else:
                action_sequence += 1
        
    elif action_sequence == 2:
        object_key = item_list.pop(0)
        action["action"] = Action.check_item.value    
        action["item"] = info["object_key_to_index"][object_key]
        
      
        if not item_list:
            action_sequence += 1
       
            
    elif action_sequence == 3:
        #[“object id”, “object x,y location”, “weight”, “benign or dangerous”, “confidence percentage”
        for key in item_list_dup:
        
            ob_idx = info["object_key_to_index"][key]
        
            if robotState.items[ob_idx]["item_danger_level"] == 1:
                danger_level = "benign"
            else:
                danger_level = "dangerous"
                
            output.append([str(key),str(int(robotState.items[ob_idx]["item_location"][0]))+","+str(int(robotState.items[ob_idx]["item_location"][1])),str(robotState.items[ob_idx]["item_weight"]),danger_level,str(robotState.items[ob_idx]["item_danger_confidence"][0])])
        
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
    
    return action,action_sequence,finished,output
    
def send_message(message, action_sequence,robotState, next_observation):

    action = env.action_space.sample()
    action["action"] = -1
    finished = True

    output = []
    action["action"] =  Action.send_message.value
    
    action["message"] = message

    return action,action_sequence,finished,output
    
def pick_up(object_id,action_sequence,robotState, next_observation):
    
    global action_retry
    
    action = env.action_space.sample()
    action["action"] = -1
    
    ego_location = np.where(robotState.latest_map == 5)

    output = []
    
    finished = False
    
    if action_sequence == 0:
        action_retry = 0
    
        action_sequence += 1    

        ob_idx = info["object_key_to_index"][str(object_id)]
     

        if not robotState.items[ob_idx]["item_weight"]:
            output = -2
            finished = True
            action["action"] = Action.get_occupancy_map.value
        else:
            location = robotState.items[ob_idx]["item_location"]
            action["action"] = position_to_action([ego_location[0][0],ego_location[1][0]],location,True)
        
    elif action_sequence == 1:
        if next_observation['action_status'][0] or action_retry == 2:
            action["action"] = Action.get_occupancy_map.value
    
            finished = True
            
            if action_retry == 2 and not next_observation['action_status'][0]:
                output = -1
            else:
                held_objects.append(str(object_id))
        else:
            ob_idx = info["object_key_to_index"][str(object_id)]
            location = robotState.items[ob_idx]["item_location"]
            action["action"] = position_to_action([ego_location[0][0],ego_location[1][0]],location,True)
            action_retry += 1
    


    return action,action_sequence,finished,output
         
def drop(action_sequence,robotState, next_observation):

    action = env.action_space.sample()
    action["action"] = -1
    finished = True

    output = held_objects
    
    action["action"] = Action.drop_object.value

    return action,action_sequence,finished,output
    
def scan_area(action_sequence,robotState, next_observation):
    action = env.action_space.sample()

    finished = True

    output = []
    
    action["action"] = Action.get_occupancy_map.value

    return action,action_sequence,finished,output
    
def ask_llm(messages, robotState, action_function, output, llm_messages):


    prompt = ""
    token_output = ""
    possible_functions = ["drop","go_to_location","pick_up","send_message","activate_sensor", "scan_area"]


    if "pick_up" in action_function:
        first_index = action_function.index("(") + 1
        arguments = action_function[first_index:].split(",")
        
        if output == -1:
            prompt += "Failed to pick up object " + arguments[0] + ". "
        elif output == -2:
            prompt += "You cannot pick an unknown object. Hint: scan the area and move closer to objects. " #+ arguments[0] + ". "
        else:
            prompt += "Succesfully picked up object " + arguments[0] + ". "
            
    elif "go_to_location" in action_function:
        first_index = action_function.index("(") + 1
        arguments = action_function[first_index:].split(",")
        
        if output:
            prompt += "Failed to move to location (" + arguments[0] + "," + arguments[1] + "). "
        else:
            prompt += "Arrived at location. "
    elif "activate_sensor" in action_function:
        if output:
            prompt += "The sensing results are the following: " + str(output) + ". "
        else:
            prompt += "No objects are close enough to be sensed. Hint: scan the area and move closer to objects. "
    elif "send_message" in action_function:
        prompt += "Sent message. "
        
    elif "drop" in action_function:
        if output:
            prompt += "Dropped object " + output[0] + ". "
            held_objects.pop(0)
        else:
            prompt += "No object to drop. "
    elif "scan_area" in action_function:
        objects_location = np.where(robotState.latest_map == 2)
    
        if objects_location[0].size > 0:
            prompt += "Last seen objects at locations: "
        
        for ob_idx in range(len(objects_location[0])):
            if ob_idx:
                prompt +=", "
            prompt += "(" + str(objects_location[0][ob_idx]) + "," + str(objects_location[1][ob_idx]) + ")"
            
        prompt += ". "
        
        robots_location = np.where(robotState.latest_map == 3)
        
        if robots_location[0].size > 0:
            prompt += "Last seen robots at locations: "
        
        for ob_idx in range(len(robots_location[0])):
            if ob_idx:
                prompt +=", "
            prompt += "(" + str(robots_location[0][ob_idx]) + "," + str(robots_location[1][ob_idx]) + ")"
            
        prompt += ". "
        
    action_status_prompt = prompt
    
    if args.openai and action_status_prompt:
        llm_messages[-1]["content"] = action_status_prompt
    
    status_prompt = ""
    
    if not robotState.object_held and held_objects:
        status_prompt += "Accidentally dropped object " + held_objects[0] + ". "
        held_objects.pop(0)
        
        
    if messages:

        status_prompt += "You have received the following messages: "
        
        for m in messages:
            status_prompt += "Message from " + str(m[0]) + ": '" + m[1] + "'. "
        status_prompt += ". "
        
    
    
    status_prompt += "Your current strength is " + str(robotState.strength) + ". "
    
    ego_location = np.where(robotState.latest_map == 5)
    
    status_prompt += "Your current location is " + "(" + str(ego_location[0][0]) + "," + str(ego_location[1][0]) + "). "
    
    prompt += status_prompt
    
    prompt += "What would be your next action? Write a single function call."
    
    print("Starting to call LLM")
    
    if not args.openai:

    
        #print(prompt)
        
        final_prompt = ""
        
        for m in llm_messages:
            if m["role"] == "system":
                final_prompt += m["content"]
            elif m["role"] == "user":
                final_prompt += "USER >> " + m["content"]
            elif m["role"] == "assistant":
                final_prompt += "ASSISTANT >> " + m["content"]
            final_prompt += "\n"
            
        final_prompt += "USER >> " + prompt + "\nASSISTANT >>\n"
        
        print(final_prompt)
        
        temperature = 0.7
        num_beams = 1
        while True:
        
            inputs = tokenizer(final_prompt, return_tensors="pt").to(device)
            
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=100, num_return_sequences=num_beams, num_beams=num_beams, early_stopping=True, temperature=temperature) #, generation_config=config)

            token_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            new_tokens = inputs.input_ids.size()[1]
            
            function_output = tokenizer.batch_decode(generate_ids[:,new_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            function_output = function_output.replace('\\', '')
            print(function_output)
	        
            x_str = re.search(r"[a-z_]+\([^)]*\)",function_output)
	        
            
            if x_str:
            

                function_output = x_str.group()
                
                
                if function_output and any(func_str in function_output for func_str in possible_functions):
                
                    if "send_message" in function_output:
                        parent_index = function_output.index('(')
                        second_parent_index = function_output.index(')')
                        msg = function_output[parent_index+1:second_parent_index].replace('"','\\"').replace("'","\\'")
                        function_output = "send_message('" + msg +  "')"
              
                    llm_messages.append({"role":"user","content":action_status_prompt + status_prompt})
                    llm_messages.append({"role":"assistant","content":function_output})
                    #output = history_prompt + action_status_prompt + status_prompt + "\nROBOT >> " + function_output #final_prompt + function_output
                    break
                else:
                    num_beams += 1
                    print("Retrying with beams", num_beams)
                    #final_prompt = output + "\nENVIRONMENT >> That is not a function, please try again.\nROBOT >>\n"
            else:
                num_beams += 1
                print("Retrying with beams", num_beams)
                #final_prompt = output + "\nENVIRONMENT >> That is not a function, please try again.\nROBOT >>\n"
	        
            #pdb.set_trace()
            
    else:
        while True:

            response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=[
                    *llm_messages,
                    {"role": "user", "content": prompt}
                ],
              functions=llm_functions,
              function_call="auto"
            )
            print(llm_messages, prompt)
            print(response)

            response_message = response["choices"][0]["message"]
            
            if response_message.get("function_call"):
                llm_messages.append({"role": "user", "content": status_prompt})
                
                function_name = response_message["function_call"]["name"]
                function_args = json.loads(response_message["function_call"]["arguments"])
                
                function_output = function_name + "("
                for key_idx,key in enumerate(function_args.keys()):
                    if key_idx:
                        function_output += ","
                        
                        
                    if "send_message" in function_output:
                        function_output += "'" + str(function_args[key]).replace('"','\\"').replace("'","\\'") +  "'"
                    else:
                        function_output += str(function_args[key])
                    
                function_output += ")"
                llm_messages.append({"role": "function", "name": function_name, "content": ""})
                
                break
    
    log_f.write(final_prompt + function_output + '\n')
    
    return token_output+'\n', function_output
        
def setup_llm(device):

    model = None
    tokenizer = None
    
    if args.openai:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        pretrained_model = "eachadea/vicuna-13b-1.1" #"tiiuae/falcon-40b-instruct" #"eachadea/vicuna-13b-1.1"
        
        model = AutoModelForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.float16, device_map='sequential', max_memory={0: '12GiB', 1: '20GiB'}, revision='main', low_cpu_mem_usage=True, offload_folder='offload')
        
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    
    return model,tokenizer

def print_map(occupancy_map): #Occupancy maps require special printing so that the orientation is correct
    new_occupancy_map = occupancy_map.copy()
    for row_id in range(occupancy_map.shape[0]):
        new_occupancy_map[row_id,:] = occupancy_map[occupancy_map.shape[0]-row_id-1,:]

    new_new_occupancy_map = new_occupancy_map.copy()
    for row_id in range(occupancy_map.shape[1]): 
        new_new_occupancy_map[:,row_id] = new_occupancy_map[:,occupancy_map.shape[1]-row_id-1]
    print(new_new_occupancy_map)


device = "cuda"
model,tokenizer = setup_llm(device)


env = gym.make('gym_collab/AICollabWorld-v0', use_occupancy=args.use_occupancy, view_radius=args.view_radius, client_number=int(args.robot_number), host=args.host, port=args.port, address=args.address, cert_file=args.cert_file, key_file=args.key_file)

observation, info = env.reset()
#observation, reward, terminated, truncated, info = env.step(17)


log_file = "log_ai.txt"
log_f = open(log_file,"w")

done = False

processed_observation = (tuple(map(tuple, observation['frame'])), bool(observation['objects_held']))

print_map(observation["frame"])

next_observation = []

actions_to_take = [*[1]*2,*[2]*3,11,19,*[3]*4,*[0]*5,16,19]

held_objects = []

action_issued = [False,False]
last_action = [0,0]

messages = []
llm_messages = []


obs_sample = env.observation_space.sample()

room_size = str(obs_sample['frame'].shape[0])

history_prompt = "Imagine you are a robot. You can move around a place, pick up objects and use a sensor to determine whether an object is dangerous or not. Your task is to find all dangerous objects in a room and bring them to the middle of that room. The size of the room is " + room_size + " by " + room_size +" meters. There are other robots like you present in the room with whom you are supposed to collaborate. Objects have a weight and whenever you want to pick up an object, you need to make sure your strength value is equal or greater than that weight value at any given moment. That means that whenever you carry a heavy object other robots will need to be next to you until you drop it. You start with a strength of 1, and each other robot that is next to you inside a radius of 3 meters will increase your strength by 1. If you pick up an object you cannot pick up another object until you drop the one you are carrying. Each sensor measurement you make to a particular object has a confidence level, thus you are never totally sure whether the object you are scanning is benign or dangerous. You need to compare measurements with other robots to reduce uncertainty. You can only sense objects by moving within a radius of 1 meter around the object and activating the sensor. You can sense multiple objects each time you activate your sensor, sensing all objects within a radius of 1 meter. You can exchange text messages with other robots, although you need to be at most 5 meters away from them to receive their messages and send them messages. All locations are given as (x,y) coodinates. The functions you can use are the following:\ngo_to_location(x,y): Moves robot to a location specified by x,y coordinates. Returns nothing.\nsend_message(text): Broadcasts message text. Returns nothing.\nactivate_sensor(): Activates sensor. You need to be at most 1 meter away from an object to be able to sense it. Returns a list of lists, each of the sublists with the following format: [“object id”, “object x,y location”, “weight”, “benign or dangerous”, “confidence percentage”]. For example: [[“1”,”4,5”,”1”,”benign”,”0.5”],[“1”,”4,5”,”1”,”benign”,”0.5”]].\npick_up(object_id): Picks up an object with object id object_id. You need to be 0.5 meters from the object to be able to pick it up. Returns nothing.\ndrop(): Drops any object previously picked up. Returns nothing.\nscan_area(): Returns the locations of all objects and robots in the scene.\n"



system_prompt = "Imagine you are a robot. You can move around a place, pick up objects and use a sensor to determine whether an object is dangerous or not. Your task is to find all dangerous objects in a room and bring them to the middle of that room. The size of the room is " + room_size + " by " + room_size +" meters. There are other robots like you present in the room with whom you are supposed to collaborate. Objects have a weight and whenever you want to pick up an object, you need to make sure your strength value is equal or greater than that weight value at any given moment. That means that whenever you carry a heavy object other robots will need to be next to you until you drop it. You start with a strength of 1, and each other robot that is next to you inside a radius of 3 meters will increase your strength by 1. If you pick up an object you cannot pick up another object until you drop the one you are carrying. Each sensor measurement you make to a particular object has a confidence level, thus you are never totally sure whether the object you are scanning is benign or dangerous. You need to compare measurements with other robots to reduce uncertainty. You can only sense objects by moving within a radius of 1 meter around the object and activating the sensor. You can sense multiple objects each time you activate your sensor, sensing all objects within a radius of 1 meter. You can exchange text messages with other robots, although you need to be at most 5 meters away from them to receive their messages and send them messages."

if args.openai:
    llm_messages.append(
        {
            "role": "system",
            "content": system_prompt,
        }
    ) 
else:
    llm_messages.append(
        {
            "role": "system",
            "content": history_prompt,
        }
    ) 

llm_functions = [
    {
        "name": "go_to_location",
        "description": "Move to the specified location",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "integer",
                    "description": "The x coordinate of the location to move to.",
                },
                "y": {
                    "type": "integer",
                    "description": "The y coordinate of the location to move to.",
                },
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "activate_sensor",
        "description": "Activates sensor. You need to be at most 1 meter away from an object to be able to sense it. Returns a list of lists, each of the sublists with the following format: [“object id”, “object x,y location”, “weight”, “benign or dangerous”, “confidence percentage”]. For example: [[“1”,”4,5”,”1”,”benign”,”0.5”],[“1”,”4,5”,”1”,”benign”,”0.5”]].",
        "parameters":{ "type": "object", "properties": {}},
        
    },
    {
        "name": "pick_up",
        "description": "Picks up an object that is 1 meter away from it at most.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "integer",
                    "description": "The ID of the object to pick up",
                }
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "send_message",
        "description": "Broadcasts message.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text message.",
                }
            },
            "required": ["text"],
        },
    },
    {
        "name": "drop",
        "description": "Drops any object previously picked up.",
        "parameters":{ "type": "object", "properties": {}},
        
    },
    {
        "name": "scan_area",
        "description": "Returns the locations of all objects and robots in the scene.",
        "parameters":{ "type": "object", "properties": {}},
        
    },

]


class RobotState:
    def __init__(self, latest_map, object_held, num_robots):
        self.latest_map = latest_map
        self.object_held = object_held
        self.items = []
        self.robots = [{}]*num_robots
        self.strength = 1
        
        
        
robotState = RobotState(observation['frame'].copy(), False, env.action_space["robot"].n-1)

action = env.action_space.sample()



reward_machine_state = 0
action["action"] = agents[reward_machine_state].get_action(processed_observation)

process_reward = 0
process_last_action = action["action"]
reward_machine_state = 0

action["action"] = Action.get_occupancy_map.value #actions_to_take.pop(0)


action_sequence = 0

path_to_follow = []

item_list = []
item_list_dup = []

high_level_action_finished = True

action_function = ""
function_output = []

while not done:

    #action = env.action_space.sample()
    
    #Make sure to issue concurrent actions but not of the same type. Else, wait.
    if action["action"] < Action.danger_sensing.value and not action_issued[0]:
        action_issued[0] = True
        last_action[0] = action["action"]
        print("Locomotion", Action(action["action"]))
    elif action["action"] != Action.wait.value and action["action"] >= Action.danger_sensing.value and not action_issued[1]:
        last_action_arguments = [action["item"],action["robot"],action["message"]]
        action_issued[1] = True
        last_action[1] = action["action"]
        
        print("Sensing", Action(action["action"]))
    else:
        action["action"] = Action.wait.value


    #action = agent.get_action(processed_observation)
    
    
        
    
    next_observation, reward, terminated, truncated, info = env.step(action)

 
    
    if reward != 0:
        print('Reward', reward)
        process_reward = reward
        
        
    if next_observation["num_items"] > len(robotState.items):
        diff_len = next_observation["num_items"] - len(robotState.items)
        robotState.items.extend([{'item_weight': 0, 'item_danger_level': 0, 'item_danger_confidence': np.array([0.]), 'item_location': np.array([-1, -1], dtype=np.int16), 'item_time': np.array([0], dtype=np.int16)}]*diff_len)
        
    robotState.strength = next_observation["strength"]
        
    #When any action has completed
    if next_observation and any(next_observation['action_status']):
        
        
        
        
        ego_location = np.where(next_observation['frame'] == 5)
        previous_ego_location = np.where(robotState.latest_map == 5)
        robotState.latest_map[previous_ego_location[0][0],previous_ego_location[1][0]] = 0
        robotState.latest_map[ego_location[0][0],ego_location[1][0]] = 5

        
        if next_observation['action_status'][2]: #If sensing action was succesful
            if Action(last_action[1]) == Action.get_occupancy_map: #Maintain the state of the occupancy map and update it whenever needed
            
                view_radius = int(args.view_radius)
                
                max_x = ego_location[0][0] + view_radius
                max_y = ego_location[1][0] + view_radius
                min_x = max(ego_location[0][0] - view_radius, 0)
                min_y = max(ego_location[1][0] - view_radius, 0)
                robotState.latest_map[min_x:max_x+1,min_y:max_y+1]= next_observation["frame"][min_x:max_x+1,min_y:max_y+1]
                


                for m_key in info['map_metadata'].keys(): #We get the ids of objects/robots present in the current view and update locations

                    for map_object in info['map_metadata'][m_key]:
                        m_key_xy = m_key.split('_')
                        if isinstance(map_object, list): #Object information

                            ob_key = info["object_key_to_index"][map_object[0]]
                            
                            
                            robotState.items[ob_key]["item_location"] = [int(m_key_xy[0]), int(m_key_xy[1])]

                            
                        elif map_object not in info['map_metadata'][str(ego_location[0][0])+'_'+str(ego_location[1][0])]: #Robot information
                            info['map_metadata'][str(ego_location[0][0])+'_'+str(ego_location[1][0])]
                            ob_key = info["robot_key_to_index"][map_object]
                            robotState.robots[ob_key]["neighbor_location"] = [int(m_key_xy[0]), int(m_key_xy[1])]

                            
                            
                        
            elif Action(last_action[1]) == Action.get_objects_held:

                robotState.object_held = bool(next_observation['objects_held'])
        
            elif Action(last_action[1]) == Action.check_item:
                robotState.items[last_action_arguments[0]] = next_observation["item_output"]
                
            elif Action(last_action[1]) == Action.check_robot:
                robotState.robots[last_action_arguments[1]-1] = next_observation["neighbors_output"]
                
            elif Action(last_action[1]) == Action.get_messages:
                print("Message arrived", info['messages'])
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
        
        
        if high_level_action_finished: #When a high level action finishes, we sense the environment
            if last_action[1] == Action.get_messages.value: #Action.get_occupancy_map.value:
                print("Messages", messages)
                history_prompt, action_function = ask_llm(messages, robotState, action_function, function_output, llm_messages)
                messages = []
                #action_function = input("Next action > ").strip()
                #action_function = "go_to_location(5,5)"
                action_function = action_function[:-1]
            
                if not ("drop" in action_function or "activate_sensor" in action_function or "scan_area" in action_function):
                    action_function += ","
                    
                action_function += "action_sequence, robotState, next_observation)"
                
                
                high_level_action_finished = False
                action_sequence = 0
            elif last_action[1] == Action.get_occupancy_map.value:
                action["action"] = Action.get_objects_held.value
            elif last_action[1] == Action.get_objects_held.value:
                action["action"] = Action.get_messages.value
            else:
                action["action"] = Action.get_occupancy_map.value
            
        
       
        
        if not high_level_action_finished:
        
            
            
            action, action_sequence, high_level_action_finished,function_output = eval(action_function) #go_to_location(x,y, action_sequence, robotState, next_observation)
            
            print(function_output, high_level_action_finished)
            
            
        print_map(robotState.latest_map)
        
     
        print(next_observation['item_output'], next_observation['objects_held'], next_observation['neighbors_output'], next_observation['strength'], next_observation['num_messages'], next_observation['num_items'], next_observation['action_status'], last_action)
        print("Object held", robotState.object_held)

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



