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
import sqlite3
import os
from enum import Enum
import json

from magnebot import ActionStatus
from gym_collab.envs.action import Action

from llm_control import LLMControl
from deepq_control import DeepQControl
from heuristic_control import HeuristicControl
from tutorial_control import TutorialControl
from decision_control import DecisionControl
from optimized_control import OptimizedControl

# GUANHUA Ji -------------------------------------------------------------------
#from ultralytics import YOLO
from PIL import Image
import math
#model = YOLO('./weights/best.pt')
import matplotlib.pyplot as plt
import BEV
bird_eye_view = BEV.BEV(150, 240, 360, 0, 0, 0.75)
ground_truth_map = None
# GUANHUA Ji -------------------------------------------------------------------

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
parser.add_argument("--control", default="optimized", type=str, help="Type of control to apply: heuristic,llm,openai,deepq,q,manual,decision,optimized")
parser.add_argument("--message-loop", action="store_true", help="Use to allow messages to be sent back to sender")
parser.add_argument("--role", default="general", help="Choose a role for the agent: general, scout, lifter")
parser.add_argument("--planning", default="equal", help="Choose a planning role for the agent: equal, coordinator, coordinated")
parser.add_argument('--webcam', action="store_true", help="Use images from virtual webcam")
parser.add_argument('--video-index', type=int, default=0, help='index of the first /dev/video device to capture frames from')
parser.add_argument('--config', type=str, default='team_structure.yaml', help='Path to team structure configuration file')
parser.add_argument('--sql', default=True, action="store_true", help='Use SQL for message parsing')
parser.add_argument('--no-reset', default=False, action="store_true", help='Continue without waiting for reset')
#parser.add_argument('--communication-distance', type=int, default=5, help='local communication distance limit')
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
    def __init__(self, latest_map, object_held, env, args):
    
        self.args = args
        self.create_tables = ''
        self.sqliteConnection = None
        self.dropped_objects = []
    
        if args.sql:
        
            database_name = "agent_db_" + str(args.robot_number) + ".db"
            
            self.create_tables = [
                    """CREATE TABLE objects (
                        object_id INTEGER PRIMARY KEY,
                        idx INTEGER NOT NULL UNIQUE,
                        weight INTEGER,
                        already_sensed TEXT,
                        carried_by TEXT
                    );""",
                    """CREATE TABLE agents (
                        agent_id TEXT PRIMARY KEY,
                        idx INTEGER NOT NULL UNIQUE,
                        type INTEGER,
                        last_seen_location TEXT,
                        last_seen_room TEXT,
                        last_seen_time REAL,
                        collaborative_score REAL,
                        collaborative_score_of_me REAL,
                        team TEXT,
                        carrying_object TEXT,
                        disabled INTEGER,
                        current_state TEXT,
                        sensor_benign REAL,
                        sensor_dangerous REAL,
                        attitude TEXT
                    );""",
                    """CREATE TABLE agent_object_estimates (
                        last_seen_location TEXT,
                        last_seen_room TEXT,
                        last_seen_time REAL,
                        danger_status TEXT,
                        estimate_correct_percentage REAL,
                        agent_id TEXT,
                        object_id INTEGER,
                        FOREIGN KEY (agent_id) REFERENCES agents (agent_id),
                        FOREIGN KEY (object_id) REFERENCES objects (object_id),
                        PRIMARY KEY (agent_id,object_id)
                    );"""]
            
            if not self.args.no_reset:
                
                if os.path.exists(database_name):
                    os.remove(database_name)
                    
                self.sqliteConnection = sqlite3.connect(database_name)
                
                self.cursor = self.sqliteConnection.cursor()
                
                #CHECK(collaborative_score >= 0 AND collaborative_score <= 10)
                
                  
                for statement in self.create_tables:
                    self.cursor.execute(statement)
            else:
                self.sqliteConnection = sqlite3.connect(database_name)
                self.cursor = self.sqliteConnection.cursor()
                #"SELECT o.object_id, o.weight, aoe.danger_status, aoe.estimate_correct_percentage FROM objects o INNER JOIN agent_object_estimates aoe ON o.object_id = aoe.object_id WHERE o.weight = 1 AND aoe.danger_status = 'dangerous' AND aoe.last_seen_room = '3' AND o.already_sensed = 'Yes';"

            
        if self.args.no_reset:
            file_name="agent_map_" + str(args.robot_number) + ".txt"
            with open(file_name, 'rb') as filetoread:
                latest_map = np.load(filetoread)
                
        self.latest_map = latest_map
        self.object_held = object_held
        self.items = []
        self.item_estimates = {}
        self.env = env
        self.current_action_description = ""
        self.possible_estimates = {}
        self.saved_locations = {}
        
        
        if self.args.sql:
            if not self.args.no_reset:
                for n in range(len(env.neighbors_info)+1):
                
                    if n == len(env.neighbors_info): #myself
                        robot_id2 = self.env.robot_id
                        sensor_parameters = env.sensor_parameters
                        robot_type = 1
                        
                    else:
                        robot_id2 = list(self.env.robot_key_to_index.keys())[list(self.env.robot_key_to_index.values()).index(n)]
                        sensor_parameters = env.neighbors_sensor_parameters[n]
                        robot_type = env.neighbors_info[n][1]
                        
                    self.cursor.execute('''INSERT INTO agents (agent_id, idx, last_seen_location, last_seen_time, type, carrying_object, disabled, sensor_benign, sensor_dangerous,collaborative_score,collaborative_score_of_me, team, current_state,last_seen_room,attitude) VALUES (?, ?, "[]", 0, ?, "None", -1, ?, ?, 10, 10, "[]", '', '','')''', (robot_id2, n, robot_type, sensor_parameters[0],sensor_parameters[1]))  
                    
            else:
                item_idx = 0            
                while True:
                    row_exists = self.cursor.execute("""SELECT COUNT(*) FROM objects WHERE idx = ?;""", (item_idx,)).fetchall()
                    if not row_exists[0][0]:
                        break
                    weight = self.cursor.execute("""SELECT weight FROM objects WHERE idx = ?;""", (item_idx,)).fetchall()[0][0]
                    estimates = self.cursor.execute("""SELECT aoe.last_seen_location, MAX(aoe.last_seen_time) FROM agent_object_estimates aoe INNER JOIN objects o ON aoe.object_id = o.object_id WHERE o.idx = ?;""", (item_idx,)).fetchall()[0]
                    self.items.append({'item_weight': weight, 'item_danger_level': 0, 'item_danger_confidence': np.array([0.]), 'item_location': np.array(eval(estimates[0]), dtype=np.int16), 'item_time': np.array([estimates[1]], dtype=np.int16)})
                    self.bayesian_fusion(item_idx)
                    item_idx += 1
        else:
            self.robots = [{"neighbor_type": env.neighbors_info[n][1], "neighbor_location": [-1,-1], "neighbor_time": [0.0], "neighbor_disabled": -1, "collaborative_score":10, "collaborative_score_of_me":10} for n in range(len(env.neighbors_info))] # 0 if human, 1 if ai
            
        self.strength = 1
        self.map_metadata = {}
        self.collaborative_score = [{"collaborative_score":[], "collaborative_score_of_me":[]} for n in range(len(env.neighbors_info)+1)]
        
    class Danger_Status(Enum):
        unknown = 0
        benign = 1
        dangerous = 2
        
        
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
    
    
        if self.args.sql:
            #object_id = list(self.env.object_key_to_index.keys())[list(self.env.object_key_to_index.values()).index(item_idx)]
            estimates = self.cursor.execute("SELECT aoe.danger_status, a.sensor_benign, a.sensor_dangerous FROM agent_object_estimates aoe INNER JOIN agents a ON a.agent_id = aoe.agent_id INNER JOIN objects o ON o.object_id = aoe.object_id WHERE o.idx = ?;""", (item_idx,)).fetchall()
            
            for ie in estimates: #we update according to results already obtained
            
                if self.Danger_Status[ie[0]].value:
                
                    samples = True
                    
                    if self.Danger_Status[ie[0]].value == 2:
                        benign = 1-ie[1]
                        dangerous = ie[2]
                    elif self.Danger_Status[ie[0]].value == 1:
                        benign = ie[1]
                        dangerous = 1-ie[2]
                
                    prob_evidence = (prior_benign*benign + prior_dangerous*dangerous)
                    
                    prior_benign = benign*prior_benign/prob_evidence
                    prior_dangerous = dangerous*prior_dangerous/prob_evidence
        else:
    
            for ie_idx,ie in enumerate(self.item_estimates[item_idx]): #we update according to results already obtained
            
                if ie["item_danger_level"]:
                
                    samples = True
                    
                    if ie_idx == len(self.item_estimates[item_idx])-1:
                        if ie["item_danger_level"] == 2:
                            benign = 1-self.env.sensor_parameters[0]
                            dangerous = self.env.sensor_parameters[1]
                        elif ie["item_danger_level"] == 1:
                            benign = self.env.sensor_parameters[0]
                            dangerous = 1-self.env.sensor_parameters[1]
                            
                    else:
                        if ie["item_danger_level"] == 2:
                            benign = 1-self.env.neighbors_sensor_parameters[ie_idx][0]
                            dangerous = self.env.neighbors_sensor_parameters[ie_idx][1]
                        elif ie["item_danger_level"] == 1:
                            benign = self.env.neighbors_sensor_parameters[ie_idx][0]
                            dangerous = 1-self.env.neighbors_sensor_parameters[ie_idx][1]
                
                    prob_evidence = (prior_benign*benign + prior_dangerous*dangerous)
                    
                    prior_benign = benign*prior_benign/prob_evidence
                    prior_dangerous = dangerous*prior_dangerous/prob_evidence
                
                
        """
        for ie_idx,ie in enumerate(self.item_estimates[item_idx]): #we get possible estimates
        
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
                
        """        
                
                
                    
                
        if samples:
            prior_list = [prior_benign,prior_dangerous]
            dangerous_level = np.argmax(prior_list)
            self.items[item_idx]["item_danger_level"] = dangerous_level + 1
            self.items[item_idx]["item_danger_confidence"] = [prior_list[dangerous_level]]
        
    
    def get_num_robots(self):
    
        if self.args.sql:
            count = len(self.cursor.execute("SELECT * FROM agents WHERE agent_id != ?;", (self.env.robot_id,)).fetchall())
        else:
            count = len(self.robots)
                
        return count
        
    def get_all_robots(self):
        agents = []
        if self.args.sql:
            agents = self.cursor.execute("SELECT * FROM agents;").fetchall()
            
        return agents
        
    def get_num_estimates(self, idx):
                
        if self.args.sql:
            count = len(self.cursor.execute("SELECT * FROM agent_object_estimates aoe INNER JOIN objects o ON o.object_id = aoe.object_id WHERE o.idx = ?;", (idx,)).fetchall())
        else:
            count = len(self.item_estimates[idx])
    
        return count
        
    def get_object_keys(self):
    
        if self.args.sql:
            keys = [k[0] for k in self.cursor.execute("SELECT idx FROM objects;").fetchall()]
        else:
            keys = self.item_estimates.keys()
            
        return keys
    
    def get_num_objects(self):
    
        if self.args.sql:
            count = len(self.cursor.execute("SELECT * FROM objects;").fetchall())
        else:
            count = len(self.items)
                
        return count
        
    def get_all_objects(self):
        objects = []
        if self.args.sql:
            objects = self.cursor.execute("SELECT * FROM objects;").fetchall()
            
        return objects
    
    def get_all_sensing_estimates(self):
        estimates = []
        if self.args.sql:
            estimates = self.cursor.execute("SELECT object_id,agent_id,danger_status FROM agent_object_estimates;").fetchall()
            
        return estimates
            
    def get_object_in_room(self, propert, idx):
        row = self.cursor.execute("SELECT " + propert + " FROM agent_object_estimates WHERE last_seen_room = ? AND agent_id = ?;", (idx[0],idx[1],)).fetchall()
        
        output = []
        if propert == "danger_status":
            for r in row:
                output.append((self.Danger_Status[r[0]].value,))
                
        
        return output
        
    def get_all_object_rooms(self):
        rooms = set()
        if self.args.sql:
            objects = self.get_all_objects()
            propert = "last_seen_room"
            try:
                for idx in range(len(objects)):
                    if not self.get("objects","last_seen_time",idx):
                        row = self.cursor.execute("SELECT aoe." + propert + " FROM agent_object_estimates aoe INNER JOIN objects o ON o.object_id = aoe.object_id INNER JOIN agents a ON a.agent_id = aoe.agent_id WHERE o.idx = ? AND a.idx = ?;", (idx ,self.get_num_robots(),)).fetchall()
                    else:
                        row = self.cursor.execute("SELECT " + propert + " FROM agent_object_estimates aoe INNER JOIN objects o ON o.object_id = aoe.object_id WHERE last_seen_time = (SELECT MAX(aoe.last_seen_time) FROM agent_object_estimates aoe INNER JOIN objects o ON o.object_id = aoe.object_id WHERE o.idx = ?) AND o.idx = ?;", (idx,idx,)).fetchall()
                    
                    row = row[0][0]
                    
                    rooms.add(row)
            except:
                pdb.set_trace() 
        return rooms
        
    def get(self, database, propert, idx):
    
        
    
        if self.args.sql:
            
            if database == "agents":
                if idx == -1:
                    row = self.cursor.execute("SELECT " + propert + " FROM " + database + ";").fetchall()

                else:
                    row = self.cursor.execute("SELECT " + propert + " FROM " + database + " WHERE idx = ?;", (idx,)).fetchall()
                    
                    
            elif database == "object_estimates":
                row = self.cursor.execute("SELECT aoe." + propert + " FROM agent_object_estimates aoe INNER JOIN objects o ON o.object_id = aoe.object_id INNER JOIN agents a ON a.agent_id = aoe.agent_id WHERE o.idx = ? AND a.idx = ?;", (idx[0],idx[1],)).fetchall()
            
                if propert == "danger_status":
                    row = [(self.Danger_Status[row[0][0]].value,)]
            elif database == "objects":
            
                if propert == "weight" or propert == "already_sensed":
                    row = self.cursor.execute("SELECT " + propert + " FROM " + database + " WHERE idx = ?;", (idx,)).fetchall()
                elif propert in ["danger_status", "estimate_correct_percentage"]:
                    legacy_to_sql = {"danger_status": "item_danger_level", "estimate_correct_percentage": "item_danger_confidence"}
                
                    row = [(self.items[idx][legacy_to_sql[propert]],)]
                    
                    if propert == "estimate_correct_percentage":
                        row = row[0]
                        
                elif propert == "last_seen_location":
                    if not self.get("objects","last_seen_time",idx):
                        row = self.cursor.execute("SELECT aoe." + propert + " FROM agent_object_estimates aoe INNER JOIN objects o ON o.object_id = aoe.object_id INNER JOIN agents a ON a.agent_id = aoe.agent_id WHERE o.idx = ? AND a.idx = ?;", (idx ,self.get_num_robots(),)).fetchall()
                    else:
                        row = self.cursor.execute("SELECT " + propert + " FROM agent_object_estimates aoe INNER JOIN objects o ON o.object_id = aoe.object_id WHERE last_seen_time = (SELECT MAX(aoe.last_seen_time) FROM agent_object_estimates aoe INNER JOIN objects o ON o.object_id = aoe.object_id WHERE o.idx = ?) AND o.idx = ?;", (idx,idx,)).fetchall()
                else:
                    row = self.cursor.execute("SELECT " + propert + " FROM agent_object_estimates aoe INNER JOIN objects o ON o.object_id = aoe.object_id WHERE last_seen_time = (SELECT MAX(aoe.last_seen_time) FROM agent_object_estimates aoe INNER JOIN objects o ON o.object_id = aoe.object_id WHERE o.idx = ?) AND o.idx = ?;", (idx,idx,)).fetchall()
             
            if idx == -1:
                row_tmp = []
                for r in row:
                    row_tmp.append(r[0])
                row = row_tmp
            else:    
                try:
                    row = row[0][0]
                except:
                    pdb.set_trace()
   
            
            if propert == "last_seen_location":
                row = self.env.convert_to_grid_coordinates(eval(row))
                
            #elif propert == "last_seen_time":
            #    pdb.set_trace()
                
        else:
        
            if database == "agents":
                legacy_to_sql = {"type": "neighbor_type", "disabled": "neighbor_disabled", "last_seen_location": "neighbor_location", "last_seen_time": "neighbor_time"}
                
                if idx == -1:
                    row = [rc[legacy_to_sql[propert]] for rc in self.robots]
                else:
                    row = self.robots[idx][legacy_to_sql[propert]]
                    
                    if propert == "last_seen_time":
                        row = row[0]
                
            elif database == "object_estimates":
                legacy_to_sql = {"danger_status": "item_danger_level", "estimate_correct_percentage": "item_danger_confidence"}
                
                row = self.item_estimates[idx[0]][idx[1]][legacy_to_sql[propert]]
                
            elif database == "objects":
                legacy_to_sql = {"weight":"item_weight", "danger_status": "item_danger_level", "estimate_correct_percentage": "item_danger_confidence", "last_seen_location": "item_location", "last_seen_time": "item_time"}
                
                row = self.items[idx][legacy_to_sql[propert]]
                
                if propert == "estimate_correct_percentage":
                    row = row[0]
                elif propert == "last_seen_time":
                    row = row[0]
                    
        return row
            
    def set(self, database, propert, idx, value, time):
    
        if propert == "collaborative_score" or propert == "collaborative_score_of_me":
            self.collaborative_score[idx][propert].append(value)
            value = sum(self.collaborative_score[idx][propert])/len(self.collaborative_score[idx][propert])
    
        if self.args.sql:
        
            if propert == "last_seen_location":
                value = self.env.convert_to_real_coordinates(value)
                room = self.env.get_room(value,False)
                value = str(value)
        
            if database == "objects":
                #print("object myself", time, self.cursor.execute("SELECT * FROM agent_object_estimates aoe INNER JOIN objects o ON o.object_id = aoe.object_id ;""", ).fetchall())
                
                if propert == "danger_status":
                    pdb.set_trace()
                
                try:
                    if propert == "last_seen_location":
                        self.cursor.execute("UPDATE agent_object_estimates SET " + propert + " = ?, last_seen_time = ?, last_seen_room = ? WHERE agent_id = ? AND object_id IN (SELECT object_id FROM objects WHERE idx = ?);", (value, float(time), room, self.env.robot_id, idx,)).fetchall()
                    else:
                        self.cursor.execute("UPDATE agent_object_estimates SET " + propert + " = ?, last_seen_time = ? WHERE agent_id = ? AND object_id IN (SELECT object_id FROM objects WHERE idx = ?);", (value, float(time), self.env.robot_id, idx,)).fetchall()
                    
                except:
                    pdb.set_trace()
            elif database == "agents":
                try:
                    if propert == "last_seen_location":
                        self.cursor.execute("UPDATE " + database + " SET " + propert + " = ?, last_seen_room = ? WHERE idx = ?;", (value, room, idx,)).fetchall()
                    else:
                        self.cursor.execute("UPDATE " + database + " SET " + propert + " = ? WHERE idx = ?;", (value, idx,)).fetchall()
                except:
                    pdb.set_trace()

            self.sqliteConnection.commit()
            
        else:
            if database == "objects":
                legacy_to_sql = {"weight":"item_weight", "danger_status": "item_danger_level", "estimate_correct_percentage": "item_danger_confidence", "last_seen_location": "item_location", "last_seen_time": "item_time"}
                
                if propert in legacy_to_sql:
                    propert = legacy_to_sql[propert]
                    
                self.items[idx][propert] = value
                
            elif database == "agents":
                legacy_to_sql = {"type": "neighbor_type", "disabled": "neighbor_disabled", "last_seen_location": "neighbor_location", "last_seen_time": "neighbor_time"}
                
                if propert in legacy_to_sql:
                    propert = legacy_to_sql[propert]
                    
                self.robots[idx][propert] = value
        
        
    def update_robots(self, neighbor_output, robot_idx):
    
        if self.args.sql:
            if neighbor_output["neighbor_type"] >= 0:
                self.cursor.execute("""UPDATE agents SET type = ? WHERE idx = ?;""", (neighbor_output["neighbor_type"], robot_idx,)) 
        
            if neighbor_output["neighbor_disabled"] >= 0:
                self.cursor.execute("""UPDATE agents SET disabled = ? WHERE idx = ?;""", (neighbor_output["neighbor_disabled"], robot_idx,))
        
            try:
                self.get("agents", "last_seen_time", robot_idx) <= neighbor_output["neighbor_time"][0]
            except:
                pdb.set_trace()
            if self.get("agents", "last_seen_time", robot_idx) <= neighbor_output["neighbor_time"][0]:
            
                grid_location = [int(neighbor_output["neighbor_location"][0]),int(neighbor_output["neighbor_location"][1])]
                robot_location = self.env.convert_to_real_coordinates(grid_location)
                room = self.env.get_room(robot_location,False)
                #print("neighbor", neighbor_output["neighbor_time"][0], self.cursor.execute("SELECT * FROM agent_object_estimates aoe INNER JOIN objects o ON o.object_id = aoe.object_id ;""", ).fetchall())
                
                last_location = robotState.get("agents", "last_seen_location", robot_idx)
                
                self.cursor.execute("""UPDATE agents SET last_seen_location = ?, last_seen_time = ?, last_seen_room = ? WHERE idx = ?;""", (str(robot_location), float(neighbor_output["neighbor_time"][0]), room, robot_idx,))
                
                '''
                ego_location = np.where(self.latest_map == 5)
                view_radius = int(self.args.view_radius)
                max_x = ego_location[0][0] + view_radius
                max_y = ego_location[1][0] + view_radius
                min_x = max(ego_location[0][0] - view_radius, 0)
                min_y = max(ego_location[1][0] - view_radius, 0)
                
                if ((grid_location[0] < min_x or grid_location[0] > max_x) or (grid_location[1] < min_y or grid_location[1] > max_y)) and tuple(grid_location) not in self.saved_locations.keys():
                    self.saved_locations[tuple(grid_location)] = self.latest_map[grid_location[0], grid_location[1]]
                '''
                
                
                
                
                if self.latest_map[grid_location[0], grid_location[1]] != 5:
                    self.latest_map[grid_location[0], grid_location[1]] = 3
                    
                    
                    
                    if not (last_location[0] == -1 and last_location[1] == -1) and not (last_location[0] == grid_location[0] and last_location[1] == grid_location[1]) and not (last_location[0] >= view_limits[0][0] and last_location[0] <= view_limits[0][1] and last_location[1] >= view_limits[1][0] and last_location[1] <= view_limits[1][1]):
                        self.latest_map[last_location[0], last_location[1]] = -2
                    
            self.sqliteConnection.commit()
        else:
            if neighbor_output["neighbor_type"] >= 0:
                self.robots[robot_idx]["neighbor_type"] = neighbor_output["neighbor_type"]
        
            if neighbor_output["neighbor_disabled"] >= 0:
                self.robots[robot_idx]["neighbor_disabled"] = neighbor_output["neighbor_disabled"]
        
            if self.robots[robot_idx]["neighbor_time"][0] <= neighbor_output["neighbor_time"][0]:
            
                self.robots[robot_idx]["neighbor_location"] = [int(neighbor_output["neighbor_location"][0]),int(neighbor_output["neighbor_location"][1])]
                self.robots[robot_idx]["neighbor_time"] = neighbor_output["neighbor_time"]
                
                if self.latest_map[self.robots[robot_idx]["neighbor_location"][0], self.robots[robot_idx]["neighbor_location"][1]] != 5:
                    self.latest_map[self.robots[robot_idx]["neighbor_location"][0], self.robots[robot_idx]["neighbor_location"][1]] = 3
    
    
    def initialize_object(self, item_id, item_idx):
    
        self.cursor.execute('''INSERT INTO objects (object_id, idx, weight, already_sensed, carried_by) VALUES (?, ?, 0, "No", NULL)''', (item_id, item_idx,))
                
        num_robots = self.get_num_robots()
        for n in range(num_robots+1):
        
            if n == num_robots:
                robot_id2 = self.env.robot_id
            else:
                robot_id2 = list(self.env.robot_key_to_index.keys())[list(self.env.robot_key_to_index.values()).index(n)]
                
            self.cursor.execute('''INSERT INTO agent_object_estimates (object_id, last_seen_location, last_seen_time, danger_status, estimate_correct_percentage, last_seen_room, agent_id) VALUES (?,"[]", 0, ?, 0, "", ?)''', (item_id, self.Danger_Status(0).name, robot_id2,))  
    
    def update_items(self,item_output, item_id, item_idx, robot_idx): #Updates items

        information_change = False
        #We save estimates from all robots
        
        num_objects = self.get_num_objects()
        if item_idx >= num_objects:
            
            diff_len = item_idx+1 - num_objects
            #print("item_change", item_idx, len(self.items), diff_len)
            
            if self.args.sql:
                for d in range(num_objects,item_idx+1):
                    item_id = list(self.env.object_key_to_index.keys())[list(self.env.object_key_to_index.values()).index(d)]
                    self.initialize_object(item_id, d)
            
            self.items.extend([{'item_weight': 0, 'item_danger_level': 0, 'item_danger_confidence': np.array([0.]), 'item_location': np.array([-1, -1], dtype=np.int16), 'item_time': np.array([0], dtype=np.int16)} for d in range(diff_len)])
            information_change = True


        if self.args.sql:
            row_exists = self.cursor.execute("""SELECT COUNT(*) FROM objects WHERE idx = ?;""", (item_idx,)).fetchall()
            
            if not row_exists[0][0]:
            
                self.initialize_object(item_id, item_idx)
                
                information_change = True
                
            if robot_idx == -1:
                robot_id2 = self.env.robot_id
            else:
                robot_id2 = list(self.env.robot_key_to_index.keys())[list(self.env.robot_key_to_index.values()).index(robot_idx)]
                
            #print("myself", item_output["item_time"][0], self.cursor.execute("SELECT * FROM agent_object_estimates aoe INNER JOIN objects o ON o.object_id = aoe.object_id ;""", ).fetchall())

            item_loc = self.env.convert_to_real_coordinates([int(item_output["item_location"][0]),int(item_output["item_location"][1])])
            if not item_loc:
                pdb.set_trace()
            room = self.env.get_room(item_loc,False)
            item_loc = str(item_loc)
            
            self.cursor.execute("""UPDATE agent_object_estimates SET last_seen_location = ?, last_seen_time = ?, last_seen_room = ? WHERE object_id = ? AND agent_id = ?;""", (item_loc, float(item_output["item_time"][0]), room, item_id, robot_id2,))
            
            
            if item_output["item_danger_level"]:
                danger_level_translate = self.Danger_Status(item_output["item_danger_level"]).name
                self.cursor.execute("""UPDATE agent_object_estimates SET danger_status = ?, estimate_correct_percentage = ? WHERE object_id = ? AND agent_id = ?;""", (danger_level_translate, float(item_output["item_danger_confidence"][0]), item_id, robot_id2,))    
                information_change = True
                
                if robot_idx == -1:
                    self.cursor.execute("""UPDATE objects SET already_sensed = ? WHERE object_id = ?;""", ("Yes", item_id,))

              
        else:     

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
        
            old_location = []
            if not (self.items[item_idx]["item_location"][0] == -1 and self.items[item_idx]["item_location"][1] == -1) and not (self.items[item_idx]["item_location"][0] == int(item_output["item_location"][0]) and self.items[item_idx]["item_location"][1] == int(item_output["item_location"][1])) and not (self.items[item_idx]["item_location"][0] >= view_limits[0][0] and self.items[item_idx]["item_location"][0] <= view_limits[0][1] and self.items[item_idx]["item_location"][1] >= view_limits[1][0] and self.items[item_idx]["item_location"][1] <= view_limits[1][1]):
                old_location = list(self.items[item_idx]["item_location"]).copy()
        
            self.items[item_idx]["item_location"] = [int(item_output["item_location"][0]),int(item_output["item_location"][1])]
            self.items[item_idx]["item_time"] = item_output["item_time"]
            
            if self.latest_map[self.items[item_idx]["item_location"][0], self.items[item_idx]["item_location"][1]] != 5 and self.latest_map[self.items[item_idx]["item_location"][0], self.items[item_idx]["item_location"][1]] != 3 and self.latest_map[self.items[item_idx]["item_location"][0], self.items[item_idx]["item_location"][1]] != 4:
                self.latest_map[self.items[item_idx]["item_location"][0], self.items[item_idx]["item_location"][1]] = 2
                #print("changing object location")
            
            if old_location:
                self.latest_map[old_location[0], old_location[1]] = -2 #See if this works, after an object is taken, asume there is nothing there anymore
            
        if item_output["item_weight"]:
            self.items[item_idx]["item_weight"] = item_output["item_weight"]
            
            if self.args.sql:
                self.cursor.execute("""UPDATE objects SET weight = ? WHERE object_id = ?;""", (item_output["item_weight"], item_id,))    
        
        
        if "carried_by" in item_output:
        
            if not item_output["carried_by"]:
                value = None
            else:
                value = item_output["carried_by"]
                
            self.cursor.execute("""UPDATE objects SET carried_by = ? WHERE object_id = ?;""", (value, item_id,)) 
        
        if information_change:
            #print(item_output)
            #self.average_fusion(item_idx)
            self.bayesian_fusion(item_idx)
        
        if self.args.sql:
            self.sqliteConnection.commit()  
        
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
    
txt_profiling = open("profiling_" + str(args.robot_number) + ".json", 'w')

just_starting = True
rearrange_observations = True

while True:

    observation, info = env.reset(options=args)
    
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
    robotState = RobotState(observation['frame'].copy(), 0, env, args)
    #observation, reward, terminated, truncated, info = env.step(17)
    done = False

    action_issued = [False,False]
    last_action = [0,0]
    previous_action = -1

    messages = []
    message_queue = []
    
    action = env.action_space.sample()
    if args.no_reset:
        action["action"] = Action.danger_sensing.value
    else:
        action["action"] = Action.get_occupancy_map.value #actions_to_take.pop(0)

    action['num_cells_move'] = 1

    high_level_action_finished = True

    action_function = ""
    function_output = []

    step_count = 0
    
    process_reward = 0
    
    disabled = False
    dropped_objects = []
    

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
    elif args.control == 'optimized':
        decision_control = OptimizedControl(env, robotState, team_structure)

    last_high_action = action["action"]
    
    fell_down = 0
    
    view_limits = []
    
    
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
        
        if info["dropped_objects"]:
            robotState.dropped_objects.append(info["dropped_objects"])
                    
        #print("HELLO", info["object_key_to_index"], env.object_info)
        #print(info["real_location"])

        # Computer Vision --------------------------------------------------------------------
        if args.webcam:
            x_loc = info['real_location'][0][0]
            y_loc = info['real_location'][0][2]
            z_loc = info['real_location'][0][1]
            yaw = math.radians(180 + info['real_location'][1][0])
            pitch = math.radians(info['real_location'][1][1])
            roll = math.radians(-90 + info['real_location'][1][2])
            cv2.namedWindow('Webcam')
            cv2.namedWindow('occupancy map')
            #cv2.namedWindow('ground truth')
            bird_eye_view.updata_camera_matrix(x_loc, y_loc, roll, pitch, yaw)

            img = info["frame"]
            if img is not None:
                results = model(img, stream=True)
                x_me, y_me = [x_loc], [-y_loc]
                x_box_abs, y_box_abs = [], []
                x_robot_abs, y_robot_abs = [], []

                box_scores = []
                robot_scores = []
                for r in results:
                    boxes = r.boxes

                    for box in boxes:
                        # bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                        c = box.cls
                        # confidence
                        confidence = math.ceil((box.conf[0]*100))/100
                        print("Confidence --->",confidence)
                        print("Class name -->", model.names[int(c)])

                        if confidence >= 0.4:
                            # put box in cam
                            cv2.circle(img, (int((x1 + x2) / 2), y2), radius = 2, color = (0, 0, 255), thickness = 2)
                            x = (x1 + x2) / 2
                            y = 240 - y2
                            #x_abs, y_abs, _ = BEV.get_BEV_of_complex_camera(x, y, x_loc, y_loc, yaw, pitch, roll)
                            x_abs, y_abs, _ = bird_eye_view.get_BEV(x, y)
                            y_abs = -y_abs
                            x_abs, y_abs = BEV.apply_offset((x_abs, y_abs), (x_me[0], y_me[0]))
                            if round(x_abs - 0.5) >= -10 and round(x_abs - 0.5) < 10 and round(y_abs - 0.5) >= -10 and round(y_abs - 0.5) < 10:
                                if model.names[int(c)] == 'box':
                                    x_box_abs.append(round(x_abs - 0.5))
                                    y_box_abs.append(round(y_abs - 0.5))
                                    box_scores.append(confidence)
                                else:
                                    x_robot_abs.append(round(x_abs - 0.5))
                                    y_robot_abs.append(round(y_abs - 0.5))
                                    robot_scores.append(confidence)
                            else:
                                print("Eliminate:", x_abs, y_abs)
                            org = [x1, y1]
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            fontScale = 0.5
                            color = (255, 0, 0)
                            thickness = 2
                            cv2.putText(img, model.names[int(c)], org, font, fontScale, color, thickness)
            
                x_box_abs, y_box_abs = BEV.non_max_suppression(x_box_abs, y_box_abs, box_scores, 0.2)
                x_robot_abs, y_robot_abs = BEV.non_max_suppression(x_robot_abs, y_robot_abs, robot_scores, 0.2)
                #plot_abs = BEV.create_BEV_image(y_box_abs, x_box_abs, y_robot_abs, x_robot_abs, y_me, x_me)

                occupancy_map = bird_eye_view.get_visibility_map(-10, 10, -10, 10)
                for x_box, y_box in zip(x_box_abs, y_box_abs):
                    x_box += 10
                    y_box += 10
                    y_box = 19 - y_box
                    occupancy_map[x_box, y_box] = 2
                for x_robot, y_robot in zip(x_robot_abs, y_robot_abs):
                    x_robot += 10
                    y_robot += 10
                    y_robot = 19 - y_robot
                    occupancy_map[x_robot, y_robot] = 3
                occupancy_map[round(x_me[0] - 0.5) + 10, 9 - round(y_me[0] - 0.5)] = 5
                BEV.visionBlocked(occupancy_map)
                if ground_truth_map is not None:
                    #plot_ground_truth = BEV.create_visibility_image(ground_truth_map)
                    BEV.add_walls(occupancy_map, ground_truth_map)
                    #cv2.imshow('ground truth', plot_ground_truth)
                elif np.count_nonzero(next_observation['frame'] == -2) == 0:
                    ground_truth_map = next_observation['frame'].copy()

                plot_occupancy_map = BEV.create_visibility_image(occupancy_map)
                cv2.imshow('Webcam', img)
                cv2.imshow('occupancy map', plot_occupancy_map)
                cv2.waitKey(100)
            else:
                print("No Image")
            #if next_observation['action_status'][2]
            #next_observation['frame'] = occupancy_map
            #print(next_observation['frame'])
            '''
            count += 1
            if count % 10 == 0:
                filename = f"frames/frame{count}.jpg"
                success = cv2.imwrite(filename,info["frame"])
                if success:
                    print("successfully saved", filename)
                else:
                    print("failed")
            '''

            '''
            cv2.waitKey(100) 
            '''
        # Computer Vision --------------------------------------------------------------------


        if args.message_loop:
            info["robot_key_to_index"][env.robot_id] = robotState.get_num_robots()-1

        
        
        if reward != 0:
            #print('Reward', reward)
            process_reward += reward
            
        #print(next_observation["num_items"])
        
        if info["status"] == ActionStatus.tipping:
            fell_down += 1
            if fell_down >= 1000:
                print("FELL DOWN")
                done = True
        else:
            fell_down = 0
            
        num_objects = robotState.get_num_objects()
        if next_observation["num_items"] > num_objects:
            diff_len = next_observation["num_items"] - num_objects
            robotState.items.extend([{'item_weight': 0, 'item_danger_level': 0, 'item_danger_confidence': np.array([0.]), 'item_location': np.array([-1, -1], dtype=np.int16), 'item_time': np.array([0], dtype=np.int16)} for d in range(diff_len)])
            
            if args.sql:
                for d in range(num_objects,next_observation["num_items"]):
                    try:
                        item_id = list(env.object_key_to_index.keys())[list(env.object_key_to_index.values()).index(d)]
                        robotState.initialize_object(item_id, d)
                    except:
                        pdb.set_trace()
            
        robotState.strength = next_observation["strength"]
            
        #When any action has completed
        if next_observation and any(next_observation['action_status']) and not disabled:
            
            
            #print(info["object_key_to_index"], env.object_info)
            
            if args.no_reset and rearrange_observations:
                objects = robotState.get_all_objects()
                objects.sort(key=lambda x:x[1])
                new_object_info = []
                new_object_to_key = {}
                for ob_idx,ob in enumerate(objects):
                    if not ob[2]:
                        location = robotState.items[ob_idx]["item_location"]
                        time = robotState.items[ob_idx]["item_time"][0]
                        new_object_info.append([str(ob[0]),ob[2],{}, location[0], location[1], time])
                    else:
                        not_found = True
                        for ob2 in env.object_info:
                            
                            if int(ob2[0]) == ob[0]:
                                new_object_info.append(ob2)
                                not_found = False
                                break
                        if not_found:
                            print("NOT FOUND ITEM FOR NO RESET")
                    new_object_to_key[str(ob[0])] = ob_idx
                    
                env.set_object_info(new_object_info.copy(), new_object_to_key)
                rearrange_observations = False
            #else:
            #    pdb.set_trace()
            ego_location = np.where(next_observation['frame'] == 5)
            previous_ego_location = np.where(robotState.latest_map == 5)
            
            robotState.latest_map[previous_ego_location[0][0],previous_ego_location[1][0]] = 0 #If there was an agent there it will eliminate it from the map
            
            robot_count = robotState.get_num_robots()
                
            for ob_key in range(robot_count):
            
                try:
                    location = robotState.get("agents", "last_seen_location", ob_key)
                except:
                    pdb.set_trace()
                neighbor_disabled = robotState.get("agents", "disabled", ob_key)

                
                if location[0] != -1 and location[1] != -1 and previous_ego_location[0][0] == location[0] and previous_ego_location[1][0] == location[1]:
                    robotState.latest_map[previous_ego_location[0][0],previous_ego_location[1][0]] = 3
                    
                if neighbor_disabled == 1 and location[0] != -1 and location[1] != -1:
                    robotState.latest_map[location[0],location[1]] = 1
            

            
            if next_observation['action_status'][2]: #If sensing action was succesful
                if Action(last_action[1]) == Action.get_occupancy_map: #Maintain the state of the occupancy map and update it whenever needed
                
                    view_radius = int(args.view_radius)
                    
                    max_x = ego_location[0][0] + view_radius
                    max_y = ego_location[1][0] + view_radius
                    min_x = max(ego_location[0][0] - view_radius, 0)
                    min_y = max(ego_location[1][0] - view_radius, 0)
                    
                    previous_robo_map = np.copy(robotState.latest_map)
                    
                    limited_map = next_observation["frame"][min_x:max_x+1,min_y:max_y+1]
                    
                    m_ids = np.where(limited_map != -2)
                    
                    if m_ids[0].size:
                        robotState.latest_map[min_x:max_x+1,min_y:max_y+1][m_ids] = limited_map[m_ids]
                    #robotState.latest_map[min_x:max_x+1,min_y:max_y+1]= next_observation["frame"][min_x:max_x+1,min_y:max_y+1]
                    
                    view_limits = [[min_x,max_x],[min_y,max_y]]
                    
                    walls = np.where(next_observation["frame"] == 1)
                    
                    for w in range(len(walls[0])):
                        if robotState.latest_map[walls[0][w],walls[1][w]] == -2:
                            robotState.latest_map[walls[0][w],walls[1][w]] = 1
                    
                    #print(next_observation["frame"])
                    
                    to_delete = []
                    #This is when we localize agents at all time regardless of view radius
                    for saved_coord in robotState.saved_locations.keys():
                        if ((saved_coord[0] < min_x or saved_coord[0] > max_x) or (saved_coord[1] < min_y or saved_coord[1] > max_y)):
                            if next_observation["frame"][saved_coord[0],saved_coord[1]] != 3:
                                robotState.latest_map[saved_coord[0],saved_coord[1]] = robotState.saved_locations[saved_coord]
                                to_delete.append(saved_coord)
                        else:
                            to_delete.append(saved_coord)
                    
                    for d in to_delete:
                        del robotState.saved_locations[d]
                    
                    if not np.array_equal(robotState.latest_map,previous_robo_map):
                        file_name="agent_map_" + str(args.robot_number) + ".txt"
                        with open(file_name, 'wb') as filetowrite:
                            np.save(filetowrite,robotState.latest_map)
                    
                    robotState.map_metadata = info['map_metadata']

                    for m_key in info['map_metadata'].keys(): #We get the ids of objects/robots present in the current view and update locations

                        for map_object in info['map_metadata'][m_key]:
                            m_key_xy = m_key.split('_')
                            if not map_object[0]: #Object information


                                ob_key = info["object_key_to_index"][map_object[1]]
                                
                                #strength = map_object[2] #ELIMINATE
                             
                                template_item_info = {'item_weight': 0, 'item_danger_level': 0, 'item_danger_confidence': np.array([0.]), 'item_location': np.array([int(m_key_xy[0]), int(m_key_xy[1])], dtype=np.int16), 'item_time': np.array([info["time"]], dtype=np.int16), "carried_by": map_object[4]}
                                robotState.update_items(template_item_info, map_object[1], ob_key, -1)
                                #robotState.items[ob_key]["item_location"] = [int(m_key_xy[0]), int(m_key_xy[1])]
                    

                                
                            elif map_object[1] != env.robot_id: #map_object not in info['map_metadata'][str(ego_location[0][0])+'_'+str(ego_location[1][0])]: #Robot information
                                
                                robot_idx = info["robot_key_to_index"][map_object[1]]
                                
                                template_robot_info = {"neighbor_type": -1, "neighbor_location": np.array([int(m_key_xy[0]), int(m_key_xy[1])], dtype=np.int16), "neighbor_time": np.array([info["time"]], dtype=np.int16), "neighbor_disabled": map_object[2]}
                                #print("Disabled:", map_object)
                                robotState.update_robots(template_robot_info, robot_idx)
                                
                                
                                
                            
                elif Action(last_action[1]) == Action.get_objects_held:

                    robotState.object_held = next_observation['objects_held']
            
                elif Action(last_action[1]) == Action.check_item:
                
                    #robotState.items[last_action_arguments[0]] = next_observation["item_output"]
                    object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(last_action_arguments[0])]
                    robotState.update_items(next_observation["item_output"], object_id, last_action_arguments[0], -1)
                    
                elif Action(last_action[1]) == Action.check_robot: #Make sure to update estimates and take the one with the highest confidence
                
                    robot_idx = last_action_arguments[1]-1
       
                    robotState.update_robots(next_observation["neighbors_output"], robot_idx)
                        

                        

                    
                elif Action(last_action[1]) == Action.get_messages:
                    #print("Message arrived", info['messages'])
                    messages = info['messages']
                    
                    for m in messages:
                        
                        location = env.convert_to_grid_coordinates(m[3][0])
                        if location:
                            template_robot_info = {"neighbor_type": -1, "neighbor_location": np.array(location, dtype=np.int16), "neighbor_time": np.array([m[2]], dtype=np.int16), "neighbor_disabled": 0}
                            robot_idx = info["robot_key_to_index"][m[0]]
                            robotState.update_robots(template_robot_info, robot_idx)
                        
                        for carried_object_loc in m[3][1].keys():
                            location = env.convert_to_grid_coordinates(m[3][1][carried_object_loc])
                            if location:
                                template_item_info = {'item_weight': 0, 'item_danger_level': 0, 'item_danger_confidence': np.array([0.]), 'item_location': np.array(location, dtype=np.int16), 'item_time': np.array([m[2]], dtype=np.int16), "carried_by": m[0]}
                                ob_key = info["object_key_to_index"][carried_object_loc]
                                robotState.update_items(template_item_info, carried_object_loc, ob_key, -1)
                    
                
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
            
            
            if next_observation['action_status'][2]: #If sensing action was succesful
                if Action(last_action[1]) == Action.get_occupancy_map: #Maintain the state of the occupancy map and update it whenever needed
    
                    robot_count = robotState.get_num_robots()
                        
                    for ob_key in range(robot_count):
                    
                        location = robotState.get("agents", "last_seen_location", ob_key)
                        neighbor_disabled = robotState.get("agents", "disabled", ob_key)

                    
                        robo_location = robotState.latest_map[location[0],location[1]]
                        if robo_location != 5 and robo_location != 3 and location[0] != -1 and location[1] != -1 and neighbor_disabled != 1:
                            #pdb.set_trace()
                            print("ROBOT NOT FOUND", ob_key, location)
                            
                            robotState.set("agents","last_seen_location",ob_key, [-1,-1], info["time"]) 
                            
                            
                            #robotState.robots[ob_key]["neighbor_time"] = info["time"]
                            
                    for ob_key in range(robotState.get_num_objects()): #If the agent is not where it was last seen, mark it
                        try:
                            ob_location = robotState.get("objects", "last_seen_location", ob_key)
                            item_location = robotState.latest_map[ob_location[0],ob_location[1]]
                            if item_location == 0:
                                robotState.set("objects", "last_seen_location", ob_key, [-1,-1], info["time"])
                        except:
                            pdb.set_trace()
            	
            
            
            if high_level_action_finished: #When a high level action finishes, we sense the environment
                if last_action[1] == Action.get_messages.value or last_action[1] == Action.check_item.value: #Action.get_occupancy_map.value:
                
                    print_map(robotState.latest_map)
                    #print("Held:",robotState.object_held)
                
                    last_action[1] = 0 #Reset last sensing action
                    step_count += 1
                    
                    #print("Messages", messages)
                    if args.control == 'llm' or args.control == 'openai':
                        action,terminated_tmp = llm_control.control(messages, robotState, info, next_observation)
                        
                        if terminated_tmp:
                            disabled = True
                            env.sio.emit("disable")
                        #high_level_action_finished = False
                        
                    elif args.control == "decision" or args.control == "optimized":
                    
                        if args.control == "optimized":
                            
                            if previous_action <= Action.move_right.value and previous_action >= Action.move_up.value and decision_control.profiling['previous_ego_location'] != [int(ego_location[0][0]),int(ego_location[1][0])]:
                                timing_value = time.time() -  decision_control.profiling['current_time']
                                if not decision_control.profiling['previous_action'] or decision_control.profiling['previous_action'] == previous_action:
                                    decision_control.profiling['moving_straight'].append(timing_value)
                                elif ((decision_control.profiling['previous_action'] == Action.move_right.value or decision_control.profiling['previous_action'] == Action.move_left.value) and (previous_action == Action.move_right.value or previous_action == Action.move_left.value)) or ((decision_control.profiling['previous_action'] == Action.move_up.value or decision_control.profiling['previous_action'] == Action.move_down.value) and (previous_action == Action.move_up.value or previous_action == Action.move_down.value)):
                                    decision_control.profiling['moving_180_turn'].append(timing_value)
                                else:
                                    decision_control.profiling['moving_turn'].append(timing_value)
                                
                                decision_control.profiling['previous_action'] = previous_action   
                                print(decision_control.profiling)
                                json.dump(decision_control.profiling,txt_profiling)
                    
                    
                        action,terminated_tmp = decision_control.control(messages, robotState, info, next_observation)
                        
                        #if action["action"] == Action.send_message.value:
                        #    print("MEssage:",action)
                        
                        if action["action"] == Action.send_message.value and "message_ai" in action:
                            if action["message_ai"]:
                                
                                if action["message"]:
                                    high_level_action_finished = False
                                    message_queue.append(action["message_ai"])
                                else:
                                    action["message"] = action["message_ai"]
                                    action["robot"] = -1
                            
                            del action["message_ai"]
                            
                        elif action["action"] == Action.send_message.value and "message_queue" in action:
                            if any(action["message_queue"]):
                                
                                if any(action["message"]):
                                    high_level_action_finished = False
                                    message_queue.append(action["message_queue"])
                                else:
                                    action["message"] = action["message_queue"]
                            
                            del action["message_queue"]
                        
                        if args.control == "optimized":
                            if action["action"] <= Action.move_right.value and action["action"] >= Action.move_up.value:
                                decision_control.profiling['current_time'] = time.time()
                                decision_control.profiling['previous_ego_location'] = [int(ego_location[0][0]),int(ego_location[1][0])]
                        previous_action = action['action']
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
            for ob_key in range(robotState.get_num_objects()):
                if robotState.get("objects", "weight", ob_key) >= len(env.map_config['all_robots'])+1 and robotState.get("objects", "danger_status", ob_key) == 2:
                    object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(ob_key)]
                    objects_to_report.append(object_id)

            if objects_to_report:
                print("Reporting", objects_to_report)
                env.sio.emit("report", (objects_to_report))
                    
                
            



print("Closing environment")
env.close()



