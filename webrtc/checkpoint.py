import glob
import pdb
import json
import numpy as np
from sklearn.cluster import DBSCAN
import re
import pandas as pd
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import shutil
import argparse

parser = argparse.ArgumentParser(
    description="Checkpoint"
)
parser.add_argument('--file', type=str, default='2025_07_09_11_42_26', help='Log file time')
parser.add_argument('--initial', action='store_true', help="Initial settings")
parser.add_argument('--log-dir', type=str, default='', help='Log dir')

args = parser.parse_args()
scenario_options = None
settings = {}


                    
def get_settings(log_state_file):


    settings = log_state_file.readline()
    
    if not settings:
        return None,None
        
    settings = eval(settings.strip())
    arguments = log_state_file.readline().strip()
    scenario_options_line = log_state_file.readline()
    if scenario_options_line:
        scenario_options = eval(scenario_options_line.strip())
    else:
        scenario_options = None

    return scenario_options,settings




if args.log_dir:
    log_dir = args.log_dir
else:
    log_dir = "log"

d = log_dir + "/" + args.file + ".txt"


log_file = open(d)
new_line = log_file.readline()

try:
    log_state_file = open("../simulator/"+ log_dir + "/" + args.file + "_state.txt")
except:
    print("Could not open file", d)
    quit()
scenario_options,settings = get_settings(log_state_file)
just_reset = []

metadata = ""
last_time = 0
latest_item_info = {}


object_agent_locations = []



#print(d)

if args.initial:

    reference_object = ['5', 3, 2]
    found_object = False
    while not scenario_options is None:
    
        for ob in scenario_options["objects"]:
            if ob[0] == reference_object[0]:
                if tuple(ob[:3]) == tuple(reference_object):
                    found_object = True
                
                break
                    
        if found_object:
            break
        scenario_options,settings = get_settings(log_state_file)
            
    if found_object:
        metadata = []
        for ob in scenario_options["objects"]:
            metadata.append([0,ob[0],ob[3]['x'],ob[3]['z'], ob[1],ob[2]])
        metadata = {"metadata":metadata}

    
else:
    while new_line:
        split_line = new_line.strip().split(',')
        
        if int(split_line[1]) == 3:
            
            agent_id = split_line[2]
            #print(new_line)
            
            first_bracket = new_line.find('{')
            second_bracket = new_line.rfind('}')
            third_bracket = new_line[second_bracket:].find('[') + second_bracket
            fourth_bracket = new_line[second_bracket:].find(']') + second_bracket
            fifth_bracket = new_line[fourth_bracket:].find('[') + fourth_bracket
            sixth_bracket = new_line.rfind(']')
            
            item_info = eval(new_line[first_bracket:second_bracket+1])
            
            if item_info:
                if agent_id not in latest_item_info.keys():
                    latest_item_info[agent_id] = {}
                latest_item_info[agent_id].update(item_info)
            
                            

        elif int(split_line[1]) == 4:
            
                
            log_line_agent = split_line[2]

            just_reset.append(log_line_agent)

            if all(robot[0] in just_reset for robot in scenario_options["robots_type"]): 

                #print("Resetting")
                scenario_options_tmp,settings = get_settings(log_state_file)
                
                if scenario_options_tmp:
                    scenario_options = scenario_options_tmp
                    
                just_reset = []
                
                

         
            
        elif int(split_line[1]) == 0:
        
            first_bracket = new_line.find('{')
            second_bracket = new_line.rfind('}')
            
            metadata = eval(new_line[first_bracket:second_bracket+1].replace('\\','').replace("false","False").replace("true","True"))
            last_time = float(split_line[0])

        new_line = log_file.readline()
        
        

if metadata:
    print("LAST METADATA", metadata)
    print(scenario_options["robots_type"])
    print(last_time)
    print(latest_item_info)
    
    json_file = open("checkpoint.json", "w")
    
    json.dump({"metadata":metadata, "sensors": scenario_options["robots_type"], "time": last_time, "item_info": latest_item_info},json_file)
        




