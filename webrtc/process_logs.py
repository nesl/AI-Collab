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
from matplotlib.patches import Patch
from collections import Counter

json_file = []
total_sessions = 0
communication_distance = 5
good_sessions = {}
good_names = {}


preamble = """Analyze the content of each message and try to tell me who are they answering to? Some messages may be directed towards a subset of the group. Output a JSON format with the following fields for each analyzed message: "number" and "reply_to", where "number" is the number of the analyzed message, and "reply_to" is any of the set {"Agent A", "Agent B", "Agent C", "Agent D", "Everyone"}.

For example, if I provide you with the next log of messages:

{Number: 1 | Sender: Agent D | Message: Hi everyone}
{Number: 2 | Sender: Agent A | Message: hi}
{Number: 3 | Sender: Agent C | Message: Hi}
{Number: 4 | Sender: Agent C | Message: I ll follow A}
{Number: 5 | Sender: Agent A | Message: ok}
{Number: 6 | Sender: Agent D | Message: Sounds good, lets each scan and establish which objects are dangerous}

[{"number": 1, "reply_to": "Everyone"},{"number": 2, "reply_to": "Agent D"},{"number": 3, "reply_to": "Agent D"},{"number": 4, "reply_to": "Agent A"},{"number": 5, "reply_to": "Agent C"},{"number": 6, "reply_to": "Agent A, Agent C"}]"""

#print(glob.glob("*.txt"))

session_log = []
json_session = {}
different_agents = []
conversations = {0: [[]]}
history_conversation_groups = {0: [[]]}
messages_present = False
distances = {}
conversation_groups = {}
last_time = {0: [[]]}
all_messages = {"messages":[],"agents":[], "preambles":[], "conversation_groups": []}
scenario_options = None
settings = {}


def format_messages_llm(scenario_options):

    global session_log,json_session,different_agents,conversations,messages_present,distances,conversation_groups,last_time,total_sessions,good_sessions,good_names,all_messages,history_conversation_groups

    if session_log:
        json_session["content"][session_num] = session_log.copy()
        #tutorial = not all(so[1] == "human" for so in scenario_options["robots_type"])
        
        #json_session["tutorial"][session_num] = tutorial
        
        print(session_num)
        
        '''
        if tutorial:
            print("tutorial")
        else:
            print("not tutorial")
        '''
        
        robots_str = ""
        for r_idx,r in enumerate(scenario_options["robots_type"]):
            if r[1] == "ai":
                robot_type = "AI"
            elif r[1] == "human":
                robot_type = "Human"
            
            if r_idx:
                robots_str += ", "
                
            robots_str += "Agent " + r[0] + " is " + robot_type
            
            
                
        
        print(robots_str)
        for m_idx,m in enumerate(json_session["content"][session_num]):
            print(m_idx+1, m)
            
        #print(json_session["content"][session_num])
        
        if len(session_log) > 10 and len(different_agents) <= 4:
            total_sessions += 1
            if len(different_agents) not in good_sessions.keys():
                good_sessions[len(different_agents)] = 0
                good_names[len(different_agents)] = []
            good_sessions[len(different_agents)] += 1
            good_names[len(different_agents)].append(json_session["file"])
        

    #if messages_present:
    #    print(session_log)

    session_log = []
    
    distances = {}
    
    #if messages_present:
    #    pdb.set_trace()
    if conversations[0] and conversations[0][0]:
        message_log = {"messages":[preamble], "preambles":[0], "conversation_groups":[[]]}
        message_count = 1
        for c_key in conversations.keys():
            
            
            for d_idx,d in enumerate(conversations[c_key]):
                history_messages = []
                for t_idx,t in enumerate(d):
                    #print(t)
                    
                     if t_idx:
                        messg = "" 
                        
                        messg += "History of messages: " + str(history_messages) + "\n"
                        messg += "New message: " + str(t) + "\n"
                        messg += "Who is " + t["Sender"] + " replying to? Output a JSON.\n"
                        message_log["messages"].append(messg)
                        message_log["conversation_groups"].append(history_conversation_groups[c_key][d_idx])
                        
                        message_count += 1
                        
                     history_messages.append(t)
                     
                     
                     if message_count == 20:
                        message_log["preambles"].append(len(message_log["messages"]))
                        message_log["conversation_groups"].append([])
                        message_count = 0
                        message_log["messages"].append(preamble)
                    
            #print(json.dumps(message_log))
                
            #print("")
        
        if len(message_log["messages"]) > 1:
            all_messages["messages"].append(message_log["messages"])
            all_messages["preambles"].append(message_log["preambles"])
            all_messages["conversation_groups"].append(message_log["conversation_groups"])
            all_messages["agents"].append(different_agents)
            
    conversations = {0: [[]]} #{}
    history_conversation_groups = {0: [[]]}
    conversation_groups = {}
    last_time = {0: [[]]} #{}
    messages_present = False
    different_agents = []
                    
def get_settings(log_state_file):


    settings = log_state_file.readline()
    
    if not settings:
        return None
        
    settings = eval(settings.strip())
    arguments = log_state_file.readline().strip()
    scenario_options_line = log_state_file.readline()
    if scenario_options_line:
        scenario_options = eval(scenario_options_line.strip())
    else:
        scenario_options = None

    return scenario_options,settings



def match_survey(json_session, survey_results):

    for us in json_session["users"].keys():
        for sr in survey_results:
            try:
                if json_session["users"][us]["token"] == sr["token"]:
                    json_session["users"][us]["survey"] = {"questions": sr["questions"], "answers": sr["answers"]}
            except:
                pdb.set_trace()
                
                
    return json_session
                
                
def reset_json_session(file_d):
    json_session = {}
    
    json_session["file"] = file_d
    json_session["content"] = {}
    json_session["users"] = {}
    
    return json_session
    

def identify_who_brought_object(metadata, heavy_objects_in_goal, goal_radius):
    
    carried_by = {ob:"" for ob in heavy_objects_in_goal}
    
    try:
        for m in metadata:
            to_delete = []
            for ob in heavy_objects_in_goal:
                if np.linalg.norm(np.array(m["metadata"][int(ob)][2:4])) < goal_radius:
                    for ag in m["metadata"][20:]:
                        if ob in ag[5:7]:
                            carried_by[ob] = ag[1]
                            to_delete.append(ob)
                            break
                            
            for td in to_delete:
                heavy_objects_in_goal.remove(td)
            
    except:
        pdb.set_trace()       
    return carried_by 
    

def pandas_process(filtering_column, condition_columns, metric_column):

    performance = {}

    for a in df.groupby(filtering_column)[[*condition_columns, metric_column]]:
    
        condition = a[1].iloc[0][condition_columns].to_string(header=False, index=False).strip().replace('\n',' ')

        try:
            if condition not in performance:
                performance[condition] = []
        except:
            pdb.set_trace()
        performance[condition].append(a[1].iloc[0][metric_column])
        
        #print(a)
    
    for ps in performance.keys():
        print(ps, len(performance[ps]), np.nanmean(performance[ps]), np.nanstd(performance[ps]))
        
     
    return performance
    
def plot_dist(title, data, discrete=False):

    #sns.kdeplot(data=data) #, bins=10, kde=True)
    if not discrete:
        sns.displot(data, kind="kde", fill=False,  bw_adjust=0.5)
    else:
        #sns.displot(data, discrete=True, multiple="dodge", shrink=.8)
        sns.displot(data, kind="kde", fill=False,  bw_adjust=0.8)
    plt.title(title)

    """
    num_elem = int(np.ceil(np.sqrt(len(data.keys()))))
    fig, axs = plt.subplots(num_elem, num_elem)
    fig.suptitle(title)
    
    
    max_x = 0
    max_y = 0
    for key_idx,key in enumerate(data.keys()):
        max_x += len(data[key])
        
        if max(data[key]) > max_y :
            max_y = max(data[key])
            
    for key_idx,key in enumerate(data.keys()):
        n2 = key_idx % num_elem
        n1 = key_idx // num_elem

        data[key] = np.array(data[key])

        data[key] = data[key][~np.isnan(data[key])]

        frq, edges = np.histogram(data[key], range = (0,max_y))
        
        #sns.histplot(data=data[key], bins=20, kde=True, ax=axs[n1,n2])

        #pdb.set_trace()
        axs[n1, n2].bar(edges[:-1], frq, width=np.diff(edges), edgecolor="black", align="edge")
        axs[n1, n2].set_title(key)
        axs[n1, n2].set_ylim(top=max_x)
    """       
    plt.show()
    
    
def copy_files(src,processed_folder,subfolder,date,identifier):


    #subfolder = ["videos/", "initial_state/", "traces/"]
    new_folder = processed_folder + subfolder
    if not os.path.isdir(new_folder):
        os.mkdir(new_folder)

    #src = ["../../simulator/videos/","../../simulator/log/","./"]
    glob_files = glob.glob(src + date + "*")
    for f in glob_files:
        name = os.path.basename(f).replace(date,identifier)
        shutil.copy(f, new_folder + name)

agent_info = []
agent_info_columns = []
message_logs = {}
human_names = {}
message_to_df = [["Date", "Time", "AI Agents", "Sender", "Received by", "Message", "Sent by AI?", "Received by AI?"]]

extra_stats_questions = ["team_quality_no_dropped", "productivity", "quality_choosing", "quality_communication", "quality_carrying", "quality_moving", "team_quality_choosing", "team_quality_moving", "team_quality_communication", "team_quality_carrying", "total_distance_traveled", "team_objects_in_goal", "percentage_carried_heavy_objects", "percentage_carried_heavy_objects_mixed", "number_carried_heavy_objects", "total_messages_exchanged", "mixed_message_exchanges", "percentage_mixed_messages", "percentage_mixed_messages_AI_sender", "mixed_carrying_leader", "carrying_leader", "complete_coverage", "overlapping_coverage_mixed", "percentage_carried_objects_dangerous_ai", "percentage_carried_objects_dangerous_human", "times_played_simulated_environment", "informational_exchanges_mixed", "average_grabbed", "average_sensed", "percentage_grabbed_ai", "percentage_sensed_ai", "avg_num_messages_human", "avg_message_len_human", "avg_num_messages_ai", "avg_message_len_ai", "avg_messages_sent", "avg_messages_directed_to_ai", "avg_messages_directed_to_human", "avg_human_message_len_compared", "percentage_collective_sensed","percentage_ai_human_sensed","percentage_all_sensed", "percentage_sensed_both_types", "freq_sense"]

stats_questions = ['distance_traveled', 'grabbed_objects', 'grab_attempts', 'dropped_outside_goal', 'objects_sensed', 'sensor_activation', 'objects_in_goal', 'dangerous_objects_in_goal', 'num_messages_sent', 'average_message_length', 'time_with_teammates_A', 'time_with_teammates_B','time_with_teammates_C','time_with_teammates_D', 'end_time', 'team_dangerous_objects_in_goal', 'total_dangerous_objects', 'quality_work', 'effort', 'human_team_effort', 'team_end_time', 'team_quality_work', 'team_speed_work', 'team_achievement', "task_completion", "team_completion", *extra_stats_questions]

final_questions = ["Dependable","Reliable","Unresponsive","Predictable","Act consistently","Function successfully","Malfunction","Have errors","Provide feedback","Meet the needs of the mission/task","Provide appropriate information","Communicate with people","Perform exactly as instructed", "Follow directions"]

files = glob.glob("log/20*.txt")
files.sort(key=lambda x: datetime.strptime(x[4:23], "%Y_%m_%d_%H_%M_%S"))

print(files)

processed_folder = "processed_traces/"

if os.path.isdir(processed_folder):
    shutil.rmtree(processed_folder)

os.mkdir(processed_folder)

#files = ["2024_09_24_11_40_52.txt"]

#df_message = pd.read_csv('message_check_processed.csv', header=None, encoding='latin-1')

obfuscated_date = 0

track_experience = {}

together_files = [["2025_07_10_17_12_54", "2025_07_10_17_28_39"]]

previous_file = ""
final_average_survey_results = {}

final_metrics = {"total_distance_traveled":[], "distribution_sensed":[], "objects_carried":[], "dangerous_objects":[], "weight_distribution_carried":[]}
tutorial = "2025_07_10_15_24_22"
debug_var = False

BOLD = "\033[32m"
RESET = "\033[0m"

for d in files:

    if debug_var:
        print("\n".join(message_history))
        quit()

    if tutorial in d:
        continue
    
    #d = "log/2025_07_10_15_56_34.txt"
    
    dtime = d[4:-4]
    
    if "events" not in d:
    
        
    
        log_file = open(d)
        new_line = log_file.readline()
        try:
            log_state_file = open("../simulator/log/" + d[4:-4] + "_state.txt")
        except:
            print("Could not open file", d)
            continue
        scenario_options,settings = get_settings(log_state_file)
        just_reset = []
        
        num_humans = 0
        robots_ais = 0
        robots_type = {}
        
        
        
        
        for agent in scenario_options["robots_type"]:
            
            if agent[1] == "human":
                num_humans += 1
                robots_type[agent[0]] = 'human'
            else:
                robots_ais += 1
                robots_type[agent[0]] = 'ai'
        
        continue_file = False
        if any(dtime in t for t in together_files) and not previous_file:
            previous_file = dtime
        elif any(dtime in t for t in together_files) and previous_file:
            continue_file = True
            previous_file = ""
            print("Continue", dtime)
        else:
            previous_file = ""
            
        if not continue_file:
            strategy = ''
            leader = ''
            agent_model = {}
            agent_reliability = {}
            metadata = ""
            last_time = 0
            latest_item_info = {}
            objects_grabbed = {}
            objects_fully_sensed = {}
            objects_partially_sensed = {}
            
            metrics = {"goal_objects":[]}
            object_agent_locations = []
            agents = {}
            survey_results = []
            average_survey_results = {}
            time_spent_carrying = {}
            message_history = []
        
        
        #print(d)
        
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
                
                                
                object_grabbed = eval(new_line[third_bracket:fourth_bracket+1])
                try:
                    objects_dropped = eval(new_line[fifth_bracket:sixth_bracket+1])
                except:
                    pdb.set_trace()
                
                accidentally_dropped = False
                alt_object_id = ""
                for ob in objects_dropped:
                    if ob[0] == agent_id:
                        accidentally_dropped = True
                        alt_object_id = ob[1]
                        break
                
                #if objects_dropped:
                #    pdb.set_trace()
                
                if any(object_grabbed) and not (agent_id in objects_grabbed and objects_grabbed[agent_id]):
                
                    object_id = ""
                    for ob in object_grabbed:
                        if ob:
                            object_id = ob
                            break
                
                    objects_grabbed[agent_id] = object_id
                        
                    #print("Object " + object_id + " grabbed at " + split_line[0] + " by " + agent_id, object_grabbed)

                elif not any(object_grabbed) and agent_id in objects_grabbed and objects_grabbed[agent_id]:
                    
                    object_id = objects_grabbed[agent_id]
                    
                    objects_grabbed[agent_id] = ""
                    
                    if not accidentally_dropped:
                        ob_str = "Object " + object_id + " dropped at " + split_line[0] + " by " + agent_id
                    else:
                        ob_str = "Object " + object_id + " accidentally dropped at " + split_line[0] + " by " + agent_id
                    
                    if object_id not in metrics["goal_objects"]:
                        for m in metadata["metadata"]:
                            if not m[0] and m[1] == object_id and np.linalg.norm(np.array(m[2:4])-np.array([0,5])) < settings["goal_radius"]:
                                ob_str += ". Object in goal"
                                metrics["goal_objects"].append(object_id)
                                break
                            
                    #print(ob_str, object_grabbed)
                #else:
                #    print("Accidentally droped object " + alt_object_id + " by team with agent " + agent_id)
                    
                    
                if agent_id not in objects_fully_sensed.keys():
                    objects_fully_sensed[agent_id] = []
                if agent_id not in objects_partially_sensed.keys():
                   objects_partially_sensed[agent_id] = []
                   
                for item_key in item_info.keys():
                    if "sensor" in item_info[item_key]:
                        if item_key not in objects_fully_sensed[agent_id]:
                            objects_fully_sensed[agent_id].append(item_key)
                        
                            #print("Fully sensed object " + item_key + " at time " + split_line[0] + " by " + agent_id, item_info[item_key])
                    elif "weight" in item_info[item_key] and item_info[item_key]["weight"]:
                        if item_key not in objects_partially_sensed[agent_id]:
                            objects_partially_sensed[agent_id].append(item_key)
                        
                            #print("Partially sensed object " + item_key + " at time " + split_line[0] + " by " + agent_id, item_info[item_key])
        
        
            elif int(split_line[1]) == 4:
                
                    
                log_line_agent = split_line[2]
        
                just_reset.append(log_line_agent)

                if all(robot[0] in just_reset for robot in scenario_options["robots_type"]): 
        
                    #print("Resetting")
                    
                    #print("NUMBER OF OBJECTS IN GOAL", metrics["goal_objects"])
                    
                    
                    scenario_options_tmp,settings = get_settings(log_state_file)
                    
                    if scenario_options_tmp:
                        scenario_options = scenario_options_tmp
                        
                    just_reset = []
                    
                    first_bracket = new_line.find('{')
                    second_bracket = new_line.rfind('}')
                    
                    newer_line = new_line[first_bracket:second_bracket+1].replace("false", "False").replace("true","True")
                    
                    team_strategy = eval(newer_line)
                    
                    for agent in scenario_options["robots_type"]:
                        if team_strategy["hierarchy"][agent[0]] == "order":
                            if agent[1] == "human":
                                strategy = "human leader"
                            else:
                                strategy = "AI leader"
                                
                            leader = agent[0]
                                
                            break
             
            elif int(split_line[1]) == 2:
                
                first_bracket = new_line.find('{')
                second_bracket = new_line.rfind('}')
                
                
                messages = eval(new_line[first_bracket:second_bracket+1])
                
                regex_str = "My goal is (\w+) (\(-?\d+\.\d+,-?\d+\.\d+\)), I'm moving towards (\(-?\d+\.\d+,-?\d+\.\d+\)). My current location is (\(-?\d+\.\d+,-?\d+\.\d+\)).( Carrying object (\w+).)?( Helping (\w+).)?"
                    
                rematch = re.search(regex_str,messages["whole"])
                if rematch:
                    messages["whole"] = messages["whole"].replace(rematch.group(0), "").strip()
                
                if messages["whole"]:
                    mess = "Message sent by " + split_line[2] + " at time " + split_line[0] + ": " + messages["whole"]
                    #print("Message sent by " + split_line[2] + " at time " + split_line[0], messages["whole"])
                    if "steering" in messages["whole"]:
                        print("Message sent by " + split_line[2] + " at time " + split_line[0], messages["whole"])
                        debug_var = True
                        mess = BOLD + mess + RESET
                    message_history.append(mess)
                    for m_key in messages.keys():
                        if m_key != "whole":
                            rematch = re.search(regex_str,messages[m_key])
                            if rematch:
                                messages[m_key] = messages[m_key].replace(rematch.group(0), "").strip()
                            
                            if messages[m_key]:
                                pass
                                #print("Received by " + m_key, messages[m_key])
                
            elif int(split_line[1]) == 0:
            
                first_bracket = new_line.find('{')
                second_bracket = new_line.rfind('}')
                
                metadata = eval(new_line[first_bracket:second_bracket+1].replace('\\','').replace("false","False").replace("true","True"))
                last_time = float(split_line[0])
                
                for ob in metadata["metadata"]:
                    if ob[0]:
                    
                        new_position = np.array(ob[2:4])
                    
                        if ob[1] not in agents.keys():
                            agents[ob[1]] = {"last_position": np.array([]), "distance_traveled": 0.0}
                            
                        if agents[ob[1]]["last_position"].size > 0:
                            agents[ob[1]]["distance_traveled"] += float(np.linalg.norm(new_position - agents[ob[1]]["last_position"]))
                            
                        agents[ob[1]]["last_position"] = np.array(new_position)
        
            
            elif int(split_line[1]) == 8:
                
                first_bracket = new_line.find('[')
                second_bracket = new_line.find(']')
                third_bracket = new_line.rfind('[')
                fourth_bracket = new_line.rfind(']')
                
                last_comma = new_line.rfind(',')
                
                questions = eval(new_line[first_bracket:second_bracket+1])
                answers = eval(new_line[third_bracket:fourth_bracket+1])
                
                if not average_survey_results:
                    average_survey_results = {q:None for q in questions}
                
                for q_idx in range(len(questions)):
                    
                    if answers[q_idx]:
                        val = float(answers[q_idx])
                        if not (val > 100 and q_idx <= 13):
                            if average_survey_results[questions[q_idx]] is None:
                                average_survey_results[questions[q_idx]] = [float(answers[q_idx])]
                            else:
                                average_survey_results[questions[q_idx]].append(float(answers[q_idx]))
                
                survey_results.append({"token":new_line[last_comma+1:].strip(), "questions": questions, "answers": [int(a) if a else None for a in answers]})
        
            new_line = log_file.readline()
            
            
        '''
        if metadata:
            print("LAST METADATA", metadata)
            print(scenario_options["robots_type"])
            print(last_time)
            print(latest_item_info)
        '''        
        
        if False:
            if strategy == 'AI leader':
                analyzed_agents = [leader]
            else:
                analyzed_agents = [a for a in robots_type.keys() if robots_type[a] == "ai"]
            
            for analyzed_a in analyzed_agents:
            
                
                d2 = "../ai_controller/log/" + dtime + "_" + analyzed_a + "_strategy.txt"
                log_file = open(d2)
                new_line = log_file.readline()
                while new_line:
                
                    info_decision = eval(new_line.replace("false", "False").replace("true","True"))
                    #pdb.set_trace()
                    if not info_decision["type"]:
                        #print("Assignments by " + analyzed_a + " at time " + str(info_decision["time"]))
                        for i in info_decision["assignments"]:
                            #print(i[0], i[1])
                            pass
                            
                        #print("Rooms: ", info_decision["rooms"])
                        
                        s_agent = {}
                        for s in info_decision["sensed_objects"]:
                            if s[0] not in s_agent.keys():
                                s_agent[s[0]] = []
                            s_agent[s[0]].append(s[1])
                        
                        #print("Sensed objects", s_agent)
                        
                    else:
                        pass
                        #print("Changed time for recs at time " + str(info_decision["time"]) + ":", info_decision["time_last_suggestion_interval"])
                
                    new_line = log_file.readline()
                    
                
                d3 = "../ai_controller/log/" + dtime + "_" + analyzed_a + "_monitor.txt"
                log_file = open(d3)
                new_line = log_file.readline()
                while new_line:
                
                    split_line = new_line.strip().split(',')
                    agent_id = split_line[2]
                    
                    if int(split_line[1]) == 0:
                    
                        first_bracket = new_line.find('{')
                        second_bracket = new_line.rfind('}')
                
                        info = eval(new_line[first_bracket:second_bracket+1])
                        
                        if agent_id not in agent_reliability.keys():
                            agent_reliability[agent_id] = info
                            #print("Reliability for agent " + agent_id + " at time " + str(split_line[0]), info)
                        else:
                            changed_reliability = []
                            for i in info.keys():
                            
                                if agent_reliability[agent_id][i] != info[i]:
                                    changed_reliability.append((i,info[i]))
                                    
                            #print("Changed reliability for agent " + agent_id + " at time " + str(split_line[0]), changed_reliability)
                            agent_reliability[agent_id] = info


                    
                    elif int(split_line[1]) == 1:
                    
                        first_bracket = new_line.find('{')
                        second_bracket = new_line.rfind('}')
                
                        info = eval(new_line[first_bracket:second_bracket+1])
                        
                        if agent_id not in agent_model.keys():
                            agent_model[agent_id] = info
                            pass
                            #print("Model for agent " + agent_id + " at time " + str(split_line[0]), info)
                        else:
                            changed_model = []
                            for i in info.keys():
                                if agent_model[agent_id][i] != info[i]:
                                    changed_model.append((i,info[i]))
                                    
                            #print("Changed model for agent " + agent_id + " at time " + str(split_line[0]), changed_model)
                            agent_model[agent_id] = info
                            
                    
                    elif int(split_line[1]) == 2:
                    
                        if split_line[3] == 1:
                            pass
                            #print("Agent " + agent_id + " moved in the wrong direction")
                        elif split_line[3] == 2:
                            pass
                            #print("Agent " + agent_id + " arrived to its destination")
                        
                    new_line = log_file.readline()
                    
                
                d4 = "../ai_controller/log/" + dtime + "_" + analyzed_a + "_LLM.txt"
                log_file = open(d4)
                new_line = log_file.readline()
                while new_line:
                    
                    llm_response = eval(new_line)
                    
                    #print("Used LLM at time " + llm_response["time"])
                    
                    new_line = log_file.readline()
        
        if not previous_file:
            print("NUMBER OF OBJECTS IN GOAL", metrics["goal_objects"])  
            dangerous = []
            benign = []
            total_dangerous = []
            total_objects = []
            weights = []
            for m in metadata["metadata"]:
                if not m[0] and any(m[1] == object_id for object_id in metrics["goal_objects"]):
                    if m[5] == 1:
                        benign.append(m[1])
                    else:
                        dangerous.append(m[1])
                        
                    weights.append(m[4])
                
                if not m[0]:
                    total_objects.append(m[1])
                    if m[5] == 2:
                        total_dangerous.append(m[1])
            
            print("Dangerous:", len(dangerous), "Benign:", len(benign), "Total dangerous:", len(total_dangerous), "Total objects", len(total_objects))    
            print("Weights:", weights)    
            
            final_metrics["dangerous_objects"].append(len(dangerous))
            final_metrics["objects_carried"].append(len(dangerous)+len(benign))
            
            final_metrics["weight_distribution_carried"].append(weights)
            
            object_sensed_per_agent = {}
            
            for ob_id in total_objects:
                object_sensed_per_agent[ob_id] = 0
            
            for agent_id in objects_fully_sensed.keys():
                for ob_id in objects_fully_sensed[agent_id]:
                    object_sensed_per_agent[ob_id] += 1
            
            num_sensed = {i: 0 for i in range(5)}
            for ob_id in object_sensed_per_agent.keys():
                num_sensed[object_sensed_per_agent[ob_id]] += 1
               
            for time_sensed in num_sensed:
                num_sensed[time_sensed] /= len(total_objects)
                
            print("Number of times an object was sensed:", num_sensed)
            
            final_metrics["distribution_sensed"].append(num_sensed)
            
            total_distance_traveled = 0
            for ag_key in agents.keys():
                total_distance_traveled += agents[ag_key]["distance_traveled"]
            
            print("Total distance traveled:", total_distance_traveled)
            
            final_metrics["total_distance_traveled"].append(total_distance_traveled)
            
            for a_key in average_survey_results.keys():
                if average_survey_results[a_key] is not None:
                    try:
                        if a_key not in final_average_survey_results:
                            final_average_survey_results[a_key] = []
                        final_average_survey_results[a_key].append(sum(average_survey_results[a_key])/len(average_survey_results[a_key]))
                    except:
                        pdb.set_trace()
                    
            #print(final_average_survey_results)
            
for a_key in final_average_survey_results.keys():
    print(a_key,final_average_survey_results[a_key])


legends = ['Human leader, no suggestions', 'Human leader, with suggestions', 'AI leader, with suggestions', 'AI leader, no suggestions']
"""

group_final = [q for q in final_average_survey_results if q in final_questions]
group_other = [q for q in final_average_survey_results if q not in final_questions]
 
def plot_group(keys, title):
    n_questions = len(keys)
    n_bars = len(legends)  # now uses the length of your legends list
    indices = np.arange(n_questions)
    bar_width = 0.8 / n_bars

    # Build one series per legend entry
    values_matrix = [
        [final_average_survey_results[q][i] for q in keys]
        for i in range(n_bars)
    ]

    plt.figure(figsize=(10, 5))
    for i, vals in enumerate(values_matrix):
        plt.bar(
            indices + i * bar_width,
            vals,
            width=bar_width,
            label=legends[i]
        )

    if title == 'Other Questions':
        keys = [k.split(":")[0] for k in keys]
    plt.xticks(indices + bar_width * (n_bars - 1) / 2, keys, rotation=45, ha='right')
    plt.ylabel('Average Score')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
# 2. Make the two plots
plot_group(group_final, 'Final Questions')
plot_group(group_other, 'Other Questions')

plt.show()
"""
'''
# Derive category indices and labels from the first dict
cat_indices = sorted(final_metrics["distribution_sensed"][0].keys())
distribution_categories = [f"Sensed {i} times" for i in cat_indices]

# Grab the default colors
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(legends)]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
fig.subplots_adjust(hspace=0.4, wspace=0.3)

for idx, (metric, vals) in enumerate(final_metrics.items()):
    ax = axes[idx]
    
    if metric != "distribution_sensed":
        x = np.arange(len(legends))
        ax.bar(x, vals, width=0.6, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(legends, rotation=45, ha="right")
        
        if "dangerous_objects" in metric:
            metric = "Dangerous objects carried"
        
        ax.set_ylabel(metric.replace("_", " ").title())
        
        ax.set_title(metric.replace("_", " ").title())
    else:
        data_dicts = vals  # list of 4 dicts
        n_cat = len(cat_indices)
        n_leg = len(legends)
        x = np.arange(n_cat)
        bar_w = 0.8 / n_leg
        
        # For each legend i, extract its series across categories
        for i in range(n_leg):
            series = [data_dicts[i][ci] for ci in cat_indices]
            ax.bar(x + i*bar_w, series, width=bar_w, color=colors[i])
        
        # Draw separators between category groups
        for sep in x + bar_w*n_leg:
            ax.axvline(sep, color='grey', linestyle='--', linewidth=1, label='_nolegend_')
        
        ax.set_xticks(x + bar_w*(n_leg-1)/2)
        ax.set_xticklabels(distribution_categories, rotation=45, ha="right")
        ax.set_ylabel("Proportion Sensed")
        ax.set_title("Frequency Distribution of Object Sensing Events")

# Remove any unused axes
for j in range(len(final_metrics), len(axes)):
    fig.delaxes(axes[j])

# Shared legend in the middle, with solid boxes
handles = [Patch(facecolor=colors[i]) for i in range(len(legends))]
fig.legend(
    handles,
    legends,
    loc='center',
    bbox_to_anchor=(0.5, 0.52),
    ncol=len(legends),
    title="Scenario",
    frameon=False
)

plt.tight_layout()
plt.show()
'''
'''
weight_distribution_carried = final_metrics["weight_distribution_carried"]
# Determine all unique weight values across all legends
# 1. Find all unique weight categories
unique_weights = sorted({w for sub in weight_distribution_carried for w in sub})

# 2. Compute frequency of each weight for each legend
freqs_by_legend = []
for w_list in weight_distribution_carried:
    cnt = Counter(w_list)
    freqs_by_legend.append([cnt[w] for w in unique_weights])

# 3. Plot grouped bar chart
n_leg = len(legends)
x = np.arange(len(unique_weights))
bar_w = 0.8 / n_leg

plt.figure(figsize=(10, 6))
for i, freqs in enumerate(freqs_by_legend):
    plt.bar(x + i*bar_w, freqs, width=bar_w, label=legends[i])

plt.xticks(x + bar_w*(n_leg-1)/2, unique_weights, rotation=45, ha='right')
plt.xlabel("Weight")
plt.ylabel("Count")
plt.title("Weight Distribution Carried by Agent")
plt.legend(title="Agent")
plt.tight_layout()
plt.show()
'''

fig, ax = plt.subplots(figsize=(10, 6))
positions = np.arange(len(legends))
weight_distribution_carried = final_metrics["weight_distribution_carried"]
# 1. Violin plots (distribution shapes)
parts = ax.violinplot(
    weight_distribution_carried,
    positions=positions,
    showextrema=False,
    widths=0.7
)
for body in parts['bodies']:
    body.set_alpha(0.5)


# 3. Jittered scatter of individual data points
for i, data in enumerate(weight_distribution_carried):
    jittered_x = np.random.normal(positions[i], 0.04, size=len(data))
    ax.scatter(jittered_x, data, alpha=0.7, color='black', s=20)

# Formatting
ax.set_xticks(positions)
ax.set_xticklabels(legends, rotation=45, ha='right')
ax.set_xlabel("Scenario")
ax.set_ylabel("Carried Weight")
ax.set_title("Carried Weights per Scenario")

plt.tight_layout()
plt.show()
