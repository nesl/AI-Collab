import glob
import pdb
import json
import numpy as np
from sklearn.cluster import DBSCAN
import re
import pandas as pd
import json

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

print(glob.glob("*.txt"))

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

    return scenario_options



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


def pandas_process(filtering_column, condition_column, metric_column):

    performance = {}

    for a in df.groupby(filtering_column)[[condition_column, metric_column]]:
        if a[1].iloc[0][condition_column] not in performance:
            performance[a[1].iloc[0][condition_column]] = []
            
        performance[a[1].iloc[0][condition_column]].append(a[1].iloc[0][metric_column])
        
        #print(a)
    
    for ps in performance.keys():
        print(ps, sum(performance[ps])/len(performance[ps]))


agent_info = []
agent_info_columns = []
message_logs = {}
human_names = {}

stats_questions = ['distance_traveled', 'grabbed_objects', 'grab_attempts', 'dropped_outside_goal', 'objects_sensed', 'sensor_activation', 'objects_in_goal', 'dangerous_objects_in_goal', 'num_messages_sent', 'average_message_length', 'time_with_teammates_A', 'time_with_teammates_B','time_with_teammates_C','time_with_teammates_D', 'end_time', 'team_dangerous_objects_in_goal', 'total_dangerous_objects', 'quality_work', 'effort', 'human_team_effort', 'team_end_time', 'team_quality_work', 'team_speed_work', 'team_achievement', "task_completion", "team_completion"]

for d in glob.glob("20*.txt"):

    if "events" not in d:
        log_file = open(d)
        new_line = log_file.readline()
        try:
            log_state_file = open("../../simulator/log/" + d[:-4] + "_state.txt")
        except:
            print("Could not open file", d)
            continue
        scenario_options = get_settings(log_state_file)
        just_reset = []
        
        num_humans = 0
        for agent in scenario_options["robots_type"]:
            if agent[1] == "human":
                num_humans += 1
        
        connected_humans = 0
        json_session = reset_json_session(d)
        session_num = 0
        survey_ready = False
        token_ready = 0

        survey_results = []
        session_log = []
        
        team_strategy_agents = {"questions":["Strategy", "Substrategy", "Role"], "answers":{}}

        print(d)
        
        while new_line:
            split_line = new_line.strip().split(',')
            
            if True:
            
                if int(split_line[1]) == 2:
                    if not messages_present:
                        print("Reset")
                        messages_present = True
                        session_num += 1
                        
                    
                    
                    last_q = new_line.rfind('"')
                    second_last_q = new_line[:last_q].rfind('"')
                    third_last_q = new_line[:second_last_q].rfind('"')
                    first_q = new_line.index('"')
                    
                    strings_extract = [first_q+1,third_last_q,second_last_q+1,last_q]
                    
                    message = new_line[strings_extract[0]:strings_extract[1]]
                    
                    received = []
                    if new_line[strings_extract[2]:strings_extract[3]]: #get recipients of message
                        received = new_line[strings_extract[2]:strings_extract[3]].split(',')[:-1]
                        
                    #print(new_line.strip())
                    
                    
                    
                    received_str = ""
                    
                    """
                    if split_line[2] in distances:
                        for d_idx,d in enumerate(distances[split_line[2]].keys()):
                            if distances[split_line[2]][d]:
                                received.append(d)
                    """
                    
                    #if new_line[strings_extract[2]:strings_extract[3]]: #get recipients of message
                    #    recipients = new_line[strings_extract[2]:strings_extract[3]].split(',')[:-1]
                        
                        
                    for r_idx in range(len(received)):
                        if r_idx and r_idx <= len(received)-1:
                            received_str += ", "
                        elif r_idx and r_idx == len(received)-1:
                            received_str += " and "
                            
                        received_str += "Agent " + received[r_idx]
                    
                    if not received_str:
                        received_str = "None"
                    
                    regex_str = "My goal is (\w+) (\(-?\d+\.\d+,-?\d+\.\d+\)), I'm moving towards (\(-?\d+\.\d+,-?\d+\.\d+\)). My current location is (\(-?\d+\.\d+,-?\d+\.\d+\)).( Carrying object (\w+).)?( Helping (\w+).)?"
                    
                    rematch = re.search(regex_str,message)
                    if rematch:
                        message = message.replace(rematch.group(0), "")
                        
                    if message.strip():
                        session_log.append({"time": split_line[0], "sender": "Agent " + split_line[2],"message":message,"received by":received_str})
                    
                    
                elif int(split_line[1]) == 4:
                
                    
                    log_line_agent = split_line[2]
            
                    just_reset.append(log_line_agent)
                    #if all(robot[0] in just_reset for robot in scenario_options["robots_type"] if robot[1] == "human"): #Check robot is real
                    if all(robot[0] in just_reset for robot in scenario_options["robots_type"]): 
            
                        #print("Resetting")
                        scenario_options_tmp = get_settings(log_state_file)
                        
                        if scenario_options_tmp:
                            scenario_options = scenario_options_tmp
                            
                        just_reset = []
                        
                        first_bracket = new_line.find('{')
                        second_bracket = new_line.rfind('}')
                        
                        newer_line = new_line[first_bracket:second_bracket+1].replace("false", "False").replace("true","True")
                        
                        team_strategy = eval(newer_line)
                        
                        type_subrole = [[0,0],[0,0]]
                        
                        for agent in scenario_options["robots_type"]:
                            if agent[1] == "human":
                                if team_strategy["role"][agent[0]] == "sensing" or team_strategy["hierarchy"][agent[0]] == "order" or team_strategy["interdependency"][log_line_agent] == "followed":
                                    type_subrole[0][0] += 1
                                else:
                                    type_subrole[0][1] += 1
                            else:
                                if team_strategy["role"][agent[0]] == "sensing" or team_strategy["hierarchy"][agent[0]] == "order" or team_strategy["interdependency"][log_line_agent] == "followed":
                                    type_subrole[1][0] += 1
                                else:
                                    type_subrole[1][1] += 1
                        
                        if team_strategy["role"][log_line_agent] != "equal":
                            type_strategy = "role"
                            
                            if type_subrole[0][0] == sum(type_subrole[0]) and type_subrole[1][1] == sum(type_subrole[1]):
                                substrategy = "Human-only sensing"
                            elif type_subrole[0][1] == sum(type_subrole[0]) and type_subrole[1][0] == sum(type_subrole[1]):
                                substrategy = "Human-only lifting"
                            else:
                                substrategy = "Mixed"
                            
                            particular_role = team_strategy["role"][log_line_agent]
                        elif team_strategy["hierarchy"][log_line_agent] != "equal":
                            type_strategy = "hierarchy"    
                            
                            if type_subrole[0][0] == sum(type_subrole[0]) and type_subrole[1][1] == sum(type_subrole[1]):
                                substrategy = "Human-only leaders"
                            elif type_subrole[0][1] == sum(type_subrole[0]) and type_subrole[1][0] == sum(type_subrole[1]):
                                substrategy = "AI-only leaders"
                            else:
                                substrategy = "Mixed"
                            
                            particular_role = team_strategy["hierarchy"][log_line_agent]
                        elif team_strategy["interdependency"][log_line_agent] != "equal":
                            type_strategy = "interdependency"
                            
                            if type_subrole[0][0] == 1:
                                substrategy = "Single unit"
                            elif type_subrole[0][0] + type_subrole[1][0] == len(scenario_options["robots_type"]):
                                substrategy = "Individuals"
                            else:
                                substrategy = "Multiple units"
                            
                            particular_role = "independent"   
                        
                        
                        
                        for robot in scenario_options["robots_type"]: 
                        
                            if robot[1] == "human":
                        
                                if team_strategy["role"][robot[0]] != "equal":
                                    particular_role = team_strategy["role"][robot[0]]
                                elif team_strategy["hierarchy"][robot[0]] != "equal":
                                    particular_role = team_strategy["hierarchy"][robot[0]]
                                elif team_strategy["interdependency"][robot[0]] != "equal":
                                    particular_role = "independent"
                                
                                team_strategy_agents["answers"][robot[0]] = [type_strategy, substrategy, particular_role]
                        
                       
                            
                        
                    
                    
                elif int(split_line[1]) == 5:
                    first_bracket = new_line.find('{')
                    second_bracket = new_line.rfind('}')
                    
                    stats = eval(new_line[first_bracket:second_bracket+1])
                    
                    agent_id = split_line[2]
                    
                    if stats["token"] and stats["total_dangerous_objects"] and not survey_ready:
                        json_session["users"][agent_id]["token"] = stats["token"]
                        
                        token_ready += 1
                        
                        stats_answers = []
                        
                        
                        for sq in stats_questions:
                            
                            if sq == "dropped_outside_goal" or sq == "objects_in_goal" or sq == 'dangerous_objects_in_goal':
                                ans = len(stats[sq])
                            elif sq == "task_completion":
                                ans = len(stats["dangerous_objects_in_goal"]) / stats["total_dangerous_objects"]
                                
                            elif sq == "team_completion":
                                ans = stats["team_dangerous_objects_in_goal"]/stats["total_dangerous_objects"]
                            elif 'time_with_teammates' in sq:
                                ans = stats['time_with_teammates'][sq[-1]] if sq[-1] in stats['time_with_teammates'].keys() else None
                            elif sq == 'num_messages_sent':
                                ans = len([True for m in session_log if m["sender"] == "Agent " + agent_id])
                            elif sq == "average_message_length":
                                message_list = [len(m["message"]) for m in session_log if m["sender"] == "Agent " + agent_id]
                                
                                ans = sum(message_list)/len(message_list)
                            else:
                                ans = stats[sq]
                                    
                            stats_answers.append(ans)
                        
                        json_session["users"][agent_id]["stats"] = {"questions": stats_questions,"answers":stats_answers}
                        
                   
                elif int(split_line[1]) == 7:
                    first_bracket = new_line.find('[')
                    second_bracket = new_line.find(']')
                    third_bracket = new_line.rfind('[')
                    fourth_bracket = new_line.rfind(']')
                    
                    questions = eval(new_line[first_bracket:second_bracket+1])
                    answers = eval(new_line[third_bracket:fourth_bracket+1])
                    
                    agent_id = chr(ord('A') + int(split_line[2]) - 1)
                    
                    
                    if not answers[0].strip():
                        print("Error, no name")
                        name_id = -1
                    else:
                        if answers[0].strip() not in human_names.keys():
                            human_names[answers[0].strip()] = len(human_names.keys())
                    
                        name_id = human_names[answers[0].strip()]
                    
                    json_session["users"][agent_id] = {"demographics": {"questions": questions, "answers": [name_id, *[int(a) if a else None for a in answers[1:]]]}, "survey": {}}
                    
                    connected_humans += 1
                    
                    
                elif int(split_line[1]) == 8:
                
                    first_bracket = new_line.find('[')
                    second_bracket = new_line.find(']')
                    third_bracket = new_line.rfind('[')
                    fourth_bracket = new_line.rfind(']')
                    
                    last_comma = new_line.rfind(',')
                    
                    questions = eval(new_line[first_bracket:second_bracket+1])
                    answers = eval(new_line[third_bracket:fourth_bracket+1])
                    
                    survey_results.append({"token":new_line[last_comma+1:].strip(), "questions": questions, "answers": [int(a) if a else None for a in answers]})
                    
                    
         
                
            
            new_line = log_file.readline()
            
        
            if connected_humans == num_humans and len(survey_results) == num_humans and token_ready == num_humans:
                json_session = match_survey(json_session, survey_results)
                
                date = d[:d.rfind(".")]
                
                agent_type = {}
                
                agent_type_count = [0,0]
                
                for agent in scenario_options["robots_type"]:
                    agent_type[agent[0]] = agent[1]
                    
                    if agent[1] == "ai":
                        agent_type_count[0] += 1
                    elif agent[1] == "human":
                        agent_type_count[1] += 1
                
                if agent_type_count[0] > agent_type_count[1]:
                    team_composition = "Majority AI"
                elif agent_type_count[0] == agent_type_count[1]:
                    team_composition = "Equal"
                else:
                    team_composition = "Majority Human"
                
                for user in json_session["users"].keys():
                    agent_info.append([date, user, agent_type[user], len(scenario_options["robots_type"]), team_composition, *team_strategy_agents["answers"][user], *json_session["users"][user]["demographics"]["answers"], *json_session["users"][user]["survey"]["answers"], *json_session["users"][user]["stats"]["answers"]])
                    
                if not agent_info_columns:
                    agent_info_columns = ["Date", "Agent", "Type", "Team Size", "Team Composition", *team_strategy_agents["questions"], *json_session["users"][user]["demographics"]["questions"], *json_session["users"][user]["survey"]["questions"], *json_session["users"][user]["stats"]["questions"]]
                
                json_file.append(json_session)
                survey_results = []
                json_session = reset_json_session(d)
                survey_ready = True
                
                message_logs[date] = session_log.copy()
            

print(len(json_file))

df = pd.DataFrame(agent_info, columns=agent_info_columns)

df.to_csv('stats.csv')  
with open('messages.json', 'w') as f:
    json.dump(message_logs, f)

pandas_process("Date", "Substrategy", "team_completion")
pandas_process("Date", "Team Composition", "team_completion")
print("")
pandas_process("Date", "Substrategy", "AI agents performed well") #AI agents are reliable teammates
pandas_process("Date", "Team Composition", "AI agents performed well")

print("")
pandas_process("Date", "Substrategy", "AI agents are reliable teammates") #AI agents are reliable teammates
pandas_process("Date", "Team Composition", "AI agents are reliable teammates")
#I was able to trust the AI agents

#pdb.set_trace()
#print(df)       

#.mean()
#
'''
#print(json_file)
out_file = open("message_logs.json", "w") 
json.dump({"logs":json_file}, out_file)
out_file.close()

out_file = open("message_gpt.json", "w")
json.dump(all_messages, out_file)
out_file.close()

print(total_sessions)
print(good_sessions)
print(good_names)
'''
