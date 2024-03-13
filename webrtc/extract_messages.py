import glob
import pdb
import json
import numpy as np
from sklearn.cluster import DBSCAN

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
for d in glob.glob("20*.txt"):

    if "events" not in d:
        log_file = open(d)
        new_line = log_file.readline()
        messages_present = False
        session_log = []
        
        json_session = {}
        different_agents = []
        session_num = 0
        
        
        
        json_session["file"] = d
        json_session["content"] = {}
        distances = {}
        conversations = {0: []} #{}
        conversation_groups = {}
        last_time = {0: []} #{}
        
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
                    
                    if new_line[strings_extract[2]:strings_extract[3]]: #get recipients of message
                        recipients = new_line[strings_extract[2]:strings_extract[3]].split(',')[:-1]
                        
                        
                    for r_idx in range(len(received)):
                        if r_idx and r_idx < len(received)-1:
                            received_str += ", "
                        elif r_idx and r_idx == len(received)-1:
                            received_str += " and "
                            
                        received_str += "Agent " + received[r_idx]
                    
                    if not received_str:
                        received_str = "None"
                    
                    session_log.append([split_line[0],split_line[2],message,received])
                    
                    if split_line[2] not in different_agents:
                        different_agents.append(split_line[2])
       
                    
                    mins, remainder = divmod(float(split_line[0]), 60)
                    secs,millisecs = divmod(remainder,1)
                    time_formatted = '{:02d}:{:02d}'.format(int(mins), int(secs))
                    #print("Message sent by Agent " + split_line[2] + " at time " + time_formatted + " and received by " + received_str + ": " + message)
                    #print("{Sender: Agent " + split_line[2] + "| Time: " + time_formatted + "| Recipients: " + received_str + "| Message: " + message + "}")
                    
                    #message_str = "{Sender: Agent " + split_line[2] + "| Time: " + time_formatted + "| Message: " + message + "}"
                    
                    group_key = 0
                    for cg in conversation_groups.keys():
                        if split_line[2] in conversation_groups[cg]:
                            group_key = cg
                    
                    if conversations[group_key][-1] and float(split_line[0]) - last_time[group_key] > 30:
                        conversations[group_key].append([])
                     
                    #message_str = "{Number: " + str(len(conversations[a_key][-1])+1) + " | Sender: Agent " + split_line[2] + " | Message: " + message + "}"
                    message_str = {"Number":len(conversations[group_key][-1])+1, "Sender": "Agent " + split_line[2], "Message": message}
                    conversations[group_key][-1].append(message_str)   
                    last_time[group_key] = float(split_line[0])
                    """
                    for a_key in [split_line[2],*received]:

                        if a_key not in conversations:
                            last_time[a_key] = 0
                            conversations[a_key] = [[]]

                        if conversations[a_key][-1] and float(split_line[0]) - last_time[a_key] > 30:
                            conversations[a_key].append([])
                            
                            
                        #message_str = "{Number: " + str(len(conversations[a_key][-1])+1) + " | Sender: Agent " + split_line[2] + " | Message: " + message + "}"
                        message_str = {"Number":len(conversations[a_key][-1])+1, "Sender": "Agent " + split_line[2], "Message": message}
                        conversations[a_key][-1].append(message_str)
                        last_time[a_key] = float(split_line[0])
                    """
                elif int(split_line[1]) == 4:
                
                    if session_log:
                        json_session["content"][session_num] = session_log.copy()
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
                    different_agents = []
                    distances = {}
                    
                    pdb.set_trace()
                    for c_key in conversations.keys():
                        for d in conversations[c_key]:
                            for t_idx,t in enumerate(d):
                                print(t)
                            
                            print("")
                    
                    """
                    for c_key in conversations.keys():
                        print("Messages with " + c_key)
                        
                        
                        
                        
                        
                        for d in conversations[c_key]:
                            history_messages = []
                            message_log = {"messages":[preamble]}
                            print("")
                            for t_idx,t in enumerate(d):
                                #print(t)
                                
                                if t_idx:
                                    messg = "" 
                                    
                                    messg += "History of messages: " + str(history_messages) + "\n"
                                    messg += "New message: " + str(t) + "\n"
                                    messg += "Who is " + t["Sender"] + " replying to? Output a JSON.\n"
                                    message_log["messages"].append(messg)
                                    
                                history_messages.append(t)
                                
                            print(json.dumps(message_log))
                                
                        print("")                                
                    """
                    #conversations = {}
                    #last_time = {}
                    
                    conversations = {0: []} #{}
                    conversation_groups = {}
                    last_time = {0: []} #{}
                    messages_present = False
                
                
                
                elif int(split_line[1]) == 0:     
                    m_idx1 = new_line.index('{')-1
                    m_idx2 = new_line.index('}')+2
        
            
                    metadata = json.loads(json.loads(new_line[m_idx1:m_idx2]))
                    
                    positions = {}
                    disabled = {}
                    
                    for m in metadata["metadata"]:
                        if m[0]:
                            positions[m[1]] = [float(m[2]),float(m[3])]
                            if m[1] not in distances.keys():
                                distances[m[1]] = {}
                            
                            disabled[m[1]] = bool(m[4])
                            
                    poses = [list(positions.keys()),list(positions.values())]
                    db = DBSCAN(eps=communication_distance, min_samples=1).fit(np.array(poses[1]))
                    
                    conversation_groups_copy = conversation_groups.copy()
                    conversation_groups = {}
                    
                    for l_idx,label in enumerate(db.labels_):
                    
                        if label not in conversation_groups.keys():
                            conversation_groups[label] = []
                            conversations[label] = [[]]
                            last_time[label] = 0
                            
                        conversation_groups[label].append(poses[0][l_idx])    
                                
                    
                    for c_key in conversation_groups_copy.keys():
                    
                    
                        if set(conversation_groups_copy[c_key]) != set(conversation_groups[c_key]):
                            if conversations[c_key]:
                                conversations[c_key].append([])
                    
                    print("")
                    
                """    
                    for m in distances.keys():
                        for m2 in distances.keys(): 
                            if m == m2:
                                continue
                            
                            try:
                                disabled[m2] or disabled[m]
                            except:
                                pdb.set_trace()
                            
                            if disabled[m2] or disabled[m]:
                                distances[m][m2] = False
                            else:
                                if np.linalg.norm(np.array(positions[m]) - np.array(positions[m2])) < communication_distance*3:
                                    distances[m][m2] = True
                                else:
                                    distances[m][m2] = False
                """
                
            
            new_line = log_file.readline()
            

        if session_log:
            json_session["content"][session_num] = session_log.copy()
            
            if len(session_log) > 10 and len(different_agents) <= 4:
                total_sessions += 1
                if len(different_agents) not in good_sessions.keys():
                    good_sessions[len(different_agents)] = 0
                    good_names[len(different_agents)] = []
                good_sessions[len(different_agents)] += 1
                good_names[len(different_agents)].append(json_session["file"])
            
        if json_session["content"]:
            json_file.append(json_session)
            
#print(json_file)
out_file = open("message_logs.json", "w") 
json.dump({"logs":json_file}, out_file)
out_file.close()

print(total_sessions)
print(good_sessions)
print(good_names)
