import glob
import pdb
import json

json_file = []
total_sessions = 0
for d in glob.glob("*.txt"):

    if "events" not in d:
        log_file = open(d)
        new_line = log_file.readline()
        messages_present = False
        session_log = []
        
        json_session = {}
        session_num = 0
        
        
        json_session["file"] = d
        json_session["content"] = {}
        
        while new_line:
            split_line = new_line.strip().split(',')
            
            try:
                if int(split_line[1]) == 2:
                    if not messages_present:
                        print(d)
                        messages_present = True
                        session_num += 1
                        
                    
                    last_q = new_line.rfind('"')
                    second_last_q = new_line[:last_q].rfind('"')
                    third_last_q = new_line[:second_last_q].rfind('"')
                    first_q = new_line.index('"')
                    
                    message = new_line[first_q:third_last_q+1]
                        
                    #print(new_line.strip())
                    session_log.append([split_line[0],split_line[2],message])
                elif int(split_line[1]) == 4:
                
                    if session_log:
                        json_session["content"][session_num] = session_log.copy()
                        total_sessions += 1
                
                    session_log = []
                    messages_present = False
                    
            except:
                pdb.set_trace()
        
            
            new_line = log_file.readline()
            

        if session_log:
            json_session["content"][session_num] = session_log.copy()
            total_sessions += 1
            
    if json_session["content"]:
        json_file.append(json_session)
            
#print(json_file)
out_file = open("message_logs.json", "w") 
json.dump({"logs":json_file}, out_file)
out_file.close()

print(total_sessions)
