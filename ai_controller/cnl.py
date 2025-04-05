import pdb
import re
import numpy as np

class MessagePattern:
    @staticmethod
    def location(goal_x,goal_y,next_loc_x,next_loc_y,convert_to_real_coordinates, current_location, carrying, helping, object_id):
        real_goal = convert_to_real_coordinates([goal_x,goal_y])
        real_next_location = convert_to_real_coordinates([next_loc_x,next_loc_y])
        real_current_location = convert_to_real_coordinates(current_location)
        output_string = "My goal is " + str(object_id) + " (" + str(real_goal[0]) + "," + str(real_goal[1]) + "), I'm moving towards (" + str(real_next_location[0]) + "," + str(real_next_location[1]) + "). My current location is (" + str(real_current_location[0]) + "," + str(real_current_location[1]) + "). "
        
        if not real_goal or not real_next_location or not real_current_location:
            pdb.set_trace()
        
        if carrying:
            output_string += "Carrying object " + str(carrying) + ". "
            
        if helping:
            output_string += "Helping " + str(helping[0]) + ". "
            
            
        return output_string
            
        
    @staticmethod
    def location_regex():
        return "My goal is (\w+) (\(-?\d+\.\d+,-?\d+\.\d+\)), I'm moving towards (\(-?\d+\.\d+,-?\d+\.\d+\)). My current location is (\(-?\d+\.\d+,-?\d+\.\d+\)).( Carrying object (\w+).)?( Helping (\w+))?"
        
    @staticmethod
    def item(robotState,item_idx,object_id, info, robot_id, convert_to_real_coordinates):

        """ Example
        Object 1 (weight: 1) Last seen in (5.5,5.5) at 00:57. Status: benign, Prob. Correct: 88.1%
        """
    
        message_text = ""
                            
        if robotState.get("objects", "weight", item_idx):
                            
            item_loc = robotState.get("objects", "last_seen_location", item_idx)
        
            mins, remainder = divmod(robotState.get("objects", "last_seen_time", item_idx), 60)

            secs,millisecs = divmod(remainder,1)
            
            
            time_formatted = '{:02d}:{:02d}'.format(int(mins), int(secs))
            
            
            if not (item_loc[0] == -1 and item_loc[1] == -1):
                real_location = convert_to_real_coordinates(item_loc)
            else:
               real_location = [99.99,99.99]
        
            message_text = "Object " + str(object_id) + " (weight: " +  str(robotState.get("objects", "weight", item_idx)) + ") Last seen in (" + str(real_location[0]) + "," + str(real_location[1]) + ") at " + time_formatted + ". "
                
            status_danger = ""
            prob_correct = ""   
            from_estimates = ""        
            for robo_idx in range(robotState.get_num_estimates(item_idx)):     
            
                roboestimate = robotState.get("object_estimates", "danger_status", [item_idx,robo_idx])
               
                if roboestimate > 0:
                    
                    if not status_danger:                
                        status_danger +=  "Status: ["
                        prob_correct += "Prob. Correct: ["
                        from_estimates += "From: ["
                    else:
                        status_danger += ","
                        prob_correct += ","
                        from_estimates += ","
                        
                    if roboestimate == 1:
                        status_danger += "benign"
                    else:
                        status_danger += "dangerous"
                        
                    try:
                        prob_correct += str(round(robotState.get("object_estimates", "estimate_correct_percentage", [item_idx,robo_idx])*100,1)) + "%"
                    except:
                        pdb.set_trace()
                    
                    if robo_idx == robotState.get_num_estimates(item_idx)-1:
                        sensing_robot = robot_id
                    else:
                        sensing_robot = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(robo_idx)]
                    
                    from_estimates += str(sensing_robot)
        

            if status_danger:
                status_danger += "], "
                prob_correct += "], "
                from_estimates += "]. "
                
            message_text += status_danger + prob_correct + from_estimates

        return message_text
    
    @staticmethod
    def translate_item_message(message,robot_id):
    
        message_text = ""
        
        for rematch in re.finditer(MessagePattern.item_regex_full_alt(),message):

            if rematch.group(5):
                old_string = rematch.group()
                new_string = " Status: [" + rematch.group(6) + "], Prob. Correct: [" + rematch.group(7) + "%], From: [" + robot_id + "]"
                message_text += old_string.replace(rematch.group(5), new_string)
                
        
        return message_text
        
    @staticmethod
    def item_regex_partial():
        return "Object (\d+) \(weight: (\d+)\) Last seen in (\(-?\d+\.\d+,-?\d+\.\d+\)) at (\d+:\d+)"
        
    @staticmethod
    def item_regex_full():
        return "Object (\d+) \(weight: (\d+)\) Last seen in (\(-?\d+\.\d+,-?\d+\.\d+\)) at (\d+:\d+).( Status: (\[\w+(,\w+)*\]), Prob. Correct: (\[\d+\.\d+%(,\d+\.\d+%)*\]), From: (\[\w+(,\w+)*\]))?"#"Object (\d+) \(weight: (\d+)\) Last seen in (\(-?\d+\.\d+,-?\d+\.\d+\)) at (\d+:\d+).( Status: (\w+), Prob. Correct: (\d+\.\d+)%)?"
        
    @staticmethod
    def item_regex_full_alt():
        return "Object (\d+) \(weight: (\d+)\) Last seen in (\(-?\d+\.\d+,-?\d+\.\d+\)) at (\d+:\d+).( Status: (\w+), Prob. Correct: (\d+\.\d+)%)?"
    
    @staticmethod
    def sensing_help(object_id):
        return "What do you know about object " + str(object_id) + ". "
        
    @staticmethod
    def sensing_help_regex():
        return "What do you know about object (\d+)" 
        
    @staticmethod
    def sensing_ask_help(robotState, item_idx, object_id, robot_id, convert_to_real_coordinates):
    
        item_loc = robotState.get("objects", "last_seen_location", item_idx)
    
        if not (item_loc[0] == -1 and item_loc[1] == -1):
            real_location = convert_to_real_coordinates(item_loc)
        else:
           real_location = [99.99,99.99]
           
        mins, remainder = divmod(robotState.get("objects", "last_seen_time", item_idx), 60)
        secs,millisecs = divmod(remainder,1)
        time_formatted = '{:02d}:{:02d}'.format(int(mins), int(secs))
    
        return "Hey " + str(robot_id) + ", can you help me sense object " + str(object_id) + " in location (" + str(real_location[0]) + "," + str(real_location[1]) + "), last seen at " + time_formatted + ". "
        
    @staticmethod
    def sensing_ask_help_regex():
        return "Hey (\w+), can you help me sense object (\d+) in location (\(-?\d+\.\d+,-?\d+\.\d+\)), last seen at (\d+:\d+)"
        
    @staticmethod
    def sensing_ask_help_confirm(robot_id, object_id):
        return "Yes, I can help you sense object " + str(object_id) + ", " + str(robot_id) + ". "
        
    @staticmethod
    def sensing_ask_help_confirm_regex():
        return "Yes, I can help you sense object (\d+), (\w+)"
        
    @staticmethod
    def sensing_ask_help_reject(robot_id):
        return "No, I cannot help you sense, " + str(robot_id) + ". "
        
    @staticmethod
    def sensing_ask_help_reject_regex():
        return "No, I cannot help you sense, (\w+)" 
        
    @staticmethod
    def sensing_ask_help_incorrect(robot_id):
        return "I didn't offer my help to you " + str(robot_id) + ". "
        
    @staticmethod
    def sensing_ask_help_incorrect_regex():
        return "I didn't offer my help to you (\w+)"     
        
    @staticmethod    
    def sensing_help_negative_response(object_id):
        return "I know nothing about object " + str(object_id) + ". "
        
    @staticmethod    
    def sensing_help_negative_response_regex():
        return "I know nothing about object (\d+)"
        
    @staticmethod
    def object_not_found(robot_id, object_id):
        return "Hey " + str(robot_id) + ", I didn't find object " + str(object_id) + ". "
        
    @staticmethod    
    def object_not_found_regex():
        return "Hey (\w+), I didn't find object (\d+)"

    @staticmethod
    def ask_for_agent(robot_id):
        return "Where is agent " + str(robot_id) + "? "
        
    @staticmethod
    def ask_for_agent_regex():
        return "Where is agent (\w+)"
        
    @staticmethod
    def agent_not_found(robot_id):
        return "I don't know where is agent " + str(robot_id) + ". "
        
    @staticmethod
    def agent_not_found_regex():
        return "I don't know where is agent (\w+)"
        
    @staticmethod
    def agent(robot_id, robo_idx, robotState, convert_to_real_coordinates):
    
        message_text = ""
                      
        robot_loc = robotState.get("agents", "last_seen_location", robo_idx)
              
        mins, remainder = divmod(robotState.get("agents", "last_seen_time", robo_idx), 60)

        secs,millisecs = divmod(remainder,1)
          
        if not (robot_loc[0] == -1 and robot_loc[1] == -1):
            real_location = convert_to_real_coordinates(robot_loc)
        else:
            real_location = [99.99,99.99]
            
        time_formatted = '{:02d}:{:02d}'.format(int(mins), int(secs))
         
        if robotState.get("agents", "type", robo_idx) == 1:
            robot_type = "ai"
        elif not robotState.get("agents", "type", robo_idx):
            robot_type = "human"
        
        message_text = "Agent " + str(robot_id) + " (type: " +  robot_type + ") Last seen in (" + str(real_location[0]) + "," + str(real_location[1]) + ") at " + time_formatted + ". "


        return message_text
        
    @staticmethod
    def agent_regex():
        return "Agent (\w+) \(type: (\w+)\) Last seen in (\(-?\d+\.\d+,-?\d+\.\d+\)) at (\d+:\d+)"

    @staticmethod
    def surroundings(xy, view_radius, robotState, info, convert_to_real_coordinates):
    
        real_location = convert_to_real_coordinates(xy)
    
        message_text = "Scanned area at (" + str(real_location[0]) + "," + str(real_location[1]) + ") with view radius " + str(view_radius) + ". "
        
        x_min = max(0,xy[0]-view_radius)
        y_min = max(0,xy[1]-view_radius)
        x_max = min(robotState.latest_map.shape[0]-1,xy[0]+view_radius)
        y_max = min(robotState.latest_map.shape[1]-1,xy[1]+view_radius)
        
        local_map = robotState.latest_map[x_min:x_max+1,y_min:y_max+1]
        
        walls = np.where(local_map == 1)

        if walls[0].size > 0:
            message_text += "Walls: ["
            for w in range(len(walls[0])):
                if w:
                    message_text += ","
                real_location = convert_to_real_coordinates([x_min+walls[0][w],y_min+walls[1][w]])
                message_text += "(" + str(real_location[0]) + "," + str(real_location[1]) + ")"
            
            message_text += "]. "
        
        objects = []        
        for it_idx in range(robotState.get_num_objects()):
        
            ob_location = robotState.get("objects", "last_seen_location", it_idx)
        
            if ob_location[0] >= x_min and ob_location[0] <= x_max and ob_location[1] >= y_min and ob_location[1] <= y_max:
                real_location = convert_to_real_coordinates(ob_location)
                object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(it_idx)]
                objects.append((object_id, real_location))
                
        if objects:
            message_text += "Objects: {"
            
            for ob_idx,ob in enumerate(objects):
                if ob_idx:
                    message_text += ","
                message_text += str(ob[0]) + ":" + "(" + str(ob[1][0]) + "," + str(ob[1][1]) + ")"
        
            message_text += "}. "
            
        return message_text
        
    @staticmethod
    def surroundings_regex():
        return "Scanned area at (\(-?\d+\.\d+,-?\d+\.\d+\)) with view radius (\d+).( Walls: (\[(,?\(-?\d+\.\d+,-?\d+\.\d+\))+\]).)?( Objects: (\{(,?\d+:\(-?\d+\.\d+,-?\d+\.\d+\))+\}))?"

    @staticmethod
    def carry_help(object_id, num_robots):
        return "I need " + str(num_robots) + " more robots to help carry object " + str(object_id) + ". "
        
    @staticmethod
    def carry_help_regex():
        return "I need (\d+) more robots to help carry object (\d+)"
        
    @staticmethod
    def carry_help_accept(robot_id):
        return "I can help you " + str(robot_id) + ". Let me know if I should follow you or you want to follow me. "
        
    @staticmethod
    def carry_help_accept_regex():
        return "I can help you (\w+)"
        
    @staticmethod
    def come_closer(robot_id):
        return "Come closer " + str(robot_id) + ". "
        
    @staticmethod
    def come_closer_regex():
        return "Come closer (\w+)"    
        
    @staticmethod
    def carry_help_participant_reject(robot_id):
        return "I cannot help you " + str(robot_id) + ". "
        
    @staticmethod
    def carry_help_participant_reject_regex():
        return "I cannot help you (\w+)"
        
    @staticmethod
    def carry_help_participant_affirm(robot_id):
        return "I'm already helping you " + str(robot_id) + ". "    
    
    @staticmethod
    def carry_help_participant_affirm_regex():
        return "I'm already helping you (\w+)"
        
    @staticmethod
    def carry_help_participant_asking(robot_id):
        return "I'm asking you " + str(robot_id) + ". "    
    
    @staticmethod
    def carry_help_participant_asking_regex():
        return "I'm asking you (\w+)"
        
    @staticmethod
    def carry_help_participant_affirm_being_helped(robot_id):
        return "You are helping me " + str(robot_id) + ". "    
    
    @staticmethod
    def carry_help_participant_affirm_being_helped_regex():
        return "You are helping me (\w+)"    
        
        
    @staticmethod
    def carry_help_participant_reject_other(robot_id):
        return "I'm already helping " + str(robot_id) + ". "    
    
    @staticmethod
    def carry_help_participant_reject_other_regex():
        return "I'm already helping (\w+)"    
        
    @staticmethod
    def carry_help_participant_reject_helping(robot_id):
        return str(robot_id) + " is helping me. "    
    
    @staticmethod
    def carry_help_participant_reject_helping_regex():
        return "(\w+) is helping me" 
        
    @staticmethod
    def carry_help_participant_reject_object():
        return "I'm carrying an object. "    
        
    @staticmethod
    def carry_help_participant_reject_asking():
        return "I'm asking for help. " 
    
    @staticmethod
    def carry_help_participant_affirm_other_regex():
        return "I'm already helping (\w+)"
        
    @staticmethod
    def carry_help_reject(robot_id):
        return "Nevermind " + str(robot_id) + ". "
        
    @staticmethod
    def carry_help_reject_regex():
        return "Nevermind (\w+)"
        
    @staticmethod
    def carry_help_finish():
        return "No need for more help. "
        
    @staticmethod
    def carry_help_cancel():
        return "Nevermind. "
        
    @staticmethod
    def carry_help_complain(): #(robot_id):
        return "Thanks for nothing. " # " + str(robot_id)
        
    @staticmethod
    def carry_help_complain_regex():
        return "Thanks for nothing (\w+)"
        
    @staticmethod
    def follow(robot_id):
        return "Thanks, follow me " + str(robot_id) + ". Stay close to me. "
        
    @staticmethod
    def follow_regex():
        return "Thanks, follow me (\w+)"
        
    @staticmethod
    def following(robot_id):
        return "Thanks, I'll follow you " + str(robot_id) + ". Stay close to me. "
        
    @staticmethod
    def following_regex():
        return "Thanks, I'll follow you (\w+)"
        
    @staticmethod
    def follow_response(robot_id):
        return "I'll follow you " + str(robot_id) + " until you tell me you don't need my help anymore. Please don't move too far away from me so I don't lose track. "
        
    @staticmethod
    def follow_response_regex():
        return "I'll follow you (\w+) until you tell me you don't need my help anymore. Please don't move too far away from me so I don't lose track"
        
    @staticmethod
    def following_response(robot_id):
        return "Follow me " + str(robot_id) + " and tell me when you don't need my help anymore. "
        
    @staticmethod
    def following_response_regex():
        return "Follow me (\w+) and tell me when you don't need my help anymore"
        
    @staticmethod
    def wait(robot_id):
        return "I'm going to wait for " + str(robot_id) + " to pass. " 
        
    @staticmethod
    def wait_regex():
        return "I'm going to wait for (\w+) to pass"
        
        
    @staticmethod
    def move_request(robot_id):
        return "Hey " + str(robot_id) + ", I need you to move. " 
        
    @staticmethod
    def move_request_regex():
        return "Hey (\w+), I need you to move"
        
    @staticmethod
    def move_order(robot_id, location, convert_to_real_coordinates):
    
        real_location = convert_to_real_coordinates([location[0],location[1]])
        
        return str(robot_id) + ", move to (" + str(real_location[0]) + "," + str(real_location[1]) + "). " 
        
    @staticmethod
    def move_order_regex():
        return "(\w+), move to (\(-?\d+\.\d+,-?\d+\.\d+\))"
        
    @staticmethod
    def explanation_question(robot_id):
        return "What are you doing " + str(robot_id) + ". "
        
    @staticmethod
    def explanation_question_regex():
        return "What are you doing (\w+)"
        
    @staticmethod
    def pickup(object_id):
        return "Going to pick up object " + str(object_id) + ". "
        
    @staticmethod
    def pickup_regex():
        return "Going to pick up object (\d+)"
        
    @staticmethod
    def sensing():
        return "Sensing area. "
        
    @staticmethod
    def returning():
        return "Returning to goal area. "
        
    @staticmethod
    def explanation_follow(robot_id):
        return "I'm following " + str(robot_id) + ". "
        
    @staticmethod
    def explanation_response(action_index):
    
        response = ""
    
        if State.get_closest_object.value == action_index:
            
            if self.planning == "coordinator":
                response = "Moving to planning location. "
            else:
                response = "Figuring out my next objective. "
        elif State.sense_area.value == action_index:
            response = "Sensing area. "
        elif State.init_check_items.value == action_index:
            response = "Going to check my sensing results. "
        elif State.check_items.value == action_index:
            response = "Checking sensing results and selecting best objective. "
        elif State.move_and_pickup.value == action_index:
            response = "Moving to pick up object. "
        elif State.pickup_and_move_to_goal.value == action_index:
            response = "Picking object. "
        elif State.drop_object.value == action_index:
            response = "Dropping object in goal area. "
        elif State.move_end.value == action_index:
            response = "Just dropped object. "
        elif State.wait_message.value == action_index:
            response = "Waiting for others to respond. "
        elif State.check_neighbors.value == action_index:
            response = "Checking nearby robots. "
        elif State.follow.value == action_index:
            response = "Following. "
        elif State.wait_free.value == action_index:
            response = "Waiting agent to move from location. "
        elif State.obey.value == action_index:
            response = "Obeying my squad leader. "
        elif State.end_meeting.value == action_index:
            if self.planning == "coordinator":
                response = "Waiting for other robots to come. "
            else:
                response = "No more tasks to do. "
                
        elif State.waiting_order.value == action_index:
            response = "Waiting for an order to be given. "
        elif State.sense_compute.value == action_index:
            response = "Creating object assignment plan. "
        elif State.sense_order.value == action_index:
            response = "Creating optimum sensing order. "
        elif State.collect_order.value == action_index:
            response = "Issuing collection orders. "
        else:
            response = "Can't explain. "
        
        return response
        
    @staticmethod
    def order_sense(robot_id, object_id, location, convert_to_real_coordinates):
   
        real_location = convert_to_real_coordinates(location)
   
        return str(robot_id) + ", sense object " + str(object_id) + " at location (" + str(real_location[0]) + "," + str(real_location[1]) + "). "
    
    @staticmethod    
    def order_sense_regex():
        return "(\w+), sense object (\d+) at location (\(-?\d+\.\d+,-?\d+\.\d+\))"
       
    @staticmethod 
    def order_sense_multiple(robot_id, object_ids, locations, convert_to_real_coordinates):
        
        real_locations = []
        for l in locations:
            real_locations.append(str(tuple(convert_to_real_coordinates(l))))
            if not tuple(convert_to_real_coordinates(l)):
                pdb.set_trace()
            
        return str(robot_id) + ", sense objects [" + ','.join(object_ids) + "] at locations [" + ','.join(real_locations) + "]. "
        
    @staticmethod    
    def order_sense_multiple_regex():
        return "(\w+), sense objects (\[(,?\d+)+\]) at locations (\[(,?(\( *-?\d+(\.(\d+)?)? *, *-?\d+(\.(\d+)?)? *\)))+\])"
        
    @staticmethod
    def order_sense_room(robot_id, room):
        return str(robot_id) + ", sense room " + room + ". "
    
    @staticmethod    
    def order_sense_room_regex():
        return "(\w+), sense room (\d+)"    
        
    @staticmethod
    def order_collect(robot_id, object_id):

        return str(robot_id) + ", collect object " + str(object_id) + ". "
    
    @staticmethod    
    def order_collect_regex():
        return "(\w+), collect object (\d+)"
        
    @staticmethod
    def order_collect_group(robot_id, other_robot_ids, object_id):
    
        
        output_string = "Team leader: " + str(robot_id) + ". Helpers: ["
        
        for ori_idx,other_robot_id in enumerate(other_robot_ids):
            if ori_idx:
                output_string += "," + other_robot_id
            else:
                output_string += other_robot_id
                
        output_string += "]. Collect object " + str(object_id) + ". "

        return output_string 
        
    @staticmethod    
    def order_collect_group_regex():
        return "Team leader: (\w+). Helpers: (\[(,?\w+)+\]). Collect object (\d+)"
        
    @staticmethod
    def order_explore(robot_id, location, convert_to_real_coordinates):
    
        real_location = convert_to_real_coordinates(location)
        return str(robot_id) + ", go to location (" + str(real_location[0]) + "," + str(real_location[1]) + ") and report anything useful. "
    
    @staticmethod    
    def order_explore_regex():
        return "(\w+), go to location (\(-?\d+\.\d+,-?\d+\.\d+\)) and report anything useful"
        
    @staticmethod
    def order_finished():
        return "Order completed. "
        
    @staticmethod
    def order_finished_regex():
        return "Order completed"
        
    @staticmethod
    def task_finished():
        return "Task finished. "
        
    @staticmethod
    def order_response(robot_id, task):
        return "I will follow your orders to " + task + ", " + str(robot_id) + ". "
        
    @staticmethod
    def order_response_regex():
        return "I will follow your orders to (\w+), (\w+)"
        
    @staticmethod
    def order_response_negative(robot_id, leader_id):
        return "I cannot help you right now " + str(robot_id) + ", I'm following the orders of " + str(leader_id) + ". "
        
    @staticmethod
    def order_response_negative_regex():
        return "I cannot help you right now (\w+), I'm following the orders of (\w+)"
    
    @staticmethod
    def order_not_obey(robot_id):
        return "You don't have the authority to order me " + str(robot_id) + ". "
    
    @staticmethod
    def order_not_obey_regex():
        return "You don't have the authority to order me (\w+)"
        
    @staticmethod
    def order_cancel(robot_id):
        return "Order cancelled " + str(robot_id) + ". "
    
    @staticmethod
    def order_cancel_regex():
        return "Order cancelled (\w+)"
        
    @staticmethod
    def finish():
        return "Let's end participation. "
        
    @staticmethod
    def finish_regex():
        return "Let's end participation"
        
    @staticmethod
    def finish_reject():
        return "Wait, let's not end participation yet. "
        
    @staticmethod
    def finish_reject_regex():
        return "Wait, let's not end participation yet"
        
    @staticmethod
    def parse_sensing_message(rematch, rm, robotState, info, other_agents, convert_to_grid_coordinates, convert_to_real_coordinates):    
        
        object_id = rematch.group(1)
        
        object_idx = info['object_key_to_index'][object_id]
        
        
        item = {}
        
        last_seen = list(eval(rematch.group(3)))
        
        max_real_coords = convert_to_real_coordinates((robotState.latest_map.shape[0]-1, robotState.latest_map.shape[1]-1))
        
        if last_seen[0] > max_real_coords[0] or last_seen[1] > max_real_coords[1]: #if last_seen[0] == 99.99 and last_seen[1] == 99.99:
            item["item_location"] = [-1,-1]
        else:
            item["item_location"] = convert_to_grid_coordinates(last_seen)
        last_time = rematch.group(4).split(":")
        item["item_time"] = [int(last_time[1]) + int(last_time[0])*60]
        item["item_weight"] = int(rematch.group(2))
        item["item_danger_level"] = 0
        item["item_danger_confidence"] = [0.0]
        
        
        sender_agent_idx = info['robot_key_to_index'][rm[0]]

        if object_id not in other_agents[sender_agent_idx].items.keys():
            other_agents[sender_agent_idx].items[object_id] = [] #{"danger_level":0,"confidence":0}
        
        if rematch.group(5):
            danger_list = rematch.group(6).strip('][').split(',')
            prob_list = rematch.group(8).strip('][').split(',')
            from_list = rematch.group(10).strip('][').split(',')
            
            for lidx in range(len(danger_list)):
            
                if "benign" in danger_list[lidx]:
                    danger_level = 1
                    
                else:
                    danger_level = 2
            
                item["item_danger_level"] = danger_level
                item["item_danger_confidence"] = [float(prob_list[lidx].strip('%'))/100]
                print("update estimates", danger_level,item["item_danger_confidence"])
        
                if from_list[lidx] not in info['robot_key_to_index']: #This means it is you!
                    agent_idx = -1
                    agent_id = ""
                else:
                    agent_idx = info['robot_key_to_index'][from_list[lidx]]
                    agent_id = from_list[lidx]
                robotState.update_items(item,object_id,object_idx, agent_idx) #Object gets updated based on higher confidence estimates
                
                if from_list[lidx] not in other_agents[sender_agent_idx].items[object_id]:
                    other_agents[sender_agent_idx].items[object_id].append(from_list[lidx])
        
        
            

        """
        other_agents[agent_idx].items[object_id]["danger_level"] = item["item_danger_level"] #Update estimates about what other robots know
        
        if item["item_danger_confidence"]:
            other_agents[agent_idx].items[object_id]["confidence"] = item["item_danger_confidence"][0]
        else:
            other_agents[agent_idx].items[object_id]["confidence"] = 0
        """
        
        
    
    @staticmethod        
    def exchange_sensing_info(robotState, info, nearby_other_agents, other_agents, robot_id, convert_to_real_coordinates):
    
        message_text = ""
        
        object_info_message = []
        
        
        for noa in nearby_other_agents:   
            
            for item_idx in range(robotState.get_num_objects()):
                danger_level = robotState.get("objects", "danger_status", item_idx)
                
               
                if danger_level > 0:
                
                    confidence = robotState.get("objects", "estimate_correct_percentage", item_idx)
                
                    object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(item_idx)]
                    
                    if object_id not in other_agents[noa].items.keys():
                        other_agents[noa].items[object_id] = [] #{"danger_level":0,"confidence":0}
                        
                    
                    robot_estimates = []
                    send_message = False
                    for rie_idx in range(robotState.get_num_estimates(item_idx)):
                        if robotState.get("object_estimates", "danger_status", [item_idx,rie_idx]) > 0:
                            if rie_idx == robotState.get_num_estimates(item_idx)-1: #This is the robot itself:
                                sensing_robot = robot_id
                            else:
                                sensing_robot = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(rie_idx)]
                            
                            if sensing_robot not in other_agents[noa].items[object_id]:
                                other_agents[noa].items[object_id].append(sensing_robot)
                                send_message = True
                    
                    if send_message:            
                        message_text += MessagePattern.item(robotState,item_idx,object_id, info, robot_id, convert_to_real_coordinates)
                    
                        
                    """
                    
                    if not (other_agents[noa].items[object_id]["danger_level"] == danger_level and other_agents[noa].items[object_id]["confidence"] == confidence):
                    
                        if object_id not in object_info_message:
                            object_info_message.append(object_id)
                            message_text += MessagePattern.item(robotState.items,item_idx,object_id, convert_to_real_coordinates)
                            
                        other_agents[noa].items[object_id]["danger_level"] = danger_level
                        other_agents[noa].items[object_id]["confidence"] = confidence
                    """
                        
        return message_text
