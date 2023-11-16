import pdb


class MessagePattern:
    @staticmethod
    def location(goal_x,goal_y,next_loc_x,next_loc_y,convert_to_real_coordinates, current_location, carrying, helping):
        real_goal = convert_to_real_coordinates([goal_x,goal_y])
        real_next_location = convert_to_real_coordinates([next_loc_x,next_loc_y])
        real_current_location = convert_to_real_coordinates(current_location)
        output_string = "My goal is (" + str(real_goal[0]) + "," + str(real_goal[1]) + "), I'm moving towards (" + str(real_next_location[0]) + "," + str(real_next_location[1]) + "). My current location is (" + str(real_current_location[0]) + "," + str(real_current_location[1]) + "). "
        
        if carrying:
            output_string += "Carrying object. "
            
        if helping:
            output_string += "Helping " + str(helping[0]) + ". "
            
            
        return output_string
            
        
    @staticmethod
    def location_regex():
        return "My goal is (\(-?\d+\.\d+,-?\d+\.\d+\)), I'm moving towards (\(-?\d+\.\d+,-?\d+\.\d+\)). My current location is (\(-?\d+\.\d+,-?\d+\.\d+\)).( Carrying object.)?( Helping (\w+))?"
        
    @staticmethod
    def item(robotState,item_idx,object_id, info, robot_id, convert_to_real_coordinates):

        """ Example
        Object 1 (weight: 1) Last seen in (5.5,5.5) at 00:57. Status Danger: benign, Prob. Correct: 88.1%
        """
    
        message_text = ""
                            
        if robotState.items[item_idx]["item_weight"]:
                            
            item_loc = robotState.items[item_idx]['item_location']
        
            try:
                mins, remainder = divmod(robotState.items[item_idx]["item_time"][0], 60)
            except:
                pdb.set_trace()
            secs,millisecs = divmod(remainder,1)
            
            
            time_formatted = '{:02d}:{:02d}'.format(int(mins), int(secs))
            
            
            if not (item_loc[0] == -1 and item_loc[1] == -1):
                real_location = convert_to_real_coordinates(item_loc)
            else:
               real_location = [99.99,99.99]
        
            message_text = "Object " + str(object_id) + " (weight: " +  str(robotState.items[item_idx]["item_weight"]) + ") Last seen in (" + str(real_location[0]) + "," + str(real_location[1]) + ") at " + time_formatted + ". "
                
            status_danger = ""
            prob_correct = ""   
            from_estimates = ""        
            for robo_idx,roboestimate in enumerate(robotState.item_estimates[item_idx]):        
                if roboestimate['item_danger_level'] > 0:
                    
                    if not status_danger:                
                        status_danger +=  "Status Danger: ["
                        prob_correct += "Prob. Correct: ["
                        from_estimates += "From: ["
                    else:
                        status_danger += ","
                        prob_correct += ","
                        from_estimates += ","
                        
                    if roboestimate['item_danger_level'] == 1:
                        status_danger += "benign"
                    else:
                        status_danger += "dangerous"
                        
                    prob_correct += str(round(roboestimate["item_danger_confidence"][0]*100,1)) + "%"
                    
                    if robo_idx == len(robotState.item_estimates[item_idx])-1:
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
    def item_regex_partial():
        return "Object (\d+) \(weight: (\d+)\) Last seen in (\(-?\d+\.\d+,-?\d+\.\d+\)) at (\d+:\d+)"
    @staticmethod
    def item_regex_full():
        return "Object (\d+) \(weight: (\d+)\) Last seen in (\(-?\d+\.\d+,-?\d+\.\d+\)) at (\d+:\d+).( Status Danger: (\[\w+(,\w+)*\]), Prob. Correct: (\[\d+\.\d+%(,\d+\.\d+%)*\]), From: (\[\w+(,\w+)*\]))?"#"Object (\d+) \(weight: (\d+)\) Last seen in (\(-?\d+\.\d+,-?\d+\.\d+\)) at (\d+:\d+).( Status Danger: (\w+), Prob. Correct: (\d+\.\d+)%)?"
    
    @staticmethod
    def sensing_help(object_id):
        return "What do you know about object " + str(object_id) + ". "
        
    @staticmethod
    def sensing_help_regex():
        return "What do you know about object (\d+)" 
        
    @staticmethod    
    def sensing_help_negative_response(object_id):
        return "I know nothing about object " + str(object_id) + ". "
        
    @staticmethod    
    def sensing_help_negative_response_regex():
        return "I know nothing about object (\d+)"
        
    @staticmethod
    def carry_help(object_id, num_robots):
        return "I need " + str(num_robots) + " more robots to help carry object " + str(object_id) + ". "
        
    @staticmethod
    def carry_help_regex():
        return "I need (\d+) more robots to help carry object (\d+)"
        
    @staticmethod
    def carry_help_accept(robot_id):
        return "I can help you " + str(robot_id) + ". "
        
    @staticmethod
    def carry_help_accept_regex():
        return "I can help you (\w+)"
        
    @staticmethod
    def carry_help_participant_reject(robot_id):
        return "I cannot help you " + str(robot_id) + ". "
        
    @staticmethod
    def carry_help_participant_reject_regex():
        return "I cannot help you (\w+)"
        
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
        return "Thanks, follow me " + str(robot_id) + ". "
        
    @staticmethod
    def follow_regex():
        return "Thanks, follow me (\w+)"
        
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
    def order_sense(robot_id, location, convert_to_real_coordinates):
   
        real_location = convert_to_real_coordinates(location)
   
        return str(robot_id) + ", sense location (" + str(real_location[0]) + "," + str(real_location[1]) + "). "
    
    @staticmethod    
    def order_sense_regex():
        return "(\w+), sense location (\(-?\d+\.\d+,-?\d+\.\d+\))"
        
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
        return "Team leader: (\w+). Helpers: \[(\w+)(,\w+)*\]. Collect object (\d+)"
        
    @staticmethod
    def order_finished():
        return "Order completed. "
        
    @staticmethod
    def task_finished():
        return "Task finished. "
        
    @staticmethod
    def finish():
        return "Let's finish. "
        
    @staticmethod
    def finish_reject():
        return "Wait, not yet. "
        
    @staticmethod
    def parse_sensing_message(rematch, rm, robotState, info, other_agents, convert_to_grid_coordinates):    
        
        object_id = rematch.group(1)
        
        object_idx = info['object_key_to_index'][object_id]
        
        
        item = {}
        
        last_seen = list(eval(rematch.group(3)))
        
        if last_seen[0] == 99.99 and last_seen[1] == 99.99:
            item["item_location"] = [-1,-1]
        else:
            item["item_location"] = convert_to_grid_coordinates(last_seen)
        last_time = rematch.group(4).split(":")
        item["item_time"] = [int(last_time[1]) + int(last_time[0])*60]
        item["item_weight"] = int(rematch.group(2))
        item["item_danger_level"] = 0
        item["item_danger_confidence"] = []
        
        
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
                else:
                    agent_idx = info['robot_key_to_index'][from_list[lidx]]
                robotState.update_items(item,object_idx,agent_idx) #Object gets updated based on higher confidence estimates
                
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
            
            for item_idx in range(len(robotState.items)):
                danger_level = robotState.items[item_idx]["item_danger_level"]
                
               
                if danger_level > 0:
                
                    confidence = robotState.items[item_idx]["item_danger_confidence"][0]
                
                    object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(item_idx)]
                    
                    if object_id not in other_agents[noa].items.keys():
                        other_agents[noa].items[object_id] = [] #{"danger_level":0,"confidence":0}
                        
                    
                    robot_estimates = []
                    send_message = False
                    for rie_idx,rie in enumerate(robotState.item_estimates[item_idx]):
                        if rie["item_danger_level"] > 0:
                            if rie_idx == len(robotState.item_estimates[item_idx])-1: #This is the robot itself:
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
