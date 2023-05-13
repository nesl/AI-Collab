import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import json_numpy

from magnebot import ActionStatus, Arm
import cv2
from aiohttp import web
from av import VideoFrame
import aiohttp_cors
import socketio
import pdb
import numpy as np
import time

from enum import Enum

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

import gymnasium as gym
from gymnasium import spaces

from .action import Action


class AICollabEnv(gym.Env):

    # MAIN & SETUP OF HTTP SERVER #########################################

    def __init__(self, use_occupancy, view_radius, client_number, address,
                 host=None, port=None, cert_file=None, key_file=None):

        self.pcs = set()
        self.relay = MediaRelay()
        self.tracks_received = 0

        self.frame_queue = ""

        self.client_number = client_number
        self.robot_id = 0
        self.use_occupancy = use_occupancy
        self.view_radius = view_radius
        self.centered_view = 0
        self.host = host
        self.port = port
        self.setup_ready = False
        self.confirm_time_threshold = 2  # Seconds to confirm

        self.ask_info_agents_str = "Ask for agent information to "
        self.ask_info_objects_str = "Ask for object information to "

        self.own_neighbors_info_entry = []
        self.waiting_output = False
        self.requested_output = []
        self.object_key_to_index = {}
        self.robot_key_to_index = {}
        self.goal_coords = []
        self.objects_in_goal = []
        self.extra = {}

        # SOCKET IO message function definitions ###########################

        self.sio = socketio.Client(ssl_verify=False)

        # When first connecting
        @self.sio.event
        def connect():
            print("I'm connected!")
            if not self.use_occupancy:
                self.sio.emit("watcher_ai",
                              (self.client_number,
                               self.use_occupancy,
                               "https://" + self.host + ":" +
                               str(self.port) + "/offer",
                               0,
                               0))
            else:
                self.sio.emit(
                    "watcher_ai",
                    (self.client_number,
                     self.use_occupancy,
                     "",
                     self.view_radius,
                     self.centered_view))
            # asyncio.run(main_ai(tracks_received))

        # Receiving simulator's robot id
        @self.sio.event
        def watcher_ai(robot_id_r, occupancy_map_config):

            print("Received id", robot_id_r)
            self.robot_id = robot_id_r

            if self.use_occupancy:  # When using only occupancy maps, run the main processing function here
                self.map_config = occupancy_map_config

                remove_self = -1
                for robot_idx, robot in enumerate(
                        self.map_config['all_robots']):
                    if robot[0] == str(self.robot_id):
                        remove_self = robot_idx
                        break
                del self.map_config['all_robots'][robot_idx] #Remove self from list of robots received
                
                #asyncio.run(self.main_ai())
                self.gym_setup()
                self.setup_ready = True

        # Receiving occupancy map
        self.maps = []
        self.map_ready = False
        self.map_config = {}

        @self.sio.event
        def occupancy_map(object_type_coords_map,
                          object_attributes_id, objects_held):

            # print("occupancy_map received")
            # s_map = json_numpy.loads(static_occupancy_map)
            c_map = json_numpy.loads(object_type_coords_map)
            self.maps = (c_map, object_attributes_id)
            self.objects_held = objects_held
            self.map_ready = True

            # print(c_map)

        # Connection error

        @self.sio.event
        def connect_error(data):
            print("The connection failed!")

        # Disconnect

        @self.sio.event
        def disconnect():
            print("I'm disconnected!")

        # Received a target object NOT USED
        @self.sio.event
        def set_goal(agent_id, obj_id):
            print("Received new goal")
            # self.target[agent_id] = obj_id

        # Update neighbor list
        self.new_neighbors = []

        @self.sio.event
        def neighbors_update(neighbors_list, source_id):

            print('neighbors update', neighbors_list)
            self.new_neighbors = neighbors_list

        # Update object list
        self.new_objects = []

        @self.sio.event
        def objects_update(objects_list, source_id):

            print("objects_update", objects_list)
            self.new_objects = objects_list

        # Receive messages from other agents
        self.messages = []

        @self.sio.event
        def message(message, timestamp, source_agent_id):

            # Special case for receiving data update
            if self.ask_info_objects_str in message:
                print("Objects UPDATE")
                self.sio.emit(
                    "objects_update", (source_agent_id, self.object_info))
            elif self.ask_info_agents_str in message:

                extended_neighbors_info = self.get_corrected_neighbors_info(
                    source_agent_id)  # Without target info and including own info
                print("Neighbors UPDATE", extended_neighbors_info)
                self.sio.emit(
                    "neighbors_update",
                    (source_agent_id,
                     extended_neighbors_info))
            else:
                self.messages.append((source_agent_id, message, timestamp))

            print("message", message, source_agent_id)

        self.new_output = ()
        # Get output from simulator

        @self.sio.event
        def ai_output(object_type_coords_map, object_attributes_id, objects_held,
                      sensing_results, ai_status, extra_status, strength, timer, disable):

            self.map = json_numpy.loads(object_type_coords_map)

            if self.waiting_output and any(
                    extra_status):  # If the robot is requesting information, save it until the next step
                self.requested_output = (
                    self.map,
                    object_attributes_id,
                    objects_held,
                    sensing_results,
                    ActionStatus(ai_status),
                    extra_status,
                    strength,
                    timer)
                self.waiting_output = False

            self.new_output = (
                self.map,
                object_attributes_id,
                objects_held,
                sensing_results,
                ActionStatus(ai_status),
                extra_status,
                strength,
                timer)

        # Receive status updates of our agent
        self.action_status = -1

        @self.sio.event
        def ai_status(status):

            self.action_status = ActionStatus(status)
            print("status", ActionStatus(status))

        self.agent_reset = False
        # Reset agent

        @self.sio.event
        def agent_reset():
            self.agent_reset = True
            print("Agent reset")

        self.run(address, cert_file, key_file)

        while not self.setup_ready:
            time.sleep(1)

    # When sharing robots information, remove the receiver robot info and
    # append yours
    def get_corrected_neighbors_info(self, target_id):
        corrected_neighbors_info = self.neighbors_info.copy()

        for ni_idx in range(len(corrected_neighbors_info)):
            if corrected_neighbors_info[ni_idx][0] == target_id:
                del corrected_neighbors_info[ni_idx]
                break

        corrected_neighbors_info.append(self.own_neighbors_info_entry)

        return corrected_neighbors_info

    # Connect to Socket.IO and optionally setup server
    def run(self, address, cert_file, key_file):

        if self.use_occupancy:
            self.sio.connect(address)
            # main_thread()
        else:
            if cert_file:
                # ssl_context = ssl.SSLContext()
                ssl_context = ssl.create_default_context(
                    ssl.Purpose.CLIENT_AUTH)
                ssl_context.load_cert_chain(cert_file, key_file)
            else:
                ssl_context = None

            app = web.Application()

            async def on_shutdown(app):
                # close peer connections
                coros = [pc.close() for pc in self.pcs]
                await asyncio.gather(*coros)
                self.pcs.clear()

            app.on_shutdown.append(on_shutdown)
            # app.router.add_get("/", index)
            # app.router.add_get("/client.js", javascript)
            app.router.add_post("/offer", self.offer)

            cors = aiohttp_cors.setup(app, defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*"
                )
            })

            for route in list(app.router.routes()):
                cors.add(route)

            self.sio.connect(address)
            web.run_app(
                app, access_log=None, host=self.host, port=self.port, ssl_context=ssl_context
            )

    # GYM SETUP ###########################################################

    def gym_setup(self):

        map_size = self.map_config['num_cells'][0]
        self.action_space = spaces.Dict(
            {
                "action": spaces.Discrete(len(Action)),
                "item": spaces.Discrete(self.map_config['num_objects']),
                # Allow for 0
                "robot": spaces.Discrete(len(self.map_config['all_robots']) + 1),
                "message" : spaces.Text(min_length=0,max_length=100)
            }
        )

        # self.observation_space = spaces.Box(0, map_size - 1, shape=(2,), dtype=int)

        self.observation_space = spaces.Dict(
            {
                "frame": spaces.Box(low=-2, high=5, shape=(map_size, map_size), dtype=np.int64),
                "objects_held": spaces.Discrete(3, start=-1),
                "action_status": spaces.MultiDiscrete(np.array([2] * 4), dtype=np.int64),

                "item_output": spaces.Dict(
                    {
                        "item_weight": spaces.Discrete(10),
                        "item_danger_level": spaces.Discrete(3),
                        "item_location": spaces.Box(low=-np.infty, high=np.infty, shape=(2,), dtype=np.int64)
                    }
                ),
                "num_items": spaces.Discrete(self.map_config['num_objects'] + 1),

                "neighbors_output": spaces.Dict(
                    {
                        "neighbor_type": spaces.Discrete(3, start=-1),
                        "neighbor_location": spaces.Box(low=-np.infty, high=np.infty, shape=(2,), dtype=np.int64)
                    }

                ),
                # Strength starts from zero
                "strength": spaces.Discrete(len(self.map_config['all_robots']) + 2),
                "num_messages": spaces.Discrete(100)

                # "objects_danger_level" : spaces.Box(low=1,high=2,shape=(self.map_config['num_objects'],), dtype=int)
            }
        )

        self.goal_count = 0

    def step(self, action):

        # previous_objects_held = self.objects_held

        world_state, sensing_output, action_terminated, action_truncated = self.take_action(
            action)
        # observed_state = {"frame": world_state, "message": self.messages}
        
        observation = {"frame": sensing_output["occupancy_map"],
                       "objects_held": sensing_output["objects_held"],
                       "action_status": np.array([int(action_terminated[0]),
                                                  int(action_truncated[0]),
                                                  int(action_terminated[1]),
                                                  int(action_truncated[1])], dtype=np.int64),
                       "num_items": len(self.object_info),
                       "item_output": sensing_output["item_output"],
                       "neighbors_output": sensing_output["neighbors_output"],
                       "num_messages": len(self.messages),
                       "strength": sensing_output["strength"]}  # Occupancy map

        info = {}
        info['map_metadata'] = sensing_output["objects_metadata"]
        info['messages'] = sensing_output["messages"]
        info['robot_key_to_index'] = self.robot_key_to_index
        info['object_key_to_index'] = self.object_key_to_index
        
        reward = 0
        

        

        #REWARD ESTIMATION
        
        object_metadata = world_state[1]
        #Give a reward whenever agent drops object in desired area
        if 'dropped_object' in self.extra and self.extra['dropped_object'] and self.extra['dropped_object'] not in self.objects_in_goal: #Check that the agent just dropped object, and that this object has not been put in the goal area in the past

            for xy in self.goal_coords:
                key = str(xy[0]) + '_' + str(xy[1])
                if key in object_metadata:
                    try:
                        if self.extra['dropped_object'] in object_metadata[key][0]:
                            reward += 1

                    except:
                        pdb.set_trace()
                        
                       
        
        
        for xy in self.goal_coords:
            key = str(xy[0]) + '_' + str(xy[1])

            if key in object_metadata and  isinstance(object_metadata[key][0], list) and object_metadata[key][0][0] not in self.objects_in_goal:
                self.objects_in_goal.append(object_metadata[key][0][0]) 

        
        
        if len(self.objects_in_goal) == self.map_config['num_objects']: #When four objects are put in the middle the episode should terminate
            terminated = True
        else:
            terminated = False
            
        #Rewards
        
        '''
        if previous_objects_held[0] and not self.objects_held[0]: #Reward given when object is left in the middle of the room

            goal_radius = 5
            max_x = np.round(world_state[0].shape[0]/2) + goal_radius
            min_x = np.round(world_state[0].shape[0]/2) - goal_radius
            max_y = np.round(world_state[0].shape[1]/2) + goal_radius
            min_y = np.round(world_state[0].shape[1]/2) - goal_radius
            ego_location = np.where(world_state[0] == 5)

            #max_x = min(ego_location[0][0] + self.view_radius, world_state[0].shape[0])
            #max_y = min(ego_location[1][0] + self.view_radius, world_state[0].shape[1])
            #min_x = max(ego_location[0][0] - self.view_radius, 0)
            #min_y = max(ego_location[1][0] - self.view_radius, 0)

            w_idxs = np.where(world_state[0][min_x:max_x+1,min_y:max_y+1] > 1)
            object_ids = {}
            for w_ix in range(len(w_idxs)):
                new_idx = (w_idxs[0][w_ix],w_idxs[1][w_ix])
                if previous_objects_held[0] in world_state[1][str(new_idx[0]+min_x) + str(new_idx[1]+min_y)]:
                    reward = 1
                    self.goal_count += 1

        if self.objects_held[0] and not previous_objects_held[0]: #Reward given when grabbing objects
            reward = 0.5

        #if action_truncated: #Penalty given when not being able to grab an object or drop an object
        #    reward = -0.5
        
        
        
        if self.goal_count == 4: #When four objects are put in the middle the episode should terminate
            terminated = True
        else:
            terminated = False
        '''
        
        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        map_size = self.map_config['num_cells'][0]

        observation = {

            "frame": np.ones((map_size, map_size), dtype=np.int64)*(-2),
            "objects_held": -1,
            "action_status": np.zeros(4, dtype=np.int64),

            "item_output": {
                "item_weight": 0,
                "item_danger_level": 0,
                "item_location": np.ones(2, dtype=np.int64)*(-1)
            },
            "num_items": 0,

            "neighbors_output": {
                "neighbor_type": -1,
                "neighbor_location": np.ones(2, dtype=np.int64)*(-1)
            },

            "strength": 1,
            "num_messages": 0,

        }

        info = {}

        goal_radius = 5
        max_x = int(np.round(map_size/2) + goal_radius)
        min_x = int(np.round(map_size/2) - goal_radius)
        max_y = int(np.round(map_size/2) + goal_radius)
        min_y = int(np.round(map_size/2) - goal_radius)
        
        self.goal_coords = [(x,y) for x in range(min_x,max_x+1) for y in range(min_y, max_y+1)]
        
        #self.goal_coords = (min_x,max_x,min_y,max_y)
        
        self.objects_in_goal = []
        self.extra = {}
        
        
        self.internal_state = [self.State.take_action, self.State.take_sensing_action]
        self.internal_data = {}
        self.object_info = []
        self.neighbors_info = [[um[0], 0 if um[1] == 'human' else 1,0,0,-1] for um in self.map_config['all_robots']]
        self.robot_key_to_index = {self.neighbors_info[i][0]:i for i in range(len(self.neighbors_info))}
        self.own_neighbors_info_entry = [self.robot_id, 1, 0, 0, -1]

        self.sio.emit("reset_ai")
        print("Reseting agent")
        while not self.agent_reset:
            continue

        self.agent_reset = False

        self.map = np.array([], dtype=np.int16)
        
        print("Waiting for location")
        while self.map.size == 0:
            continue

        self.old_output = self.new_output
        print("Got location")

        observation["frame"] = self.map

        self.messages = []
        self.goal_count = 0

        return observation, info

    # ROBOT API ###########################################################

    # Forwarded magnebot API from
    # https://github.com/alters-mit/magnebot/blob/main/doc/manual/magnebot/actions.md\

    def turn_by(self, angle, aligned_at=1):
        return ["turn_by", str(angle), str(aligned_at)]

    def turn_to(self, target, aligned_at=1):
        return ["turn_to", str(target), str(aligned_at)]

    def move_by(self, distance, arrived_at=0.1):
        return ["move_by", str(distance), str(arrived_at)]

    def move_to(self, target, arrived_at=0.1, aligned_at=1, arrived_offset=0):
        return ["move_to", json.dumps(target), str(arrived_at),
                str(aligned_at), str(arrived_offset)]

    def reach_for(self, target, arm):
        return ["reach_for", str(target), str(arm.value)]

    def grasp(self, target, arm):
        return ["grasp", str(target), str(arm.value)]

    def drop(self, target, arm):
        return ["drop", str(target), str(arm.value)]

    def reset_arm(self, arm):
        return ["reset_arm", str(arm.value)]

    def reset_position(self):
        return ["reset_position"]

    def rotate_camera(self, roll, pitch, yaw):
        return ["rotate_camera", str(roll), str(pitch), str(yaw)]

    def look_at(self, target):
        return ["look_at", str(target)]

    def move_camera(self, position):
        return ["move_camera", str(position)]

    def reset_camera(self):
        return ["reset_camera"]

    def slide_torso(self, height):
        return ["slide_torso", str(height)]

    def danger_sensor_reading(self):
        return ["send_danger_sensor_reading"]

    def get_occupancy_map(self):
        return ["send_occupancy_map"]

    def get_objects_held_status(self):
        return ["send_objects_held_status"]

    # CONTROLLER DEFINITION ###############################################

    # Controller states

    class State(Enum):
        take_action = 1
        waiting_ongoing = 2
        grasping_object = 3
        reseting_arm = 4
        reverse_after_dropping = 5
        take_sensing_action = 6
        wait_sensing = 7
        action_end = 8
        wait_get_objects = 9
        wait_get_agents = 10

    def take_action(self, action):

        terminated = False
        truncated = False
        objects_obs = []
        neighbors_obs = []

        # print(action)

        action_message, self.internal_state, self.internal_data, sensing_output, terminated, truncated = self.controller(
            action, self.old_output, self.internal_state, self.internal_data)

        if action_message:  # Action message is an action to take by the robot that will be communicated to the simulator
            print("action", action_message)
            self.sio.emit("ai_action", (action_message))

        while not self.new_output:  # Sync with simulator
            pass

        if action_message and any(self.new_output[5]):
            print(self.new_output[5])

        if self.new_output:
            self.old_output = self.new_output
            self.new_output = ()

        return self.old_output, sensing_output, terminated, truncated

    # Only works for occupancy maps not centered in magnebot

    def controller(self, complete_action, observations, internal_state, data):

        action_message = []
        movement_commands = 8
        grab_commands = 16

        occupancy_map = observations[0]
        objects_metadata = observations[1]
        objects_held = observations[2]
        danger_sensing_data = observations[3]
        action_status = observations[4]
        extra_status = observations[5]
        strength = observations[6]
        timer = observations[7]
        terminated = [False, False]
        truncated = [False, False]
        state = internal_state[0]
        sensing_state = internal_state[1]
        action = Action(complete_action["action"])

        sensing_output = {
            "occupancy_map": occupancy_map,
            "item_output": {
                "item_weight": 0,
                "item_danger_level": 0,
                "item_location": np.array([-1, -1], dtype=np.int64)},
            "messages": "",
            "neighbors_output": {
                "neighbor_type": -1,
                "neighbor_location": np.array([-1, -1], dtype=np.int64)},
            "objects_held": -1,
            "strength": strength,
            "objects_metadata": {}
        }
        
        

        # print(state, sensing_state)

        ego_location = np.where(occupancy_map == 5)
        ego_location = np.array([ego_location[0][0], ego_location[1][0]])
        

        self.own_neighbors_info_entry[2] = float(ego_location[0])
        self.own_neighbors_info_entry[3] = float(ego_location[1])
        self.own_neighbors_info_entry[4] = float(timer)

        if state == self.State.take_action:

            #if action_status != ActionStatus.ongoing:
            #print("Original ", action)
            #print(occupancy_map)
        
            #self.action_status = -1
            data['timer_locomotion'] = time.time()
            
            if action.value < movement_commands:

                action_index = [
                    Action.move_up,
                    Action.move_right,
                    Action.move_down,
                    Action.move_left,
                    Action.move_up_right,
                    Action.move_up_left,
                    Action.move_down_right,
                    Action.move_down_left].index(action)

                original_location = np.copy(ego_location)

                ego_location = self.check_bounds(
                    action_index, ego_location, occupancy_map)

                if not np.array_equal(ego_location, original_location):
                    target_coordinates = np.array(
                        self.map_config['edge_coordinate']) + ego_location * self.map_config['cell_size']
                    target = {
                        "x": target_coordinates[0],
                        "y": 0,
                        "z": target_coordinates[1]}
                    state = self.State.waiting_ongoing
                    data["next_state"] = self.State.action_end
                    action_message.append(self.move_to(target=target))
                else:
                    print("Movement not possible")
                    truncated[0] = True

            elif action.value < grab_commands:

                object_location = np.copy(ego_location)

                action_index = [
                    Action.grab_up,
                    Action.grab_right,
                    Action.grab_down,
                    Action.grab_left,
                    Action.grab_up_right,
                    Action.grab_up_left,
                    Action.grab_down_right,
                    Action.grab_down_left].index(action)

                object_location = self.check_bounds(action_index, object_location, occupancy_map)
                key = str(object_location[0]) + '_' + str(object_location[1])
                
                if (not np.array_equal(object_location,ego_location)) and key in objects_metadata: #occupancy_map[object_location[0],object_location[1]] == 2:
                    print("Grabbing object")
                    #object_location = np.where(occupancy_map == 2)
                    #key = str(object_location[0][0]) + str(object_location[1][0])

                    
                    if isinstance(objects_metadata[key][0], list): 
                        object_id = objects_metadata[key][0][0]
                    else:
                        object_id = objects_metadata[key][0]
                    action_message.append(self.turn_to(object_id))
                   
                    state = self.State.waiting_ongoing
                    data["next_state"] = self.State.grasping_object
                    data["object"] = object_id
                    

                else:
                    print("No object to grab")
                    truncated[0] = True

            elif action == Action.drop_object:


                if objects_held[0]:
                    action_message.append(self.drop(objects_held[0], Arm.left))

                    state = self.State.waiting_ongoing
                    data["next_state"] = self.State.reverse_after_dropping
                    self.extra['dropped_object'] = objects_held[0]

                else:
                    print("No object to drop")
                    truncated[0] = True

            else:
                # print("Not implemented", action)
                pass

            '''
            else:
                truncated[0] = 1
                print("Ongoing truncated", timer)
            '''

        elif state == self.State.waiting_ongoing:

            # or action_status == ActionStatus.success:
            if action_status == ActionStatus.ongoing:
                print("waiting", action_status, timer)
                state = data["next_state"]
            elif time.time() - data['timer_locomotion'] > 5 and action_status == ActionStatus.success:
                print("waiting", action_status, timer)
                state = data["next_state"]
            elif time.time() - data['timer_locomotion'] > 10: #Timer if it gets stuck
                truncated[1] = True
                print("Action stuck", action_status)
                    
        elif state == self.State.grasping_object:
            if action_status != ActionStatus.ongoing:
                state = self.State.waiting_ongoing
                print("waited to grasp objective")
                action_message.append(self.grasp(data["object"], Arm.left))
                del data["object"]
                data["next_state"] = self.State.reseting_arm

        elif state == self.State.reseting_arm:

            if action_status != ActionStatus.ongoing:
                print("waited to reset arm")
                action_message.append(self.reset_arm(Arm.left))
                state = self.State.waiting_ongoing
                data["next_state"] = self.State.action_end

        elif state == self.State.reverse_after_dropping:
            if action_status != ActionStatus.ongoing:
                print("waited to reverse after dropping")
                action_message.append(self.move_by(-0.5))
                state = self.State.waiting_ongoing
                data["next_state"] = self.State.action_end

        elif state == self.State.action_end:
            if action_status != ActionStatus.ongoing:
                print("action end", action_status, timer)
                terminated[0] = True

        if terminated[0] or truncated[0]:
            state = self.State.take_action

        # Check for sensing/comms actions
        if sensing_state == self.State.take_sensing_action:

            if action == Action.danger_sensing:
                action_message.append(self.danger_sensor_reading())
                sensing_state = self.State.wait_sensing
                self.waiting_output = True

            elif action == Action.get_occupancy_map:
                action_message.append(self.get_occupancy_map())
                sensing_state = self.State.wait_sensing
                self.waiting_output = True

            elif action == Action.get_objects_held:
                action_message.append(self.get_objects_held_status())
                sensing_state = self.State.wait_sensing
                self.waiting_output = True

            elif action == Action.check_item:
                if complete_action["item"] >= len(self.object_info):
                    truncated[1] = True
                else:
                    sensing_output["item_output"]["item_weight"] = self.object_info[complete_action["item"]][1]
                    sensing_output["item_output"]["item_danger_level"] = self.combine_danger_info(
                        self.object_info[complete_action["item"]][2])
                    sensing_output["item_output"]["item_location"] = np.array(self.object_info[complete_action["item"]][3:5])
                    terminated[1] = True

            elif action == Action.check_robot:

                # If 0, it means broadcast so we ignore it
                if complete_action["robot"] > 0:
                    robot_idx = complete_action["robot"] - 1
                    sensing_output["neighbors_output"]["neighbor_type"] = self.neighbors_info[robot_idx][1]
                    sensing_output["neighbors_output"]["neighbor_location"] = self.neighbors_info[robot_idx][2:4]
                    terminated[1] = True
                else:
                    truncated[1] = True

            elif action == Action.get_messages:
                if self.messages:
                    sensing_output["message"] = self.messages.copy()
                    self.messages = []
                    terminated[1] = True
                else:
                    truncated[1] = True

            elif action == Action.send_message:
                if complete_action["robot"] > 0:

                    robot_data = self.neighbors_info[complete_action["robot"] - 1]
                    neighbors_dict = {
                        robot_data[0]: "human" if not robot_data[1] else "ai"}
                else:
                    neighbors_dict = {
                        robot_data[0]: "human" if not robot_data[1] else "ai" for robot_data in self.neighbors_info}

                self.sio.emit(
                    "message", (complete_action["message"], timer, neighbors_dict))

                terminated[1] = True

            elif action == Action.request_item_info or action == Action.request_agent_info:

                if complete_action["robot"] > 0:
                    if action == Action.request_item_info:
                        message_str = self.ask_info_objects_str
                        sensing_state = self.State.wait_get_objects

                    elif action == Action.request_agent_info:
                        message_str = self.ask_info_agents_str
                        sensing_state = self.State.wait_get_agents

                    data['timer'] = time.time()

                    robot_data = self.neighbors_info[complete_action["robot"] - 1]
                    neighbors_dict = {
                        robot_data[0]: "human" if not robot_data[1] else "ai"}
                    data['agent_type'] = neighbors_dict[robot_data[0]]
                    print(message_str + str(robot_data[0]), neighbors_dict)
                    self.sio.emit("message", (message_str +
                                              str(robot_data[0]), timer, neighbors_dict))
                else:
                    truncated[1] = True

            else:
                # print("Not implemented sensing action", action)
                pass

            '''
            elif action.value >= Action.message_help_accept.value and action.value <= Action.message_cancel_request.value:


                if complete_action["robot"] > 0:

                    robot_data = self.neighbors_info[complete_action["robot"]-1]
                    neighbors_dict = {robot_data[0]: "human" if not robot_data[1] else "ai"}
                else:
                    neighbors_dict = {robot_data[0]: "human" if not robot_data[1] else "ai" for robot_data in self.neighbors_info}

                if action == Action.message_help_accept:
                    message = "I will help "
                elif action == Action.message_help_request_sensing:
                    if complete_action["item"] < len(self.object_info):
                        message = "I need help with sensing " + str(self.object_info[complete_action["item"]][0])
                    else:
                        truncated[1] = True
                elif action == Action.message_help_request_lifting:
                    if complete_action["item"] < len(self.object_info):
                        message = "I need help with lifting " + str(self.object_info[complete_action["item"]][0])
                    else:
                        truncated[1] = True
                elif action == Action.message_reject_request:
                    if complete_action["robot"] > 0:
                        message = "I cannot help you right now " + str(robot_data[0])
                    else:
                        truncated[1] = True
                elif action == Action.message_cancel_request:
                    message = "No more need for help"

                if not truncated[1]:
                    self.sio.emit("message", (message,neighbors_dict))
                    terminated[1] = True
            '''

        elif sensing_state == self.State.wait_sensing:

            if not self.waiting_output:

                occupancy_map = self.requested_output[0]
                objects_metadata = self.requested_output[1]
                objects_held = self.requested_output[2]
                danger_sensing_data = self.requested_output[3]
                action_status = self.requested_output[4]
                extra_status = self.requested_output[5]
                strength = self.requested_output[6]
                timer = self.requested_output[7]

                if any(extra_status):
                    terminated[1] = True

                    if extra_status[0]:  # Occupancy map received

                        sensing_output["occupancy_map"] = occupancy_map
                        sensing_output["objects_metadata"] = objects_metadata
                        
                        #Update objects locations
                        object_locations = np.where((occupancy_map == 2) | (occupancy_map == 4))
                        #object_locations = np.array([object_locations[0][:],object_locations[1][:]])

                        for ol_idx in range(len(object_locations[0])):
                            key = str(
                                object_locations[0][ol_idx]) + '_' + str(object_locations[1][ol_idx])

                            self.update_objects_info(
                                objects_metadata[key][0][0], timer, {}, [
                                    object_locations[0][ol_idx], object_locations[1][ol_idx]], objects_metadata[key][0][1], False)
                            '''
                            for ob_idx,ob in enumerate(self.object_info):
                                if ob[0] == objects_metadata[key][0][0]:
                                    self.object_info[ob_idx][3] = object_locations[0][ol_idx]
                                    self.object_info[ob_idx][4] = object_locations[1][ol_idx]
                                    self.object_info[ob_idx][5] = timer
                                    known_object = True
                                    break
                            if not known_object:
                                self.object_info.append([objects_metadata[key][0][0],objects_metadata[key][0][1],0,object_locations[0][ol_idx],object_locations[1][ol_idx],timer])
                            '''

                        # Update robots locations
                        robots_locations = np.where(occupancy_map == 3)
                        for ol_idx in range(len(robots_locations[0])):
                            key = str(
                                robots_locations[0][ol_idx]) + '_' + str(robots_locations[1][ol_idx])
                            self.update_neighbors_info(
                                objects_metadata[key][0], timer, [
                                    robots_locations[0][ol_idx], robots_locations[1][ol_idx]], False)
                            '''
                            for ob_idx,ob in enumerate(self.neighbors_info):

                                if ob[0] == str(objects_metadata[key][0]):
                                    self.neighbors_info[ob_idx][2] = robots_locations[0][ol_idx]
                                    self.neighbors_info[ob_idx][3] = robots_locations[1][ol_idx]
                                    self.neighbors_info[ob_idx][4] = timer
                                    break
                            '''

                    if extra_status[1]:  # Danger estimate received
                        for object_key in danger_sensing_data.keys():
                            '''
                            min_pos = self.map_config['edge_coordinate']
                            multiple = self.map_config['cell_size']
                            pos_new = [round((danger_sensing_data[object_key]['location'][0]+abs(min_pos))/multiple), round((danger_sensing_data[object_key]['location'][2]+abs(min_pos))/multiple)]
                            '''

                            self.update_objects_info(
                                object_key,
                                danger_sensing_data[object_key]['time'],
                                danger_sensing_data[object_key]['sensor'],
                                [
                                    danger_sensing_data[object_key]['location'][0],
                                    danger_sensing_data[object_key]['location'][2]],
                                danger_sensing_data[object_key]['weight'],
                                True)
                            '''
                            for ob_idx, ob in enumerate(self.object_info):
                                if ob[0] == object_key:
                                    self.object_info[ob_idx][2] = self.combine_danger_info(danger_sensing_data[object_key]['sensor'])
                                    self.object_info[ob_idx][3] = pos_new[0]
                                    self.object_info[ob_idx][4] = pos_new[1]
                                    self.object_info[ob_idx][5] = timer
                                    known_object = True
                                    break
                            if not known_object:
                                self.object_info.append([object_key,danger_sensing_data[object_key]['weight'],self.combine_danger_info(danger_sensing_data[object_key]['sensor']),pos_new[0],pos_new[1], timer])
                            '''

                    if extra_status[2]:  # Objects held
                        sensing_output["objects_held"] = int(
                            any(oh != 0 for oh in objects_held))

        elif sensing_state == self.State.wait_get_objects or sensing_state == self.State.wait_get_agents:

            if self.new_objects or self.new_neighbors:

                if data['agent_type'] == 'human':
                    coords_conversion = True
                else:
                    coords_conversion = False

                if sensing_state == self.State.wait_get_objects:

                    for ob_idx in range(len(self.new_objects)):
                        self.update_objects_info(
                            self.new_objects[ob_idx][0],
                            self.new_objects[ob_idx][5],
                            self.new_objects[ob_idx][2],
                            [
                                self.new_objects[ob_idx][3],
                                self.new_objects[ob_idx][4]],
                            self.new_objects[ob_idx][1],
                            coords_conversion)
                    self.new_objects = []

                elif sensing_state == self.State.wait_get_agents:

                    for ob_idx in range(len(self.new_neighbors)):
                        self.update_neighbors_info(
                            self.new_neighbors[ob_idx][0], self.new_neighbors[ob_idx][4], [
                                self.new_neighbors[ob_idx][2], self.new_neighbors[ob_idx][3]], coords_conversion)
                    self.new_agents = []

                terminated[1] = True
            elif time.time() - data['timer'] > self.confirm_time_threshold:
                truncated[1] = True

        if terminated[1] or truncated[1]:
            sensing_state = self.State.take_sensing_action

        return action_message, [
            state, sensing_state], data, sensing_output, terminated, truncated

    # Design an intelligent way of combining danger estimates

    def combine_danger_info(self, estimates):
        if estimates:
            key = list(estimates.keys())[0]
            return_value = estimates[key]['value']
        else:
            return_value = 0

        return return_value

    # When receiving info about objects, update your internal representation
    def update_objects_info(self, object_key, timer,
                            danger_data, position, weight, convert_coordinates):

        if convert_coordinates:
            position = self.convert_to_grid_coordinates(position)

        known_object = False
        for ob_idx, ob in enumerate(self.object_info):
            if ob[0] == object_key:

                if danger_data:
                    self.object_info[ob_idx][2].update(danger_data)

                if ob[5] < timer:  # If data is fresh
                    self.object_info[ob_idx][3] = float(position[0])
                    self.object_info[ob_idx][4] = float(position[1])
                    self.object_info[ob_idx][5] = float(timer)
                    known_object = True
                break
        if not known_object:

            self.object_info.append([object_key,int(weight),danger_data,float(position[0]),float(position[1]),float(timer)])
            self.object_key_to_index[object_key] = len(self.object_info)-1
          
    #When receiving info about robots, update your internal representation  
    def update_neighbors_info(self, agent_key, timer, position, convert_coordinates):

        if convert_coordinates:
            position = self.convert_to_grid_coordinates(position)

        for ob_idx, ob in enumerate(self.neighbors_info):

            if ob[0] == str(agent_key) and (ob[4] == -1 or ob[4] < timer):
                self.neighbors_info[ob_idx][2] = float(position[0])
                self.neighbors_info[ob_idx][3] = float(position[1])
                self.neighbors_info[ob_idx][4] = float(timer)
                break

    # This AI controller relies on having coordinates relative to the grid
    # world, which is not the same system the simulator uses
    def convert_to_grid_coordinates(self, location):

        min_pos = self.map_config['edge_coordinate']
        multiple = self.map_config['cell_size']
        pos_new = [round((location[0] + abs(min_pos[0])) / multiple),
                   round((location[1] + abs(min_pos[1])) / multiple)]

        return pos_new

    # Check movement limits

    def check_bounds(self, action_index, location, occupancy_map):

        if action_index == 0:  # Up
            if location[0] < occupancy_map.shape[0] - 1:
                location[0] += 1
        elif action_index == 1:  # Right
            if location[1] > 0:
                location[1] -= 1
        elif action_index == 2:  # Down
            if location[0] > 0:
                location[0] -= 1
        elif action_index == 3:  # Left
            if location[1] < occupancy_map.shape[1] - 1:
                location[1] += 1
        elif action_index == 4:  # Up Right
            if location[0] < occupancy_map.shape[0] - 1 and location[1] > 0:
                location += [1, -1]
        elif action_index == 5:  # Up Left
            if location[0] < occupancy_map.shape[0] - \
                    1 and location[1] < occupancy_map.shape[1] - 1:
                location += [1, 1]
        elif action_index == 6:  # Down Right
            if location[0] > 0 and location[1] > 0:
                location += [-1, -1]
        elif action_index == 7:  # Down Left
            if location[0] > 0 and location[1] < occupancy_map.shape[1] - 1:
                location += [-1, 1]

        return location

    #### WEBRTC SETUP ########################################################

    # This function is used as part of the setup of WebRTC

    async def offer(self, request):

        print("offer here")
        # async def offer_async(server_id, params):
        params = await request.json()
        print(params)

        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        self.robot_id = params["id"]
        pc = RTCPeerConnection()
        pc_id = "PeerConnection(%s)" % uuid.uuid4()
        self.pcs.add(pc)

        def log_info(msg, *args):
            logger.info(pc_id + " " + msg, *args)

        log_info("Created for %s", request.remote)

        # prepare local media

        if args.record_to:
            recorder = MediaRecorder(args.record_to)
        else:
            recorder = MediaBlackhole()

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log_info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        @pc.on("track")
        async def on_track(track):

            log_info("Track %s received", track.kind)

            if track.kind == "video":

                if args.record_to:
                    print("added record")
                    recorder.addTrack(self.relay.subscribe(track))

                if not self.tracks_received:
                    # processing_thread = threading.Thread(target=main_thread, args = (track, ))
                    # processing_thread.daemon = True
                    # processing_thread.start()
                    self.frame_queue = asyncio.Queue()
                    # print("waiting queue")
                    # track = await tracks_received.get()
                    print("waiting gather")
                    self.tracks_received += 1
                    await asyncio.gather(self.get_frame(track, self.frame_queue), self.actuate(self.frame_queue))
                # tracks_received.append(relay.subscribe(track))

                # print(tracks_received.qsize())

            @track.on("ended")
            async def on_ended():
                log_info("Track %s ended", track.kind)
                await recorder.stop()

        # handle offer
        await pc.setRemoteDescription(offer)
        await recorder.start()

        # send answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        print("offer", json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}))

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )


if __name__ == "__main__":
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
    parser.add_argument(
        "--use-occupancy",
        action='store_true',
        help="Use occupancy maps instead of images")
    parser.add_argument(
        "--address",
        default='https://172.17.15.69:4000',
        help="Address where our simulation is running")
    parser.add_argument(
        "--robot-number",
        default=1,
        help="Robot number to control")
    parser.add_argument(
        "--view-radius",
        default=0,
        help="When using occupancy maps, the view radius")

    args = parser.parse_args()

    logger = logging.getLogger("pc")

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    aicollab = AICollabEnv(args.use_occupancy,
                           args.view_radius,
                           int(args.robot_number),
                           args.host,
                           args.port,
                           args.address,
                           args.cert_file,
                           args.key_file)
    # aicollab.run(args.address,args.cert_file, args.key_file)
    # print("Finished here")
    # while not aicollab.setup_ready:
    #    time.sleep(1)
    aicollab.step(0)
