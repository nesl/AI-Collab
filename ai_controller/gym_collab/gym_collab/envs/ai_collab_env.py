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


class AICollabEnv(gym.Env):

    #### MAIN & SETUP OF HTTP SERVER ##############################################################################


    def __init__(self, use_occupancy,view_radius, client_number, host, port, address, cert_file, key_file):

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
        self.objects_held = []

        
        

        #### SOCKET IO message function definitions ########################################################

        self.sio = socketio.Client(ssl_verify=False)

        #When first connecting
        @self.sio.event
        def connect():
            print("I'm connected!")
            if not self.use_occupancy:
                self.sio.emit("watcher_ai", (self.client_number, self.use_occupancy, "https://"+self.host+":"+str(self.port)+"/offer", 0, 0))
            else:
                self.sio.emit("watcher_ai", (self.client_number, self.use_occupancy, "", self.view_radius, self.centered_view))
            #asyncio.run(main_ai(tracks_received))

        #Receiving simulator's robot id
        @self.sio.event
        def watcher_ai(robot_id_r, occupancy_map_config):

            print("Received id", robot_id_r)
            self.robot_id = robot_id_r

            if self.use_occupancy: #When using only occupancy maps, run the main processing function here
                self.map_config = occupancy_map_config
                #asyncio.run(self.main_ai())
                self.gym_setup()
                self.setup_ready = True


        #Receiving occupancy map
        self.maps = []
        self.map_ready = False
        self.map_config = {}
        @self.sio.event
        def occupancy_map(object_type_coords_map, object_attributes_id, objects_held):

            #print("occupancy_map received")
            #s_map = json_numpy.loads(static_occupancy_map)
            c_map = json_numpy.loads(object_type_coords_map)
            self.maps = (c_map, object_attributes_id)
            self.objects_held = objects_held
            self.map_ready = True
            
            #print(c_map)


        #Connection error
        @self.sio.event
        def connect_error(data):
            print("The connection failed!")


        #Disconnect
        @self.sio.event
        def disconnect():
            print("I'm disconnected!")

        #Received a target object
        @self.sio.event
        def set_goal(agent_id,obj_id):
            print("Received new goal")
            #self.target[agent_id] = obj_id

        #Update neighbor list
        self.neighbors = {}
        self.new_neighbors = False
        @self.sio.event
        def neighbors_update(neighbors_dict):

            print('neighbors update', neighbors_dict)
            self.neighbors.update(neighbors_dict)
            self.new_neighbors = True

        #Update object list
        self.objects = {}
        self.new_objects = False
        @self.sio.event
        def objects_update(object_dict):

            
            self.objects.update(object_dict)
            
            print("objects_update", object_dict)
            self.new_objects = True
            
        #Receive messages from other agents
        self.messages = []
        @self.sio.event
        def message(message, source_agent_id):

            self.messages.append((source_agent_id,message))
            print("message", message, source_agent_id)

        #Receive status updates of our agent
        self.action_status = -1
        @self.sio.event
        def ai_status(status):

            self.action_status = ActionStatus(status)
            print("status", ActionStatus(status))


        self.run(address, cert_file, key_file)
        
        while not self.setup_ready:
            time.sleep(1)

        
    def run(self, address, cert_file, key_file):
    
        if self.use_occupancy:
            self.sio.connect(address)
            #main_thread()
        else:
            if cert_file:
                #ssl_context = ssl.SSLContext()
                ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
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
            #app.router.add_get("/", index)
            #app.router.add_get("/client.js", javascript)
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



    #### GYM SETUP ######################################################################################
    
    def gym_setup(self):
    
        map_size = self.map_config['num_cells'][0]
        self.action_space = spaces.Discrete(18)
        
        #self.observation_space = spaces.Box(0, map_size - 1, shape=(2,), dtype=int)
        
        self.observation_space = spaces.Dict(
            {
                "frame" : spaces.Box(0, map_size - 1, shape=(2,), dtype=int),
                "objects_held" : spaces.Discrete(2)
            }
        )
        
    
    def step(self, action):
        
        world_state, objects_info, neighbors_info, objects_held = self.take_action(action)
        #observed_state = {"frame": world_state, "message": self.messages}
        observation = {"frame": world_state[0], "objects_held": int(any(objects_held))} #Occupancy map
        info = {}
        info['objects'] = objects_info
        info['neighbors'] = neighbors_info
        info['messages'] = self.messages
        info['map_metadata'] = world_state[1]
        info['objects_held'] = objects_held
        
        reward = 0
        terminated = False
        


        return observation, reward, terminated, False, info
        
    def reset(self, seed=None, options=None):
    
        super().reset(seed=seed)
        observation = {"frame": [], "objects_held": 0}
        info = {}
        
        return observation, info
        
    #### ROBOT API ######################################################################################

    #Forwarded magnebot API from https://github.com/alters-mit/magnebot/blob/main/doc/manual/magnebot/actions.md\

    def turn_by(self, angle, aligned_at=1):
        return ["turn_by", str(angle), "aligned_at=" + str(aligned_at)]
    def turn_to(self, target, aligned_at=1):
        return ["turn_to", str(target), "aligned_at=" + str(aligned_at)]
    def move_by(self, distance, arrived_at=0.1):
        return ["move_by", str(distance), "arrived_at=" + str(arrived_at)]
    def move_to(self, target, arrived_at=0.1, aligned_at=1, arrived_offset=0):
        return ["move_to", str(target), "arrived_at=" + str(arrived_at), "aligned_at=" + str(aligned_at), "arrived_offset="+ str(arrived_offset)]
    def reach_for(self, target, arm):
        return ["reach_for", str(target), str(arm)]
    def grasp(self, target, arm):
        return ["grasp", str(target), str(arm)]
    def drop(self, target, arm):
        return ["drop", str(target), str(arm)]
    def reset_arm(self, arm):
        return ["reset_arm", str(arm)]
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
    def danger_sensor_reading(self, robot_id):
        return ["danger_sensor_reading", str(robot_id)]


    




    #### CONTROLLER DEFINITION #####################################################################

    #Function that retrieves the newest occupancy map and makes some pre-processing if needed
    async def get_map(self, frame_queue): 


        while True:

            if self.map_ready: #Maps has all the occupancy maps and metadata
                self.map_ready = False
                await self.frame_queue.put(self.maps)
            else:
                await asyncio.sleep(0.01)
            
                
    #Function that retrieves the newest video frame and makes some pre-processing if needed
    async def get_frame(self, track,frame_queue):

        while True:
            frame = await track.recv()
            print("Processing frame")
            #frame.to_image() (av.VideoFrame)
            await self.frame_queue.put(frame)

    #Controller states
    class State(Enum):
        take_action = 1
        waiting_ongoing = 2
        grasping_object = 3
        reseting_arm = 4
        reverse_after_dropping = 5
        wait_objects = 6
        action_end = 7
        
    class Action(Enum):
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




    def take_action(self, action):
    
        state = self.State.take_action
        data = {}
        terminated = False
        objects_obs = []
        neighbors_obs = []
        
        action = self.Action(action)
        print(action)
            
        while not self.map_ready:
            pass

        while not terminated:
            #frame = await self.frame_queue.get()
            if self.map_ready:
                frame = self.maps
                self.map_ready = False
            #print("Frame actuated")
            
            action_message,state,data,terminated = self.controller(action, frame, state, data)

            if action_message: #Action message is an action to take by the robot that will be communicated to the simulator
                print("action", action_message)
                self.sio.emit("ai_action", (action_message, str(self.robot_id)))
                
        if self.new_objects:
            objects_obs = self.objects
        if self.new_neighbors:
            neighbors_obs = self.neighbors
        
        self.new_objects = False
        self.new_neighbors = False
            
        return frame, objects_obs, neighbors_obs, self.objects_held
    
    
    #Function that waits for input and then makes the robot actuate 
    async def actuate(self, frame_queue):


        state = self.State.take_action
        data = {}

        #Robot controller: messages are received async from other robots and data is whatever needs to be saved for future calls to the controller
        def simple_controller(frame, state, messages, data):
            action_message = []

            #Wait until specific message is received from a human controlled robot
            if state == self.State.waiting:
                if messages:
                    
                    message = messages.pop(0)
                    if "I need help with " in message[1]:
                        if "sensing" in message[1]:
                            pass
                        elif "lifting" in message[1]:
                            data = int(message[1][25:])
                            if self.sio:
                                self.sio.emit("message", ("I will help " + message[0], self.neighbors,str(self.robot_id)))
                            state = self.State.moving_to_objective
                            
            #Move close to object specified by human controlled robot
            elif state == self.State.moving_to_objective:
                #pdb.set_trace()
                if self.action_status == ActionStatus.tipping:
                    action_message.append(reset_position())
                elif self.action_status != ActionStatus.ongoing:
                    print("moving to objective")
                    action_message.append(move_to(target=data, arrived_offset=0.5))
                    state = self.State.moving_to_objective_waiting

            #Wait until the robot starts actuating
            elif state == self.State.moving_to_objective_waiting:

                if self.action_status == ActionStatus.ongoing or self.action_status == ActionStatus.success:
                    print("waited for moving to objective")
                    state = self.State.moving_to_objective


            return action_message,state,data
            
        
            
        while not self.map_ready:
            pass

        while True:
            #frame = await self.frame_queue.get()
            if self.map_ready:
                frame = self.maps
                self.map_ready = False
            #print("Frame actuated")
            
            action_message,state,data,terminated = self.controller(frame, state, data)

            if action_message: #Action message is an action to take by the robot that will be communicated to the simulator
                print("action", action_message)
                self.sio.emit("ai_action", (action_message, str(self.robot_id)))


    #Only works for occupancy maps not centered in magnebot
    def controller(self, action, frame, state, data):

        
        action_message = []
        movement_commands = 8
        grab_commands = 16
        
        occupancy_map = frame[0]
        objects_metadata = frame[1]
        terminated = False
        

        if state == self.State.take_action:
            if self.action_status != ActionStatus.ongoing:
            
                #print(occupancy_map[10:20,15:30])
            
                self.action_status = -1
                ego_location = np.where(occupancy_map == 5)
                #pdb.set_trace()
                ego_location = np.array([ego_location[0][0],ego_location[1][0]])


                if action.value < movement_commands:
                
                    action_index = [self.Action.move_up,self.Action.move_right,self.Action.move_down,self.Action.move_left,self.Action.move_up_right,self.Action.move_up_left,self.Action.move_down_right,self.Action.move_down_left].index(action)
                    
                    original_location = np.copy(ego_location)
                    
                    ego_location = self.check_bounds(action_index, ego_location, occupancy_map)


                    if not np.array_equal(ego_location,original_location):
                        target_coordinates = np.array(self.map_config['edge_coordinate']) + ego_location*self.map_config['cell_size']
                        target = {"x": target_coordinates[0],"y": 0, "z": target_coordinates[1]}
                        state = self.State.waiting_ongoing
                        data["next_state"] = self.State.action_end
                        action_message.append(self.move_to(target=target))
                    else:
                        print("Movement not possible")
                        terminated = True
                    
                elif action.value < grab_commands:    
                
                    object_location = np.copy(ego_location)
                    
                    action_index = [self.Action.grab_up,self.Action.grab_right,self.Action.grab_down,self.Action.grab_left,self.Action.grab_up_right,self.Action.grab_up_left,self.Action.grab_down_right,self.Action.grab_down_left].index(action)
                    
                    object_location = self.check_bounds(action_index, object_location, occupancy_map)
                    
                    
                    if (not np.array_equal(object_location,ego_location)) and occupancy_map[object_location[0],object_location[1]] == 2:
                        #object_location = np.where(occupancy_map == 2)
                        #key = str(object_location[0][0]) + str(object_location[1][0])
                        key = str(object_location[0]) + str(object_location[1])
                        action_message.append(self.turn_to(objects_metadata[key][0]))
                       
                        state = self.State.waiting_ongoing
                        data["next_state"] = self.State.grasping_object
                        data["object"] = objects_metadata[key][0]
                    else:
                        print("No object to grab")
                        terminated = True
                    
                elif action == self.Action.drop_object:

                    if "object" in data:
                        action_message.append(self.drop(data["object"], Arm.left))
                       
                        state = self.State.waiting_ongoing
                        data["next_state"] = self.State.reverse_after_dropping
                        del data["object"]
                    else:
                        print("No object to drop")
                        terminated = True
                    
                elif action == self.Action.danger_sensing:
                    action_message.append(self.danger_sensor_reading(self.robot_id))
                    state = self.State.wait_objects
                    
                else:
                    print("Not implemented", action)
                    terminated = True
                
                    
                    
                
        elif state == self.State.waiting_ongoing:

            if self.action_status == ActionStatus.ongoing or self.action_status == ActionStatus.success:
                print("waiting")
                state = data["next_state"]
                    
        elif state == self.State.grasping_object:
             if self.action_status != ActionStatus.ongoing:
                state = self.State.waiting_ongoing
                print("waited to grasp objective")
                action_message.append(self.grasp(data["object"], Arm.left))
                data["next_state"] = self.State.reseting_arm
            
        elif state == self.State.reseting_arm:

            if self.action_status != ActionStatus.ongoing:
                print("waited to reset arm")
                action_message.append(self.reset_arm(Arm.left))
                state = self.State.waiting_ongoing
                data["next_state"] = self.State.action_end
                
        elif state == self.State.reverse_after_dropping:
            if self.action_status != ActionStatus.ongoing:
                print("waited to reverse after dropping")
                action_message.append(self.move_by(-0.5))
                state = self.State.waiting_ongoing
                data["next_state"] = self.State.action_end
                
        elif state == self.State.wait_objects:
            if self.new_objects:
                terminated = True
                
        elif state == self.State.action_end:
            if self.action_status != ActionStatus.ongoing:  
                terminated = True
            
            
        return action_message,state,data,terminated


    def check_bounds(self, action_index, location, occupancy_map):
    
        if action_index == 0: #Up
            if location[0] < occupancy_map.shape[0]-1:
                location[0] += 1
        elif action_index == 1: #Right
            if location[1] > 0:
                location[1] -= 1
        elif action_index == 2: #Down
            if location[0] > 0:
                location[0] -= 1
        elif action_index == 3: #Left
            if location[1] < occupancy_map.shape[1]-1:
                location[1] += 1
        elif action_index == 4: #Up Right
            if location[0] < occupancy_map.shape[0]-1 and location[1] > 0:
                location += [1,-1]
        elif action_index == 5: #Up Left
            if location[0] < occupancy_map.shape[0]-1 and location[1] < occupancy_map.shape[1]-1:
                location += [1,1]
        elif action_index == 6: #Down Right
            if location[0] > 0 and location[1] > 0:
                location += [-1,-1]
        elif action_index == 7: #Down Left
            if location[0] > 0 and location[1] < occupancy_map.shape[1]-1:
                location += [-1,1]
                
        return location


    #These two next functions are used to initiate the control for the robot when using only occupancy maps
    def main_thread(self):
        asyncio.run(self.main_ai())

    async def main_ai(self):


        self.gym_setup()
        #tracks_received = asyncio.Queue()
        
        self.frame_queue = asyncio.Queue()
        #print("waiting queue")
        #track = await tracks_received.get()
        print("waiting gather")
        #await asyncio.gather(get_frame(track,frame_queue),actuate(frame_queue))
        #await asyncio.gather(get_map(frame_queue),actuate(frame_queue))
        #await asyncio.gather(self.actuate(self.frame_queue))



    #### WEBRTC SETUP #####################################################################################

    #This function is used as part of the setup of WebRTC
    async def offer(self, request):


        print("offer here")
        #async def offer_async(server_id, params):
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
                    #processing_thread = threading.Thread(target=main_thread, args = (track, ))
                    #processing_thread.daemon = True
                    #processing_thread.start()
                    self.frame_queue = asyncio.Queue()
                    #print("waiting queue")
                    #track = await tracks_received.get()
                    print("waiting gather")
                    self.tracks_received += 1
                    await asyncio.gather(self.get_frame(track,self.frame_queue),self.actuate(self.frame_queue))
                #tracks_received.append(relay.subscribe(track))
                
                #print(tracks_received.qsize())
            
                
                

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
        print("offer",json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}))
        
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
    parser.add_argument("--use-occupancy", action='store_true', help="Use occupancy maps instead of images")
    parser.add_argument("--address", default='https://172.17.15.69:4000', help="Address where our simulation is running")
    parser.add_argument("--robot-number", default=1, help="Robot number to control")
    parser.add_argument("--view-radius", default=0, help="When using occupancy maps, the view radius")

    args = parser.parse_args()
    
    logger = logging.getLogger("pc")

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    aicollab = AICollabEnv(args.use_occupancy,args.view_radius, int(args.robot_number), args.host, args.port, args.address,args.cert_file, args.key_file)
    #aicollab.run(args.address,args.cert_file, args.key_file)
    #print("Finished here")
    #while not aicollab.setup_ready:
    #    time.sleep(1)
    aicollab.step(0)
