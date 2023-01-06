import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import json_numpy

from magnebot import ActionStatus
import cv2
from aiohttp import web
from av import VideoFrame
import aiohttp_cors
import socketio
import pdb

from enum import Enum

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()
tracks_received = 0

frame_queue = ""

client_number = 1
robot_id = 0
use_occupancy = False

address = ''



#### ROBOT API ######################################################################################

#Forwarded magnebot API from https://github.com/alters-mit/magnebot/blob/main/doc/manual/magnebot/actions.md\

def turn_by(angle, aligned_at=1):
    return ["turn_by", str(angle), "aligned_at=" + str(aligned_at)]
def turn_to(target, aligned_at=1):
    return ["turn_to", str(target), "aligned_at=" + str(aligned_at)]
def move_by(distance, arrived_at=0.1):
    return ["move_by", str(distance), "arrived_at=" + str(arrived_at)]
def move_to(target, arrived_at=0.1, aligned_at=1, arrived_offset=0):
    return ["move_to", str(target), "arrived_at=" + str(arrived_at), "aligned_at=" + str(aligned_at), "arrived_offset="+ str(arrived_offset)]
def reach_for(target, arm):
    return ["reach_for", str(target), str(arm)]
def grasp(target, arm):
    return ["grasp", str(target), str(arm)]
def drop(target, arm):
    return ["drop", str(target), str(arm)]
def reset_arm(arm):
    return ["reset_arm", str(arm)]
def reset_position():
    return ["reset_position"]
def rotate_camera(roll, pitch, yaw):
    return ["rotate_camera", str(roll), str(pitch), str(yaw)]
def look_at(target):
    return ["look_at", str(target)]
def move_camera(position):
    return ["move_camera", str(position)]
def reset_camera():
    return ["reset_camera"]
def slide_torso(height):
    return ["slide_torso", str(height)]


#### SOCKET IO message function definitions ########################################################

sio = socketio.Client(ssl_verify=False)

#When first connecting
@sio.event
def connect():
    print("I'm connected!")
    if not use_occupancy:
        sio.emit("watcher_ai", (client_number, use_occupancy, "https://"+args.host+":"+str(args.port)+"/offer"))
    else:
        sio.emit("watcher_ai", (client_number, use_occupancy, ""))
    #asyncio.run(main_ai(tracks_received))

#Receiving simulator's robot id
@sio.event
def watcher_ai(robot_id_r):
    global robot_id
    print("Received id", robot_id_r)
    robot_id = robot_id_r

    if use_occupancy: #When using only occupancy maps, run the main processing function here
        asyncio.run(main_ai())


#Receiving occupancy map
maps = []
map_ready = False
@sio.event
def occupancy_map(static_occupancy_map, object_type_coords_map, object_attributes_id):
    global maps, map_ready
    print("occupancy_map received")
    s_map = json_numpy.loads(static_occupancy_map)
    c_map = json_numpy.loads(object_type_coords_map)
    maps = (s_map,c_map, object_attributes_id)
    map_ready = True

#Connection error
@sio.event
def connect_error(data):
    print("The connection failed!")


#Disconnect
@sio.event
def disconnect():
    print("I'm disconnected!")

#Received a target object
@sio.event
def set_goal(agent_id,obj_id):
    print("Received new goal")
    #self.target[agent_id] = obj_id

#Update neighbor list
neighbors = []
@sio.event
def neighbors_update(neighbors_list):
    global neighbors
    print('neighbors update', neighbors_list)
    neighbors = neighbors_list

#Update object list
objects = []
@sio.event
def objects_update(object_list):
    global objects
    objects = object_list
    
#Receive messages from other agents
messages = []
@sio.event
def message(message, source_agent_id):
    global messages
    messages.append((source_agent_id,message))
    print("message", message, source_agent_id)

#Receive status updates of our agent
action_status = -1
@sio.event
def ai_status(status):
    global action_status
    action_status = ActionStatus(status)
    print("status", ActionStatus(status))




#### CONTROLLER DEFINITION #####################################################################

#Function that retrieves the newest occupancy map and makes some pre-processing if needed
async def get_map(frame_queue): 
    global map_ready

    while True:

        if map_ready: #Maps has all the occupancy maps and metadata
            map_ready = False
            await frame_queue.put(maps)
        else:
            await asyncio.sleep(0.01)
        
            
#Function that retrieves the newest video frame and makes some pre-processing if needed
async def get_frame(track,frame_queue):

    while True:
        frame = await track.recv()
        print("Processing frame")
        #frame.to_image() (av.VideoFrame)
        await frame_queue.put(frame)

#Controller states
class State(Enum):
    waiting = 1
    moving_to_objective = 2
    moving_to_objective_waiting = 3


#Function that waits for input and then makes the robot actuate 
async def actuate(frame_queue):
    global messages

    state = State.waiting
    data = ""

    #Robot controller: messages are received async from other robots and data is whatever needs to be saved for future calls to the controller
    def controller(state, messages, data):
        action_message = []

        #Wait until specific message is received from a human controlled robot
        if state == State.waiting:
            if messages:
                
                message = messages.pop(0)
                if "I need help with " in message[1]:
                    if "sensing" in message[1]:
                        pass
                    elif "lifting" in message[1]:
                        data = int(message[1][25:])
                        if sio:
                            sio.emit("message", ("I will help " + message[0], neighbors,str(robot_id)))
                        state = State.moving_to_objective
                        
        #Move close to object specified by human controlled robot
        elif state == State.moving_to_objective:
            #pdb.set_trace()
            if action_status == ActionStatus.tipping:
                action_message.append(reset_position())
            elif action_status != ActionStatus.ongoing:
                print("moving to objective")
                action_message.append(move_to(target=data, arrived_offset=0.5))
                state = State.moving_to_objective_waiting

        #Wait until the robot starts actuating
        elif state == State.moving_to_objective_waiting:

            if action_status == ActionStatus.ongoing or action_status == ActionStatus.success:
                print("waited for moving to objective")
                state = State.moving_to_objective


        return action_message,state,data

    while True:
        frame = await frame_queue.get()
        print("Frame actuated")
        
        action_message,state,data = controller(state,messages, data)

        if action_message: #Action message is an action to take by the robot that will be communicated to the simulator
            print("action", action_message)
            sio.emit("ai_action", (action_message, str(robot_id)))


#These two next functions are used to initiate the control for the robot when using only occupancy maps
def main_thread():
    asyncio.run(main_ai())

async def main_ai():
    #global tracks_received,frame_queue

    
    #tracks_received = asyncio.Queue()
    
    frame_queue = asyncio.Queue()
    #print("waiting queue")
    #track = await tracks_received.get()
    print("waiting gather")
    #await asyncio.gather(get_frame(track,frame_queue),actuate(frame_queue))
    await asyncio.gather(get_map(frame_queue),actuate(frame_queue))



#### WEBRTC SETUP #####################################################################################

#This function is used as part of the setup of WebRTC
async def offer(request):
    global tracks_received,frame_queue,robot_id

    print("offer here")
    #async def offer_async(server_id, params):
    params = await request.json()
    print(params)
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    robot_id = params["id"]
    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

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
            pcs.discard(pc)

   

    @pc.on("track")
    async def on_track(track):
        global tracks_received,frame_queue

        log_info("Track %s received", track.kind)


        if track.kind == "video":


            if args.record_to:
                print("added record")
                recorder.addTrack(relay.subscribe(track))

            if not tracks_received:
                #processing_thread = threading.Thread(target=main_thread, args = (track, ))
                #processing_thread.daemon = True
                #processing_thread.start()
                frame_queue = asyncio.Queue()
                #print("waiting queue")
                #track = await tracks_received.get()
                print("waiting gather")
                tracks_received += 1
                await asyncio.gather(get_frame(track,frame_queue),actuate(frame_queue))
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





#### MAIN & SETUP OF HTTP SERVER ##############################################################################

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

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
    parser.add_argument("--robot_number", default=1, help="Robot number to control")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    use_occupancy = args.use_occupancy 

    address = args.address
    client_number = int(args.robot_number)

    if use_occupancy:
        sio.connect(address)
        main_thread()
    else:
        if args.cert_file:
            #ssl_context = ssl.SSLContext()
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(args.cert_file, args.key_file)
        else:
            ssl_context = None

        app = web.Application()
        

        app.on_shutdown.append(on_shutdown)
        #app.router.add_get("/", index)
        #app.router.add_get("/client.js", javascript)
        app.router.add_post("/offer", offer)

        cors = aiohttp_cors.setup(app, defaults={
          "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*"
          )
        })

        for route in list(app.router.routes()):
            cors.add(route)

        sio.connect(address)
        web.run_app(
            app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
        )

