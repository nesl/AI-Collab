import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

from magnebot import ActionStatus
import cv2
from aiohttp import web
from av import VideoFrame
import aiohttp_cors
import socketio

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

address = 'https://172.17.15.69:4000'


class State(Enum):
    waiting = 1
    moving_to_objective = 2

def turn_by(angle):
    return ["turn_by", angle]
def turn_to(target):
    return ["turn_to", target]
def move_by(distance):
    return ["move_by", distance]
def move_to(target):
    return ["move_to", target]
def reach_for(target, arm):
    return ["reach_for", target, arm]
def grasp(target, arm):
    return ["grasp", target, arm]
def drop(target, arm):
    return ["drop", target, arm]
def reset_arm(arm):
    return ["reset_arm", arm]
def reset_position():
    return ["reset_position"]
def rotate_camera(roll, pitch, yaw):
    return ["rotate_camera", roll, pitch, yaw]
def look_at(target):
    return ["look_at", target]
def move_camera(position):
    return ["move_camera", position]
def reset_camera():
    return ["reset_camera"]
def slide_torso(height):
    return ["slide_torso", height]




sio = socketio.Client(ssl_verify=False)
@sio.event
def connect():
    print("I'm connected!")
    sio.emit("watcher_ai", (client_number, "https://"+args.host+":"+str(args.port)+"/offer"))
    #asyncio.run(main_ai(tracks_received))


@sio.event
def connect_error(data):
    print("The connection failed!")

@sio.event
def disconnect():
    print("I'm disconnected!")
    
@sio.event
def set_goal(agent_id,obj_id):
    print("Received new goal")
    self.target[agent_id] = obj_id

neighbors = []

@sio.event
def neighbors_update(neighbors_list):
    neighbors = neighbors_list

objects = []

@sio.event
def objects_update(object_list):
    objects = object_list
    
messages = []

@sio.event
def message(message, source_agent_id):

    messages.append((source_agent_id,message))
    print("message", message, source_agent_id)

action_status = ActionStatus.ongoing

@sio.event
def ai_status(status):

    action_status = ActionStatus(status)
    print("status", ActionStatus(status))


async def get_frame(track,frame_queue):

    while True:
        frame = await track.recv()
        print("Processing frame")
        await frame_queue.put(frame)

async def actuate(frame_queue):
    global messages

    state = State.waiting
    data = ""

    def controller(state, messages, data):
        action_message = []

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
                        
        elif state == State.moving_to_objective:
            
            if action_status == ActionStatus.tipping:
                action_message.append(reset_position())
            elif action_status != ActionStatus.ongoing:
                #print("moving to objective")
                action_message.append(move_to(target=data, arrived_offset=1))

        return action_message

    while True:
        frame = await frame_queue.get()
        print("Frame actuated")
        #Call controller(frame)
        action_message = controller(state,messages, data)
        if action_message:
            sio.emit("ai_action", action_message)

"""
@sio.event
def offer(server_id, description, new_client_id):
    asyncio.run(offer_async(server_id, description))
"""

def main_thread(track):
    asyncio.run(main_ai(track))

async def main_ai(track):
    #global tracks_received,frame_queue

    
    #tracks_received = asyncio.Queue()
    
    frame_queue = asyncio.Queue()
    #print("waiting queue")
    #track = await tracks_received.get()
    print("waiting gather")
    await asyncio.gather(get_frame(track,frame_queue),actuate(frame_queue))


async def offer(request):
    global tracks_received,frame_queue,robot_id
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
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

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

