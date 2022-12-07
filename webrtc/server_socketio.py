import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import socketio
import cv2
from aiohttp import web
from av import VideoFrame
import aiohttp_cors

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription,RTCIceServer, RTCConfiguration
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()
client_number = 1

address = 'https://172.17.15.69:4000'

sio = socketio.Client(ssl_verify=False)
recording_file = "file.mp4"

@sio.event
def connect():
    print("I'm connected!")
    sio.emit("watcher", client_number)

@sio.event
def connect_error(data):
    print("The connection failed!")

@sio.event
def disconnect():
    print("I'm disconnected!")


@sio.event
def offer(server_id, description, new_client_id):
    asyncio.run(offer_async(server_id, description))

async def offer_async(server_id,params):

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    #config = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])
    pc = RTCPeerConnection()#config)
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)



    # prepare local media


    recorder = MediaRecorder(recording_file)


    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)


        if track.kind == "video":



            recorder.addTrack(relay.subscribe(track))

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

    sio.emit("answer", (server_id, json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        )))
    
    """
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )
    """


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    sio.connect(address)

