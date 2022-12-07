import socketio
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCIceCandidate, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRecorder, MediaRelay
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
import asyncio
import re

address = 'https://172.17.15.69:4000'

sio = socketio.Client(ssl_verify=False)
client_number = 1
client_id = ""


#config = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])

peerConnection = RTCPeerConnection() #configuration=config)

relay = MediaRelay()
recorder = MediaRecorder("record.mp4")

regexpression = re.compile("a=(candidate:.*)\r")

rtcicecandidates = []

video = []

count_offer = 0
count_candidate = 0
            
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
    asyncio.run(offer_async(server_id, description, new_client_id))


async def offer_async(server_id, description, new_client_id):

    global peerConnection, count_offer, count_candidate

    count_offer += 1
    
    
    print("heloo", peerConnection, description, count_offer, count_candidate)

    @peerConnection.on("track")
    def on_track(track):
        recorder.addTrack(relay.subscribe(track))
        print("adding track")

    @peerConnection.on("iceconnectionstatechange")
    def on_iceconnectionstatechange():
        print(f'ICE connection state is {peerConnection.iceConnectionState}')

    @peerConnection.on("icegatheringstatechange")
    def on_icegatheringstatechange():
        print(f'ICE gathering state is {peerConnection.iceGatheringState}')


    """
    @peerConnection.on("icegatheringstatechange")
    def on_icegatheringstatechange():
        
        if peerConnection.iceGatheringState == 'complete':
            print("icecandiadate")
            candidates = re.findall(regexpression,peerConnection.localDescription.sdp)
            for c in candidates:
                print(candidates)
            # candidates are ready
            #candidates = peerConnection.sctp.transport.transport.iceGatherer.getLocalCandidates()
            # add ice candidate iteratively
            #sio.emit("candidate", server_id, event.candidate)
            #print("candidates", candidates)
    """

    await peerConnection.setRemoteDescription(RTCSessionDescription(description['sdp'],description['type']))
    await recorder.start()
    await peerConnection.setLocalDescription(await peerConnection.createAnswer())
    print("set local description")
    
        

    sio.emit("answer", (server_id, {'type': peerConnection.localDescription.type, 'sdp': peerConnection.localDescription.sdp}))

    candidates = re.findall(regexpression,peerConnection.localDescription.sdp)
    for c in candidates:
        icecandidate = candidate_from_sdp(c.split(":", 1)[1])
        icecandidate.protocol = icecandidate.protocol.upper()
        for c2 in rtcicecandidates:
            
            if icecandidate.ip == c2.ip and icecandidate.protocol == c2.protocol and icecandidate.type == c2.type:
                icecandidate.foundation = c2.foundation
                new_ice_candidate = "candidate:" + candidate_to_sdp(icecandidate)
                rtcicecandidate = {'candidate':new_ice_candidate, 'sdpMLineIndex': 0, 'sdpMid': "0"}
                print(rtcicecandidate)
                sio.emit("candidate", (server_id, rtcicecandidate))
                break

        

@sio.event
def candidate(server_id, candidate):
    asyncio.run(candidate_async(server_id, candidate))


async def candidate_async(server_id, candidate):
    global peerConnection, count_offer, count_candidate
    count_candidate +=1

    parameters = candidate['candidate'][10:].split()
    print(candidate, count_offer, count_candidate, parameters)

    #if(len(parameters) > 7 and parameters[2] == 'UDP' and parameters[7] == 'host'):
    #rtcicecandidate = RTCIceCandidate(foundation=int(parameters[0]),component=parameters[1],protocol=parameters[2],priority=int(parameters[3]),ip=parameters[4],port=int(parameters[5]),type=parameters[7], sdpMid=candidate['sdpMid'], sdpMLineIndex=candidate['sdpMLineIndex'])
    if(candidate.get('candidate')):
        rtcicecandidate = candidate_from_sdp(candidate.get('candidate').split(":", 1)[1])
        rtcicecandidate.sdpMid = candidate.get('sdpMid')
        rtcicecandidate.sdpMLineIndex = candidate.get('sdpMLineIndex')
        rtcicecandidates.append(rtcicecandidate)
        await peerConnection.addIceCandidate(rtcicecandidate)

"""
@sio.event
def broadcaster():
    sio.emit("watcher", client_number)
"""




sio.connect(address)
