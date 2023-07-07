const peerConnections = {};
const config = {
  sdpSemantics: 'unified-plan',

  iceServers: [
    { 
      "urls": "stun:stun.l.google.com:19302",
    },
    // { 
    //   "urls": "turn:TURN_IP?transport=tcp",
    //   "username": "TURN_USERNAME",
    //   "credential": "TURN_CREDENTIALS"
    // }
  ]
};

const socket = io.connect(window.location.origin);

socket.on("answer", (id, description) => {
  console.log("description", description)
  //peerConnections[id].setRemoteDescription(JSON.parse(description));
  peerConnections[id].setRemoteDescription(description);
});



var current_idx = 1;
var connectedPeers = [];

socket.on("watcher", (id, client_number) => { //When human controlled robot connects, setup WebRTC and forward correct video track
  const peerConnection = new RTCPeerConnection(config);
  peerConnections[id] = peerConnection;
  console.log(current_idx);
  console.log(id)
  connectedPeers.push(id);

  
  //For each connected client, we send both the third person view and a first person view of one of the user agents
  let stream = videoElements[client_number].srcObject;
  current_idx += 1;
  stream.getTracks().forEach(track => peerConnection.addTrack(track, stream));

  //let stream2 = videoElements[0].srcObject;

  //stream2.getTracks().forEach(track => peerConnection.addTrack(track, stream2));

  peerConnection.onicecandidate = event => {
    if (event.candidate) {
      console.log(event.candidate)
      socket.emit("candidate", id, event.candidate);
    }
  };
  peerConnection.onicecandidateerror = event => {
    console.log("error", event);
    
  };

  peerConnection
    .createOffer()
    .then(sdp => peerConnection.setLocalDescription(sdp))
    .then(() => {
      socket.emit("offer", id, peerConnection.localDescription);
    });
    

  
});

socket.on("watcher_ai", (id, client_number, server_address, robot_id) => { //When AI controlled robot connects, setup WebRTC and forward correct video track
  const peerConnection = new RTCPeerConnection(config);
  peerConnections[id] = peerConnection;
  console.log(current_idx);
  console.log(id)
  connectedPeers.push(id);
  console.log("gelloS", server_address)
  //For each connected client, we send both the third person view and a first person view of one of the user agents
  let stream = videoElements[client_number].srcObject;
  current_idx += 1;
  stream.getTracks().forEach(track => peerConnection.addTrack(track, stream));

  //let stream2 = videoElements[0].srcObject;

  //stream2.getTracks().forEach(track => peerConnection.addTrack(track, stream2));

  peerConnection.onicecandidate = event => {
    if (event.candidate) {
      console.log(event.candidate)
      socket.emit("candidate", id, event.candidate);
    }
  };
  peerConnection.onicecandidateerror = event => {
    console.log("error", event);
    
  };

  peerConnection
    .createOffer()
    .then(sdp => peerConnection.setLocalDescription(sdp))
    
    .then(function() {
        // wait for ICE gathering to complete
        
        return new Promise(function(resolve) {
            if (peerConnection.iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (peerConnection.iceGatheringState === 'complete') {
                        peerConnection.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                peerConnection.addEventListener('icegatheringstatechange', checkState);
            }
        });
    })
    .then(function() {
        var offer = peerConnection.localDescription;
        var codec;
        console.log("sending to ", server_address)
        return fetch(server_address, { //'https://172.17.15.69:8080/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                id: robot_id
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then(function(response) {
        console.log(response)
        return response.json();
    }).then(function(answer) {

        return peerConnection.setRemoteDescription(answer);
    }).catch(function(e) {
        alert(e);
    });
    /*
    .then(() => {
      socket.emit("offer", id, peerConnection.localDescription);
    });
    */

  
});


socket.on("candidate", (id, candidate) => { //WebRTC exchange candidates
  console.log("new_candidate", candidate)
  peerConnections[id].addIceCandidate(new RTCIceCandidate(candidate));
});

socket.on("disconnectPeer", id => {
  peerConnections[id].close();
  delete peerConnections[id];
});

socket.on("key", (id,key) => {
  console.log(id)
  console.log(key)
});



window.onunload = window.onbeforeunload = () => {
  socket.close();
};

const videoElements = [];
//deviceInfos = navigator.mediaDevices.enumerateDevices();

var first_video_idx = 0;
socket.on("simulator", (video_idx) => {
    console.log("simulator")
    first_video_idx = video_idx;
    getDevices().then(gotDevices);
});



//getStream()
//  .then(getDevices)
//  .then(gotDevices);

function getDevices() {
  return navigator.mediaDevices.enumerateDevices();
}

//Get simulated webcams and their video streams

async function gotDevices(deviceInfos) {
    var v = 0;
    const currentDiv = document.getElementById("videos_div");

    /*
    var first_video = 0;
    const searchParams = new URLSearchParams(window.location.search);
    
    if (searchParams.has('v_idx')){
        first_video = parseInt(searchParams.get('v_idx'));
    }
    */

    for (let i = first_video_idx; i < deviceInfos.length; i++) {
    
    		if(!(deviceInfos[i]["kind"] === "videoinput")){
    			continue;
    		}
    
		let videoElement = document.createElement('video')
		videoElement.setAttribute("playsinline",true);
		videoElement.setAttribute("autoplay",true);
		//videoElement.playsinline = true;
		//videoElement.autoplay = true;
		videoElement.muted = true;
		videoElement.id = 'videoSource' + v;
		
		videoElements.push(videoElement);
		const constraints = {
		video: { deviceId: deviceInfos[i].deviceId }
		};
		
		//console.log(constraints)
		
		await navigator.mediaDevices
		.getUserMedia(constraints)
		.then(function(result){return gotStream(result, v)})
		.catch(handleError);
		document.body.insertBefore(videoElement, currentDiv);
		v += 1;

	}
/*
  window.deviceInfos = deviceInfos;
  for (const deviceInfo of deviceInfos) {
    const option = document.createElement("option");
    option.value = deviceInfo.deviceId;

    if (deviceInfo.kind === "audioinput") {
      option.text = deviceInfo.label || `Microphone ${audioSelect.length + 1}`;
      audioSelect.appendChild(option);
    } else if (deviceInfo.kind === "videoinput") {
      option.text = deviceInfo.label || `Camera ${videoSelect.length + 1}`;
      videoSelect.appendChild(option);
    }
  }
*/
}


function gotStream(stream,idx) {
  /*
  window.stream = stream;
  audioSelect.selectedIndex = [...audioSelect.options].findIndex(
    option => option.text === stream.getAudioTracks()[0].label
  );
  videoSelect.selectedIndex = [...videoSelect.options].findIndex(
    option => option.text === stream.getVideoTracks()[0].label
  );
  */
  //console.log(idx)
  //console.log(videoElements)
  videoElements[idx].srcObject = stream;
  socket.emit("broadcaster");
}

function handleError(error) {
  console.error("Error: ", error);
}

function manualReset(){
  socket.emit("reset");
}

socket.emit("broadcaster_load");
