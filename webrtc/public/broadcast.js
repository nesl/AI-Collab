const peerConnections = {};
const config = {
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
  peerConnections[id].setRemoteDescription(description);
});



var current_idx = 1;
var connectedPeers = [];

socket.on("watcher", (id, client_number) => {
  const peerConnection = new RTCPeerConnection(config);
  peerConnections[id] = peerConnection;
  console.log(current_idx);
  console.log(id)
  connectedPeers.push(id);
  
  //For each connected client, we send both the third person view and a first person view of one of the user agents
  let stream = videoElements[client_number].srcObject;
  current_idx += 1;
  stream.getTracks().forEach(track => peerConnection.addTrack(track, stream));

  let stream2 = videoElements[0].srcObject;

  stream2.getTracks().forEach(track => peerConnection.addTrack(track, stream2));

  peerConnection.onicecandidate = event => {
    if (event.candidate) {
      socket.emit("candidate", id, event.candidate);
    }
  };

  peerConnection
    .createOffer()
    .then(sdp => peerConnection.setLocalDescription(sdp))
    .then(() => {
      socket.emit("offer", id, peerConnection.localDescription);
    });
});

socket.on("candidate", (id, candidate) => {
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

getDevices().then(gotDevices);


//getStream()
//  .then(getDevices)
//  .then(gotDevices);

function getDevices() {
  return navigator.mediaDevices.enumerateDevices();
}

async function gotDevices(deviceInfos) {
var v = 0;
const currentDiv = document.getElementById("videos_div");
	for (const deviceInfo of deviceInfos) {
		let videoElement = document.createElement('video')
		videoElement.playsinline = true;
		videoElement.autoplay = true;
		videoElement.muted = true;
		videoElement.id = 'videoSource' + v;
		
		videoElements.push(videoElement);
		const constraints = {
		video: { deviceId: deviceInfo.deviceId }
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
