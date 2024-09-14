//https://gabrieltanner.org/blog/webrtc-video-broadcast/
const peerConnections = {};


const config = {
  sdpSemantics: 'unified-plan',

  iceServers: [
    //{ 
    //  urls: ["stun:stun.l.google.com:19302"]
    //},
    { 
       "urls": "turn:44.203.1.205:3478?transport=tcp",
       "username": config_api.username,
       "credential": config_api.password
    }
  ]
};

/*
var config = {};
(async() => {
  const response = await fetch("https://nesl.metered.live/api/v1/turn/credentials?apiKey=" + config_api.API_KEY);
  const iceServers = await response.json();
  config.iceServers = iceServers
})();
*/
const socket = io.connect(window.location.origin);

socket.on("answer", (id, description) => {
  console.log("description", description)
  //peerConnections[id].setRemoteDescription(JSON.parse(description));
  peerConnections[id].setRemoteDescription(description);
});

var desc_answer = [];
socket.on("answer_ai", (id, description) => {
  console.log("description", description)
  answer = JSON.parse(description);
  //peerConnections[id].setRemoteDescription(JSON.parse(description));
  desc_answer = answer;
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

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}


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

  console.log("peerconnection 1", peerConnection);
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
    }) /*
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
	console.log(answer)
        return peerConnection.setRemoteDescription(answer);
    }).catch(function(e) {
        alert(e);
    });
    */
    .then(() => {
      var offer = peerConnection.localDescription;
      var resp = JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                id: robot_id
            });
      return new Promise(function(resolve) {
	      socket.emit("offer_ai", id, resp, function process_answer(description){
		      console.log("description", description);
		      answer = JSON.parse(description);
		      //peerConnections[id].setRemoteDescription(JSON.parse(description));
		      desc_answer = answer;
		      resolve();
		      });
      });
    }).then(function() {
    	answer = desc_answer
	console.log("description", answer)
        return peerConnection.setRemoteDescription(answer);
    }).catch(function(e) {
        alert(e);
    });
    


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

var map_config = {};
var first_video_idx = 0;
var neighbors_list_store = [];
socket.on("simulator", (video_idx, config_options) => {
    console.log("simulator")
    first_video_idx = video_idx;
    getDevices().then(gotDevices);
    
    map_config = config_options;
    
    var neighbor_info_div = document.getElementById("collapsible_nearby_team_members");
    
    var div_element = document.createElement("div");
    div_element.setAttribute("class", "wrapper");
    var input_element = document.createElement("input");
    input_element.setAttribute("type", "radio");
    input_element.setAttribute("id", "All");
    input_element.setAttribute("name", "neighbors");
    input_element.setAttribute("value", "All");
    input_element.setAttribute("checked", true);
    var label_element = document.createElement("label");
    label_element.setAttribute("for", "All");
    label_element.style.color = "black";
    label_element.appendChild(document.createTextNode("All"));
    
    
    div_element.appendChild(input_element);	
    div_element.appendChild(label_element);
    neighbor_info_div.appendChild(div_element);
    
    for(um_idx in map_config['all_robots']){
    
        var agent_type  = 0;
        
        if(map_config['all_robots'][um_idx][1] === 'ai'){
            agent_type = 1;
        }
    
    	neighbors_list_store.push([map_config['all_robots'][um_idx][0], agent_type]);
    
        var div_element = document.createElement("div");
        div_element.setAttribute("class", "wrapper");
        var input_element = document.createElement("input");
        input_element.setAttribute("type", "radio");
        input_element.setAttribute("id", String(map_config['all_robots'][um_idx][0]));
        input_element.setAttribute("name", "neighbors");
        input_element.setAttribute("value", String(map_config['all_robots'][um_idx][0]));
        var label_element = document.createElement("label");
        label_element.setAttribute("for", String(map_config['all_robots'][um_idx][0]));
        label_element.setAttribute("id", String(map_config['all_robots'][um_idx][0]) + '_entry');
        label_element.style.color = "black";
        
        const tbl = document.createElement('table');
        const tr1 = tbl.insertRow();
	const td1 = tr1.insertCell();
	td1.appendChild(document.createTextNode("Agent " + String(map_config['all_robots'][um_idx][0]) + " (type: " + map_config['all_robots'][um_idx][1] + ")"));
	
	
        label_element.appendChild(tbl);
        
        
        div_element.appendChild(input_element);	
        div_element.appendChild(label_element);
        neighbor_info_div.appendChild(div_element);


        
    }
    
    
});



//getStream()
//  .then(getDevices)
//  .then(gotDevices);

function getDevices() {
  return navigator.mediaDevices.enumerateDevices();
}

//Get simulated webcams and their video streams

async function gotDevices(deviceInfos) {
    var v = 0, real_video = 0;
    const currentDiv = document.getElementById("videos_div");

    /*
    var first_video = 0;
    const searchParams = new URLSearchParams(window.location.search);
    
    if (searchParams.has('v_idx')){
        first_video = parseInt(searchParams.get('v_idx'));
    }
    */

    for (let i = 0; i < deviceInfos.length; i++) {
    
    		if(!(deviceInfos[i]["kind"] === "videoinput")){
    			continue;
    		} else {
    			real_video += 1
    			if (real_video-1 < first_video_idx){
    				continue;
			}
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
		//document.body.insertBefore(videoElement, currentDiv);
		currentDiv.appendChild(videoElement);
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

function manualResetTutorial(){
  socket.emit("reset_tutorial");
}

function removeTags(str) {
    if ((str===null) || (str===''))
        return false;
    else
        str = str.toString();
          
    // Regular expression to identify HTML tags in
    // the input string. Replacing the identified
    // HTML tag with a null string.
    return str.replace( /(<([^>]+)>)/ig, '');
}

function newMessage(message, id){
	const chat = document.getElementById("chat");
	var p_element = document.createElement("p");
	p_element.innerHTML = "<strong>"+ removeTags(String(id)) + "</strong>: " + message;

	chat.appendChild(p_element);
	
	p_element.scrollIntoView({block: "nearest", inline: "nearest"});
}



function sendCommand() {

	final_string = document.getElementById('command_text').value;
	document.getElementById('command_text').value = "";
	
	if(final_string){
	
	
	    newMessage(final_string, "ADMIN");

	    var agents = document.getElementsByName('neighbors');
	    var command_string;
	    for(i = 0; i < agents.length; i++) {
		    if(agents[i].checked){
			    command_string = agents[i].value;
			    break;
		    }
	    }
	    
	    neighbors_dict = {}
	    
	    if(command_string === "All"){
	    
	    	    /*
		    for(nl_idx in neighbors_list_store){

			    var human_or_robot = 0;
			    if(! neighbors_list_store[nl_idx][1]){
			        human_or_robot = "human";
			    } else{
			        human_or_robot = "ai";
			    }
			    neighbors_dict[neighbors_list_store[nl_idx][0]] = human_or_robot;
		        
		    }
		    */
	    } else{
	    
	        var robot_id = command_string.split(" ")[0];

	        
	        for(nl_idx in neighbors_list_store){
	            if(neighbors_list_store[nl_idx][0] === robot_id){
	                	    
            	    if(! neighbors_list_store[nl_idx][1]){
	                    human_or_robot = "human";
	                } else{
	                    human_or_robot = "ai";
	                }
	                
	                break;
	            }
	        }
	        
	        

	        
	        neighbors_dict[robot_id] = human_or_robot;
	    }
	    
	    
	    socket.emit("message", final_string, 0, neighbors_dict);
	}
}


socket.on("message", (message, timestamp, id) => {
	console.log("Received message");
	newMessage(message, id);

});


let chat_input_text = document.getElementById("command_text");

chat_input_text.addEventListener("keydown", (event) => {

	if( event.key.includes("Enter")){
		sendCommand();
	}

});

socket.emit("broadcaster_load");
