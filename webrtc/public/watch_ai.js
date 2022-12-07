let peerConnection;
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
const video = document.querySelectorAll("video");

var client_id;

var num_video = 0;

const queryString = window.location.search;
const urlParams = new URLSearchParams(queryString);
const client_number = urlParams.get('client');


//We set here the video streams
socket.on("offer", (id, description, new_client_id) => {

  client_id = new_client_id;
  peerConnection = new RTCPeerConnection(config);
  peerConnection
    .setRemoteDescription(description)
    .then(() => peerConnection.createAnswer())
    .then(sdp => peerConnection.setLocalDescription(sdp))
    .then(() => {
      socket.emit("answer", id, peerConnection.localDescription);
    });
  peerConnection.ontrack = event => {

    video[num_video].srcObject = event.streams[0];

    num_video += 1;

    if(num_video == 1){
        pc = createPeerConnection();
        var constraints = {
            audio: false,
            video: true
        };

        pc.addTrack(event.track);
       
        negotiate();
            
       

    }

  };
  peerConnection.onicecandidate = event => {
    if (event.candidate) {
      console.log(event.candidate)
      socket.emit("candidate", id, event.candidate);
    }
  };

  peerConnection.oniceconnectionstatechange = () => {
    if (peerConnection.iceConnectionState === "completed"){
        start_streaming();
    }    
  };
});


socket.on("candidate", (id, candidate) => {

  peerConnection
    .addIceCandidate(new RTCIceCandidate(candidate))
    .catch(e => console.error(e));

  socket.emit("get_id");
});

socket.on("get_id", id => {
	console.log("got id " + String(id));
	client_id = id;
});

socket.on("connect", () => {
  socket.emit("watcher", client_number);
});

socket.on("broadcaster", () => {
  socket.emit("watcher", client_number);
});

window.onunload = window.onbeforeunload = () => {
  socket.close();
  peerConnection.close();
};

var play_area = document.getElementById("play_area");

//Only in the play area do we catch keyboard events
play_area.onkeydown = function(evt) {
    console.log(evt.key);
    socket.emit("key", evt.key);
};

var object_list_store = {};

//Update object info
socket.on("objects_update", object_list => {


  var object_info_div = document.getElementById("collapsible_object_information");

  var collapsible_tag = document.getElementById("collapsible_object_tag");
  
  collapsible_tag.innerHTML = "Object Information (" + String(Object.keys(object_list).length) + ")";
  Object.keys(object_list).forEach(function(key) {

	if (! object_list_store.hasOwnProperty(String(key))){
	    	var div_element = document.createElement("div");
		div_element.setAttribute("class", "wrapper");
		var input_element = document.createElement("input");
		input_element.setAttribute("type", "radio");
		input_element.setAttribute("id", String(key));
		input_element.setAttribute("name", "objects");
		input_element.setAttribute("value", String(key));
		var label_element = document.createElement("label");
		label_element.setAttribute("for", String(key));
		label_element.setAttribute("id", "label_" + String(key));
		var weight = object_list[key].weight
		console.log(object_list)
		if(object_list[key].hasOwnProperty("sensor")){
			sensor_key_list = Object.keys(object_list[key].sensor);

			var confidence_max = 0;
			var sensor_user;
			for(s in sensor_key_list){
				if(object_list[key].sensor[sensor_key_list[s]].confidence > confidence_max){
					confidence_max = object_list[key].sensor[sensor_key_list[s]].confidence;
					sensor_user = sensor_key_list[s];
				}
			}
			//sensor_user = sensor_key_list[0];
			var danger = object_list[key].sensor[sensor_user].value;
			var color;
			if(danger == 1){
				color = 'green';
			}
			else{
				color = 'red';
			}

			label_element.innerHTML = String(key) + " (weight: " + String(weight) + ") <div style=\"color:" + color + "\">&#9632;</div> "+ String(object_list[key].sensor[sensor_user].confidence*100)+"%";
		} else {
			label_element.innerHTML = String(key) + " (weight: " + String(weight) + ")";
		}
		
		//label_element.appendChild(document.createTextNode(String(key) + " (weight: " + String(weight) + ")"));
		
		div_element.appendChild(input_element);	
		div_element.appendChild(label_element);
		object_info_div.appendChild(div_element);

		object_list_store[key] = object_list[key]

		
	} else { //TODO update estimation
		
		if(object_list_store[key].hasOwnProperty("sensor")){
			var object_list_store_keys = Object.keys(object_list_store[key].sensor)
			Object.keys(object_list[key].sensor).forEach(function(key2) {
				if(object_list_store_keys.indexOf(key2) == -1){
					object_list_store[key].sensor[key2] = object_list[key].sensor[key2]

				}
			});
		}
		else {
			object_list_store[key]["sensor"] = object_list[key].sensor
		}
	}

	
  });

});

var neighbors_list_store = {};

//Update neighbors info
socket.on("neighbors_update", neighbors_list => {


  var neighbor_info_div = document.getElementById("collapsible_nearby_team_members");
  
  var collapsible_tag = document.getElementById("collapsible_nearby_team_members_tag");
  

  console.log(neighbors_list);
  var neighbor_keys = Object.keys(neighbors_list);

  neighbor_keys.forEach(function(key) {

	if (! neighbors_list_store.hasOwnProperty(String(key))){
		var div_element = document.createElement("div");
		div_element.setAttribute("class", "wrapper");
		var input_element = document.createElement("input");
		input_element.setAttribute("type", "radio");
		input_element.setAttribute("id", String(key));
		input_element.setAttribute("name", "neighbors");
		input_element.setAttribute("value", String(key));
		var label_element = document.createElement("label");
		label_element.setAttribute("for", String(key));
		var agent_type = neighbors_list[key]
		label_element.appendChild(document.createTextNode(String(key) + " (type: " + agent_type + ")"));
		
		div_element.appendChild(input_element);	
		div_element.appendChild(label_element);
		neighbor_info_div.appendChild(div_element);

		neighbors_list_store[key] = neighbors_list[key]
	}

	
  });

  console.log(neighbor_keys);
  Object.keys(neighbors_list_store).forEach(function(key) {

	if(neighbor_keys.indexOf(key) == -1){
		const div_node = document.getElementById(String(key));
		div_node.parentElement.remove();
		delete neighbors_list_store[key];
	}
  });
  collapsible_tag.innerHTML = "Nearby Team Members (" + String(Object.keys(neighbors_list_store).length) + ")";
  

});


function findCheckedRadio(radio_elements,final_string,pattern){

	var command_string;
	for(i = 0; i < radio_elements.length; i++) {
		if(radio_elements[i].checked){
			command_string = radio_elements[i].value;
			break;
		}
	}
	if(command_string == null){
		return "";
	}
	var result = final_string.replace(pattern,command_string);

	return result;

}

function newMessage(message, id){
	const chat = document.getElementById("chat");
	var p_element = document.createElement("p");
	p_element.innerHTML = "<strong>"+ String(id) + "</strong>: " + message;

	chat.appendChild(p_element);
}

var help_requests = {};

//Set Command based on templates
function setCommand (){

	var final_string = "";

	var ele = document.getElementsByName('command');
              
	for(i = 0; i < ele.length; i++) {
		if(ele[i].checked){
			final_string = ele[i].value;
			break;
		}
	}
	if(final_string.includes('[agent]')){
		var agents = document.getElementsByName('neighbors');
		final_string = findCheckedRadio(agents,final_string,'[agent]');
	
	}
	else if(final_string.includes('[object]')){
		var objects = document.getElementsByName('objects');
		final_string = findCheckedRadio(objects,final_string,'[object]');
	}
	
	if(final_string.length == 0){
		return;
	}

	
	document.getElementById('command_text').value = final_string;

	
}

function sendCommand() {

	final_string = document.getElementById('command_text').value;
	document.getElementById('command_text').value = "";
	newMessage(final_string, client_id);

	if(final_string.includes("I will help ")){
		console.log(help_requests)
		console.log(help_requests[final_string.substring(12)])
		socket.emit("set_goal", help_requests[final_string.substring(12)]);
	}
	
	socket.emit("message", final_string, neighbors_list_store, client_id);
}

socket.on("message", (message, id) => {
	console.log("Received message");
	newMessage(message, id);
    dc.send(message);
	
	if(message.includes("I need help with ")){
		help_requests[id] = message.substring(25);
	}
});



// peer connection
var pc = null;

// data channel
var dc = null, dcInterval = null;

var num_track = 0;
var moment = 0;

function createPeerConnection() {
    var config = {
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


    pc = new RTCPeerConnection(config);

    // register some listeners to help debugging
    pc.addEventListener('icegatheringstatechange', function() {
       console.log(pc.iceGatheringState);
    }, false);


    pc.addEventListener('iceconnectionstatechange', function() {
        console.log(pc.iceConnectionState);
    }, false);

    pc.addEventListener('signalingstatechange', function() {
        console.log(pc.signalingState);
    }, false);


    return pc;
}

function negotiate() {
    return pc.createOffer().then(function(offer) {

        return pc.setLocalDescription(offer);
    }).then(function() {
        // wait for ICE gathering to complete
        return new Promise(function(resolve) {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(function() {
        var offer = pc.localDescription;
        var codec;
        
        return fetch('https://172.17.15.69:8080/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then(function(response) {

        return response.json();
    }).then(function(answer) {

        return pc.setRemoteDescription(answer);
    }).catch(function(e) {
        alert(e);
    });
}


function create_data_channel() {
    dc = pc.createDataChannel('chat', parameters);
    dc.onclose = function() {
        clearInterval(dcInterval);
        dataChannelLog.textContent += '- close\n';
    };
    /*
    dc.onopen = function() {
        dataChannelLog.textContent += '- open\n';
        dcInterval = setInterval(function() {
            var message = 'ping ' + current_stamp();
            dataChannelLog.textContent += '> ' + message + '\n';
            dc.send(message);
        }, 1000);
    };
    */
    dc.onmessage = function(evt) {

        newMessage(evt.data, client_id);
        
    };
}







