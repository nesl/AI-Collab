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

  //client_id = new_client_id;
  peerConnection = new RTCPeerConnection(config);
  peerConnection
    .setRemoteDescription(description)
    .then(() => peerConnection.createAnswer())
    .then(sdp => peerConnection.setLocalDescription(sdp))
    .then(() => {
      socket.emit("answer", id, peerConnection.localDescription);
    });
  peerConnection.ontrack = event => {
    if(num_video == 0){
    	video[num_video].srcObject = event.streams[0];
    	num_video += 1;
    }

  };
  peerConnection.onicecandidate = event => {
    if (event.candidate) {
      console.log(event.candidate)
      socket.emit("candidate", id, event.candidate);
    }
  };
});


socket.on("candidate", (id, candidate) => {
  console.log(candidate)
  peerConnection
    .addIceCandidate(new RTCIceCandidate(candidate))
    .catch(e => console.error(e));

  socket.emit("get_id");
});



socket.on("connect", () => {
  socket.emit("watcher", client_number);
  
});

socket.on("broadcaster", () => {
  socket.emit("watcher", client_number);
});


var own_neighbors_info_entry = [];
var object_list_store = [];
var object_html_store = {};
var neighbors_list_store = [];





let chat_input_text = document.getElementById("command_text");

chat_input_text.addEventListener("keydown", (event) => {

	if( event.key.includes("Enter")){
		sendCommand();
	}

});


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

function reset(){

    object_list_store = [];
    neighbors_list_store = [];
    object_html_store = {};
    own_neighbors_info_entry = [client_id, 0, 0, 0, -1];
    
    var neighbor_info_div = document.getElementById("collapsible_nearby_team_members");
    
    neighbor_info_div.innerHTML = '';

    var collapsible_tag = document.getElementById("collapsible_nearby_team_members_tag");
    
    
    var object_info_div = document.getElementById("collapsible_object_information");
    
    object_info_div.innerHTML = '';
    
    var text_search = document.createElement("input");
    text_search.setAttribute("type", "text");
    text_search.setAttribute("placeholder", "Search...");
    text_search.addEventListener("change",function (event) {
		
		   
    	var object_info_div = document.getElementById("collapsible_object_information");

		const original_length = object_info_div.childNodes.length;

		for (let i = original_length-1; i > 0; i--) { 
			object_info_div.removeChild(object_info_div.childNodes[i]);
		}
		
		Object.keys(object_html_store).forEach(function(object_key) {
		
			const pattern = new RegExp('^' + event.target.value);
			if(pattern.test(object_key)){
		    	object_info_div.appendChild(object_html_store[object_key]);
		    }
		});
		
	});
	
    object_info_div.appendChild(text_search);
    
    
    var collapsible_tag_objects = document.getElementById("collapsible_object_tag");
    
    collapsible_tag_objects.innerHTML = "Object Information (0)";
    
    var chat = document.getElementById("chat");
    
    chat.innerHTML = '';
    
    
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
    
        neighbors_list_store.push([map_config['all_robots'][um_idx][0], agent_type ,0,0,-1, false]);
        
    
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
        label_element.appendChild(document.createTextNode(String(map_config['all_robots'][um_idx][0]) + " (type: " + map_config['all_robots'][um_idx][1] + ") " + "(distance: -1)"));
        
        
        div_element.appendChild(input_element);	
        div_element.appendChild(label_element);
        neighbor_info_div.appendChild(div_element);


        
    }
    
    collapsible_tag.innerHTML = "Team Members (" + removeTags(String(neighbors_list_store.length)) + ")";


}

socket.on("agent_reset", () => {
    reset();
});


var map_config = {};

socket.on("watcher", (robot_id_r, config_options) => {

    client_id = robot_id_r;
    map_config = config_options;
    

    for(ob_idx = 0; ob_idx < map_config['all_robots'].length; ob_idx++){ //Remove self
        if(map_config['all_robots'][ob_idx][0] === client_id){
            map_config['all_robots'].splice(ob_idx,1);
            break;
        }
    }
    
    var communication_distance_limit = document.getElementById("comms_text");
    
    communication_distance_limit.innerHTML = "Maximum distance for communication: " + removeTags(String(map_config['communication_distance_limit'])) + ' m';
    
    var strength_distance_limit = document.getElementById("distance_text");
    
    strength_distance_limit.innerHTML = "Maximum distance for carrying objects: " + removeTags(String(map_config['strength_distance_limit'])) + ' m';
    
    
    reset();
});

simulator_timer  = -1;

socket.on("human_output", (location, item_info, neighbors_info, timer) => {


    simulator_timer = timer;

    Object.keys(item_info).forEach(function(object_key) {
        update_objects_info(object_key, item_info[object_key]['time'], item_info[object_key]['sensor'], [item_info[object_key]['location'][0],item_info[object_key]['location'][2]], item_info[object_key]['weight'], false);
    });
    
    Object.keys(neighbors_info).forEach(function(neighbor_key) {
        update_neighbors_info(neighbor_key, timer, neighbors_info[neighbor_key][1], false);
    });
    
    nearby_keys = Object.keys(neighbors_info);
    
    for(ob_idx = 0; ob_idx < neighbors_list_store.length; ob_idx++){
    
        const text_node = document.getElementById(neighbors_list_store[ob_idx][0] + '_entry');
        
        const n_idx = Object.keys(neighbors_info).indexOf(neighbors_list_store[ob_idx][0]);
        
        const distance_string = "distance: ";
        const distance_idx = text_node.innerHTML.indexOf(distance_string);
        
        //if(Object.keys(neighbors_info).includes(neighbors_list_store[ob_idx][0])){
        //if(n_idx > -1){
        if(Object.keys(neighbors_info).includes(neighbors_list_store[ob_idx][0])){
 
            /*
            const x = Math.pow(neighbors_info[neighbors_list_store[ob_idx][0]][1][0] - location[0],2);
            const y = Math.pow(neighbors_info[neighbors_list_store[ob_idx][0]][1][1] - location[2],2);
            
            var distance = Math.sqrt(x+y);
            */
            text_node.style.color = "red";
            text_node.innerHTML = text_node.innerHTML.slice(0,distance_idx+distance_string.length) + removeTags(String(neighbors_info[neighbors_list_store[ob_idx][0]][2].toFixed(1))) + " m)";
            neighbors_list_store[ob_idx][5] = true;
             
        } else {
            text_node.style.color = "black";
            text_node.innerHTML = text_node.innerHTML.slice(0,distance_idx+distance_string.length) + "-1 m)";
            neighbors_list_store[ob_idx][5] = false;
        }
    }
    
    own_neighbors_info_entry[2] = location[0];
    own_neighbors_info_entry[3] = location[2];
    own_neighbors_info_entry[4] = timer;
});

window.onunload = window.onbeforeunload = () => {
  socket.close();
  peerConnection.close();
};

var play_area = document.getElementById("play_area");

//Only in the play area do we catch keyboard events
play_area.onkeydown = function(evt) {
    
    
    var kkey;
    
    if(evt.key.includes("Left")){
        kkey = "LeftArrow";
    } else if(evt.key.includes("Right")){
        kkey = "RightArrow";
    } else if(evt.key.includes("Up")){
        kkey = "UpArrow";
    } else if(evt.key.includes("Down")){
        kkey = "DownArrow";
    } else if(/^\d$/.test(evt.key)){
        kkey = "Alpha" + evt.key;
    } else{
        kkey = evt.key.toUpperCase();
    }
    
    /*
    if(kkey == "W"){
    	let chat_input_text = document.getElementById("command_text");
    	chat_input_text.focus();
    }
    */
    console.log(evt.key, kkey);
    socket.emit("key", kkey, simulator_timer);
};



function togglePopup(){
	document.getElementById("popup-1").classList.toggle("active");
}


function convert_to_real_coordinates(position){

    min_pos = map_config['edge_coordinate']
    multiple = map_config['cell_size']
    pos_new = [position[0]*multiple - Math.abs(min_pos), position[1]*multiple - Math.abs(min_pos)]

    
    return pos_new
}


function update_danger_estimate(label_string, danger_data){

    var confidence_max = 0;
    var sensor_user;
    for (const s in danger_data){ //gets value with max confidence
	    if(danger_data[s].confidence > confidence_max){
		    confidence_max = danger_data[s].confidence;
		    sensor_user = danger_data[s];
	    }
    }
    //sensor_user = sensor_key_list[0];
    var danger = sensor_user.value;
    var color;
    if(danger == 1){
	    color = 'blue';
    }
    else{
	    color = 'red';
    }

    label_string +=  '<p style="color:' + color + '">' + removeTags(String((sensor_user.confidence*100).toFixed(2)))+"% </p>" //" <div style=\"color:" + color + "\">&#9632;</div> "+ String((sensor_user.confidence*100).toFixed(2))+"%";
    
    return label_string;

}

function update_objects_info(object_key, timer, danger_data, position, weight, convert_coordinates){

	var known_object = false;
	
 	var object_info_div = document.getElementById("collapsible_object_information");

    var collapsible_tag = document.getElementById("collapsible_object_tag");
	
	if(convert_coordinates){
		position = convert_to_real_coordinates(position);
	}
	
	for(ob_idx = 0; ob_idx < object_list_store.length; ob_idx++){
 		if(object_key == object_list_store[ob_idx][0]){ 
 			if(Object.keys(danger_data).length > 0){
 			    object_list_store[ob_idx][2] = Object.assign({}, danger_data, object_list_store[ob_idx][2]); //TODO update estimation in ui
 			    var label_string = removeTags(String(object_list_store[ob_idx][0]) + " (weight: " + String(object_list_store[ob_idx][1]) + ")");
 			    label_string = update_danger_estimate(label_string, object_list_store[ob_idx][2]);
 			    //label_element = document.getElementById("label_" + String(object_list_store[ob_idx][0]));
 			    label_element = object_html_store[object_key].children[1]
 			    label_element.innerHTML = label_string;
 			}
 			
 			if(object_list_store[ob_idx][5]	< timer){
 			    object_list_store[ob_idx][3] = position[0]
 			    object_list_store[ob_idx][4] = position[1]
 			    object_list_store[ob_idx][5] = timer
 			}
 			
 			known_object = true;
 			
 			break;
 			
		}
		
 	}
 	
 	if(! known_object){
	    object_list_store.push([object_key,weight,danger_data,position[0],position[1],timer]);
	    
	    var div_element = document.createElement("div");
	    div_element.setAttribute("class", "wrapper");
	    var input_element = document.createElement("input");
	    input_element.setAttribute("type", "radio");
	    input_element.setAttribute("id", String(object_key));
	    input_element.setAttribute("name", "objects");
	    input_element.setAttribute("value", String(object_key));
	    var label_element = document.createElement("label");
	    label_element.setAttribute("for", String(object_key));
	    label_element.setAttribute("id", "label_" + String(object_key));
	    
	    var label_string = removeTags(String(object_key) + " (weight: " + String(weight) + ")");
	    
	    if(Object.keys(danger_data).length > 0){ 
	    
	        label_string = update_danger_estimate(label_string, danger_data);
	    }
	    
	    label_element.innerHTML = label_string;
	    
	    
	    div_element.appendChild(input_element);	
	    div_element.appendChild(label_element);
	    
	    object_html_store[object_key] = div_element;
	    
	    


	}
 	

  
    collapsible_tag.innerHTML = "Object Information (" + removeTags(String(object_list_store.length)) + ")";
    
    fill_info();
}


function fill_info(){

    var object_info_div = document.getElementById("collapsible_object_information");

	const original_length = object_info_div.childNodes.length;

	for (let i = original_length-1; i > 0; i--) { 
		object_info_div.removeChild(object_info_div.childNodes[i]);
	}
    
    Object.keys(object_html_store).forEach(function(object_key) {
        object_info_div.appendChild(object_html_store[object_key]);
    });

}

//Update object info
socket.on("objects_update", (object_list, source_id) => {



  if(type_of_agent(source_id) == 'human'){
        coords_conversion = true;
  }
  else{
        coords_conversion = false;
  }

  for(ob_idx = 0; ob_idx < object_list.length; ob_idx++){
      update_objects_info(object_list[ob_idx][0], object_list[ob_idx][5], object_list[ob_idx][2], [object_list[ob_idx][3],object_list[ob_idx][4]], object_list[ob_idx][1], coords_conversion);
  }

});



//Update neighbors info
socket.on("neighbors_update", (neighbors_list, source_id) => {


  
  if(type_of_agent(source_id) == 'human'){
        coords_conversion = true;
  }
  else{
        coords_conversion = false;
  }
  
  for(ob_idx = 0; ob_idx < neighbors_list.length; ob_idx++){
    update_neighbors_info(neighbors_list[ob_idx][0], neighbors_list[ob_idx][4], [neighbors_list[ob_idx][2],neighbors_list[ob_idx][3]], coords_conversion);
  }
  

});

function update_neighbors_info(agent_key, timer, position, convert_coordinates){


    if(convert_coordinates){
        position = convert_to_real_coordinates(position)
    }
            
    for(ob_idx = 0; ob_idx < neighbors_list_store.length; ob_idx++){
                    
        if(neighbors_list_store[ob_idx][0] == agent_key && (neighbors_list_store[ob_idx][4] == -1 || neighbors_list_store[ob_idx][4] < timer)){
            neighbors_list_store[ob_idx][2] = position[0]
            neighbors_list_store[ob_idx][3] = position[1]
            neighbors_list_store[ob_idx][4] = timer
            
            break
        }
    }

    
    /*
    console.log(neighbor_keys);
    Object.keys(neighbors_list_store).forEach(function(key) {

    if(neighbor_keys.indexOf(key) == -1){
        const div_node = document.getElementById(String(key));
        div_node.parentElement.remove();
        delete neighbors_list_store[key];
    }
    });

    */

}


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
	p_element.innerHTML = "<strong>"+ removeTags(String(id)) + "</strong>: " + message;

	chat.appendChild(p_element);
	
	p_element.scrollIntoView({block: "nearest", inline: "nearest"});
}


function type_of_agent(robot_id){

    var human_or_robot;
    
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
    
    return human_or_robot;
}


/*
function addPrompt(info_type){

	const chat = document.getElementById("chat");
	var button_element_yes = document.createElement("button");
	var button_element_no = document.createElement("button");
	button_element_yes.innerHTML = "Yes";
	button_element_no.innerHTML = "No";
	button_element_yes.onclick = ;
	button_element_no.onclick = ;

}
*/

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
		selector = document.getElementById('select_channel');
		option_element = document.createElement("option");
		option_element.setAttribute("value",help_requests[final_string.substring(12)]);
		option_element.appendChild(document.createTextNode(help_requests[final_string.substring(12)]));
		selector.appendChild(option_element);
		
		//socket.emit("set_goal", help_requests[final_string.substring(12)]);
	}
	
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
		for(nl_idx in neighbors_list_store){
		    if(neighbors_list_store[nl_idx][5]){ //If it's closeby
		        var human_or_robot = 0;
		        if(! neighbors_list_store[nl_idx][1]){
		            human_or_robot = "human";
		        } else{
		            human_or_robot = "ai";
		        }
		        neighbors_dict[neighbors_list_store[nl_idx][0]] = human_or_robot;
		    }
		}
	} else{
	
	    var robot_id = command_string.split(" ")[0];

	    
	    for(nl_idx in neighbors_list_store){
	        if(neighbors_list_store[nl_idx][0] === robot_id && neighbors_list_store[nl_idx][5]){
	            	    
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
	
	
	socket.emit("message", final_string, simulator_timer, neighbors_dict);
}



socket.on("message", (message, timestamp, id) => {
	console.log("Received message");
	newMessage(message, id);

	
	if(message.includes("I need help with ")){
		help_requests[id] = message.substring(25);
	}
	else if(message.includes("Ask for object information to ")){
	    socket.emit("objects_update", String(id), object_list_store);
	}
	else if(message.includes("Ask for agent information to ")){
		socket.emit("neighbors_update", String(id), get_corrected_neighbors_info(String(id)));
		
	}
});

function get_corrected_neighbors_info(target_id){

    var corrected_neighbors_info = Array.from(neighbors_list_store);
    
    for(ni_idx = 0; ni_idx < corrected_neighbors_info.length; ni_idx++) {
        if(corrected_neighbors_info[ni_idx][0] == target_id){
            corrected_neighbors_info.splice(ni_idx, 1);
            break;
        }
    }
    
    corrected_neighbors_info.push(own_neighbors_info_entry);
    
    return corrected_neighbors_info;
}


togglePopup();

