let peerConnection;

var xmlDoc;
var first_state;

//Load xml doc with strings
function loadXMLDoc() {
    let xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function () {
 
        // Request finished and response 
        // is ready and Status is "OK"
        if (this.readyState == 4 && this.status == 200) {
            xmlDoc = this.responseXML;
            if(first_state){
                tutorial_popup(first_state);
            }
        }
    };
 
    // employee.xml is the external xml file
    xmlhttp.open("GET", "values/strings.xml", true);
    xmlhttp.send();
}
 
loadXMLDoc(); 
 
//TODO: popup report clickable, obatining correct object number, announcing reset, save last state tutorial

function resize_alert(){

    var isAtMaxWidth = screen.availWidth - window.innerWidth === 0;

    if(! isAtMaxWidth){
        alert("Make sure this window is maximized before continuing with the session, otherwise you may experience difficulties");
    }
}

//window.addEventListener('resize', resize_alert);
resize_alert();

//RECORDING EVENTS

let events = [];
let key_events = [];


rrwebRecord({
  emit(event, isCheckout) {
    // push event into the events array
    events.push(event);
    
    //if (isCheckout){
    //	save();
    //}
  }
});


// this function will send events to the backend and reset the events array
function save() {
  const body = JSON.stringify({"events": events, "key_events": key_events });
  events = [];
  key_events = [];
  socket.emit("log_user_events", body);
  
}

// save events every 10 seconds
const saving_events = setInterval(save, 5 * 1000);


//WEBRTC CONFIG

const config = {
  iceServers: [
    { 
      urls: ["stun:stun.l.google.com:19302"]
    },
    { 
       "urls": "turn:54.85.22.234:3478?transport=tcp",
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
const video = document.querySelectorAll("video");

var client_id;

var num_video = 0;

const queryString = window.location.search;
const urlParams = new URLSearchParams(queryString);
const client_number = urlParams.get('client').match(/\d+/)[0];

var passcode = urlParams.get('pass');

if(passcode){
    socket.emit("watcher", client_number, passcode);
}

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


socket.on('disconnect', function (reason) {
    console.log('Socket disconnected because of ' + reason);
});


function submitCode(){

 var code = document.getElementById("pass-text").value;

 socket.emit("watcher", client_number, code);

}

socket.on("connect", () => {
  document.getElementById("popup-pass").classList.toggle("active");
});

socket.on("passcode-rejected", () => {
    document.getElementById("pass-result").innerHTML = "Password rejected";
});

socket.on("broadcaster", () => {
  socket.emit("watcher", client_number);
});

socket.on("reset_announcement", () => {
    //alert("Resetting the game! Wait for 1 minute...");
    document.getElementById("popup-reset").classList.add("active");
});

var tutorial_client_side = "wait";
var tutorial_object = "";
var tutorial_mode;


function find_entry(entries, state){
    for (let i = 0; i < entries.length; i++) {
        if(entries[i].getAttribute("name") == state){
            return entries[i];
        }
    }
    
}

var images_entry = {"move":[["<img src='/media/question_mark.png' style='width:10%;height:auto;'/>"], ["<img src='/media/box.png' style='width:20%;height:auto;'/>"], ["<img src='/media/magnebot.png' style='width:20%;height:auto;'/>"], ["<img src='/media/first_person_screen.png' style='width:50%;height:auto;'/>"], ["<img src='/media/my_strength_bar.png' style='width:20%;height:auto;'/>"], ["<img src='/media/timer_display.png' style='width:10%;height:auto;'/>"], ["<img src='/media/my_location.png' style='width:20%;height:auto;'/>","<img src='/media/points_floor.png' style='width:30%;height:auto;'/>"], ["<img src='/media/carried_object.png' style='width:20%;height:auto;'/>"], ["<img src='/media/action_status.png' style='width:20%;height:auto;'/>"], ["<img src='/media/targets_in_goal.png' style='width:15%;height:auto;'/>"], ["<img src='/media/white_dot.png' style='width:5%;height:auto;'/>"], ["<img src='/media/active.png' style='width:20%;height:auto;'/>"], ["<img src='/media/layout_tutorial.png' style='width:20%;height:auto;'/>"], ["<img src='/media/arrow_keys.png' style='width:20%;height:auto;'/>"],["<img src='/media/fire.gif' style='width:10%;height:auto;'/>"]], 
"move_to_object":[["<img src='/media/fall.gif' style='width:20%;height:auto;'/>"], ["<img src='/media/stuck.jpg' style='width:20%;height:auto;'/>"],[]], 
"activate_sensor":[["<img src='/media/keyW.png' style='width:5%;height:auto;'/>","<img src='/media/headup.gif' style='width:30%;height:auto;'/>","<img src='/media/keyS.png' style='width:5%;height:auto;'/>","<img src='/media/headdown.gif' style='width:30%;height:auto;'/>"], ["<img src='/media/keyQ.png' style='width:5%;height:auto;'/>","<img src='/media/sensor.gif' style='width:30%;height:auto;'/>"], []], 
"pickup_object":[["<img src='/media/object_info.png' style='width:20%;height:auto;'/>"], [], ["<img src='/media/keyE.png' style='width:5%;height:auto;'/>","<img src='/media/scan.gif' style='width:30%;height:auto;'/>"], ["<img src='/media/keyA.png' style='width:5%;height:auto;'/>","<img src='/media/pickup.gif' style='width:30%;height:auto;'/>","<img src='/media/keyD.png' style='width:5%;height:auto;'/>"], ["<img src='/media/fail_pick.gif' style='width:30%;height:auto;'/>"], []], 
"move_to_goal":[["<img src='/media/goal.png' style='width:20%;height:auto;'/>"]], 
"drop_object":[["<img src='/media/keyA.png' style='width:5%;height:auto;'/>","<img src='/media/drop.gif' style='width:30%;height:auto;'/>","<img src='/media/keyD.png' style='width:5%;height:auto;'/>"]], 
"move_to_heavy_object":[["<img src='/media/object_goal.png' style='width:20%;height:auto;'/>"],[]], 
"activate_sensor_heavy":[[]], 
"move_to_agent":[[],["<img src='/media/strength2.gif' style='width:30%;height:auto;'/>"], ["<img src='/media/distance.gif' style='width:20%;height:auto;'/>"], []], 
"ask_sensor":[[],[],["<img src='/media/chat.gif' style='width:20%;height:auto;'/>"],[]], 
"end":[["<img src='/media/end_control.png' style='width:20%;height:auto;'/>","<img src='/media/end.gif' style='width:50%;height:auto;'/>"], ["<img src='/media/select.png' style='width:50%;height:auto;'/>"], ["<img src='/media/results.png' style='width:50%;height:auto;'/>"], []], 
"ask_for_sensing":[["<img src='/media/object_chosen.gif' style='width:30%;height:auto;'/>"], ["<img src='/media/object_write.gif' style='width:20%;height:auto;'/>"], []], 
"send_object_info":[], 
"ask_for_help":[], 
"exchange_info":[], 
"session":[]};

var object_num;
var popup_history;
var popup_current_index;
var content_popup_1;

for(let z=0; z < document.getElementById("popup-1").children.length; z++){
    if(document.getElementById("popup-1").children[z].className == "content"){
        content_popup_1 = document.getElementById("popup-1").children[z];
    }
}

function add_dot(reset){
  progress_bar = document.getElementById('tutorial_progress');
  dot = document.createElement('div');
  dot.classList.add('dot');
  dot.style.background = 'gray';
  progress_bar.appendChild(dot);
  
  if(reset){
      for(let y=0; y < popup_current_index; y++){
        if(progress_bar.childNodes[y].style.background != "black"){
            progress_bar.childNodes[y].style.background = "black";
        }
      }
  }
}

function tutorial_popup(state){

  var popup_text = document.getElementById("popup_text");
  
  const entries = xmlDoc.getElementsByTagName("entry");
  const child_entries = find_entry(entries, state).childNodes;
  
  popup_text.innerHTML = "";
  
  var start_of_list = true;
  var list_index = 0;
  var first_title = false;
  
  for (let i = 0; i < child_entries.length; i++) {
  
    while(true){
        match_result = child_entries[i].textContent.match(/\[(\w+)\]/);
        if(! match_result){
            break;
        } else{
            child_entries[i].textContent = child_entries[i].textContent.replace("["+ match_result[1] +"]", String(eval(match_result[1])));
        }
    }
  
    
    switch(child_entries[i].nodeName){
        
        case "title":
        
            if(! start_of_list){
                popup_text.appendChild(ol);
                start_of_list = true;
                
            }      
            
            if(first_title){
                popup_history.push(popup_text.innerHTML);
                
                if(popup_current_index == popup_history.length-1){
                    add_dot(true);
                } else{
                    add_dot(false);
                }
                popup_text.innerHTML = "";
            } else{
                popup_current_index = popup_history.length; 
            }
              
            popup_text.innerHTML += "<h1>" + child_entries[i].textContent + "</h1>";
            first_title = true;
            break;
        
        case "head":
            popup_text.innerHTML += child_entries[i].textContent;
            break;
            
        case "list":
        
            if(start_of_list){
                start_of_list = false;
                var ol = document.createElement('ol');
            }
        
            let li = document.createElement('li');
            li.innerHTML += child_entries[i].textContent;
            
            if(images_entry[state].length){
                
                if(images_entry[state][list_index].length > 0){
                    li.innerHTML += "<span class='br'></span>";
                    for (let j = 0; j < images_entry[state][list_index].length; j++) {
                        li.innerHTML += images_entry[state][list_index][j];
                    }
                }
            }
            
            ol.appendChild(li);
            list_index += 1;
            break;
            
        case "hint":
            const hint_txt = document.getElementById("hint-txt");
            hint_txt.innerHTML = child_entries[i].textContent;
            break;
    }
  }
  
  if(! start_of_list){
    popup_text.appendChild(ol);
  }
  
  
  if(state == "move_to_agent"){
    var object_info_div = document.getElementById("object_entries");
        
    list_child_nodes = object_info_div.childNodes;
        
    for (let i = 0; i < list_child_nodes.length; i++) {
        if(list_child_nodes[i].children[1].children[0].rows[0].cells[0].textContent.match(/weight: ([0-9]+)/)[1] == "2"){
            tutorial_object = list_child_nodes[i].children[1].children[0].rows[0].cells[0].textContent;
            break;
        }
    }
    
    object_num = tutorial_object.match(/Object ([0-9]+)/)[1];
    
  } else if(state == "ask_sensor"){
    tutorial_client_side = "ask_for_sensing";
  } else if(state == "session"){
    popup_text.innerHTML += "<span class='br'></span><img src='/media/layout.png' style='width:30%;height:auto;'/>";
  }
  
  popup_history.push(popup_text.innerHTML);
  
  if(state == "ask_for_help" || state == "ask_for_sensing" || state == "send_object_info"){
    setTimeout(function() { activatePopup("popup-1"); }, 2000);
  
  } else{
    document.getElementById("popup-1").classList.add("active");
  }
  
  if(popup_current_index == popup_history.length-1){
    add_dot(true);
  } else{
    add_dot(false);
  }
  
  
  popup_text.innerHTML = popup_history[popup_current_index];
  content_popup_1.scrollTo({ top: 0, behavior: 'smooth' });
}

function help_toggle(){

    document.getElementById("popup-1").classList.add("active");
    
    progress_bar = document.getElementById('tutorial_progress');
    progress_bar.childNodes[popup_current_index].style.background = "gray";
    content_popup_1.scrollTo({ top: 0, behavior: 'smooth' });
    
}

socket.on("tutorial", (state) => {
  
  tutorial_mode = true;
  if(xmlDoc){
      tutorial_popup(state);
      
  } else{
    first_state = state;
  }
  
});

function backPopup(){

    if(popup_current_index-1 >= 0){
        
        progress_bar = document.getElementById('tutorial_progress').childNodes;
    
        popup_current_index -= 1;
        popup_text.innerHTML = popup_history[popup_current_index];
        progress_bar[popup_current_index].style.background = "gray";
        //content_popup_1.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

function nextPopup(){

    progress_bar = document.getElementById('tutorial_progress').childNodes;
    progress_bar[popup_current_index].style.background = "black";

    if(popup_current_index+1 >= popup_history.length){
        document.getElementById("popup-1").classList.remove("active");
    } else{
        popup_current_index += 1;
        popup_text.innerHTML = popup_history[popup_current_index];
        content_popup_1.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

var own_neighbors_info_entry = [];
var object_list_store = [];
var neighbors_list_store = [];





let chat_input_text = document.getElementById("command_text");

chat_input_text.addEventListener("keydown", (event) => {

	if( event.key.includes("Enter")){
		sendCommand();
	}

});


let pass_input_text = document.getElementById("pass-text");

pass_input_text.addEventListener("keydown", process_input_code);

function process_input_code(event){

	if(event.key.includes("Enter")){
		submitCode();
	}

}


let popup_window = document.getElementById("popup-1");

popup_window.addEventListener("keydown", (event) => {

    if(event.key.includes("Enter")){
		document.getElementById("popup-1").classList.remove("active");
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

//TODO: use the communication distance limit for something
var map_config = {};
var communication_distance_limit, strength_distance_limit, sensing_distance_limit;
var heaviest_objects;
var previous_scenario;

function reset(config_options){


    map_config = config_options;
    tutorial_mode = false;
    
    document.getElementById("agent_name").innerText = "Agent " + client_id;
    for(ob_idx = 0; ob_idx < map_config['all_robots'].length; ob_idx++){ //Remove self
        if(map_config['all_robots'][ob_idx][0] === client_id){
            map_config['all_robots'].splice(ob_idx,1);
            break;
        }
    }
    
    heaviest_objects = [];
    
    communication_distance_limit = removeTags(String(map_config['communication_distance_limit']));
    
    strength_distance_limit = removeTags(String(map_config['strength_distance_limit']));
    
    sensing_distance_limit = removeTags(String(map_config['sensing_distance_limit']));

    object_list_store = [];
    neighbors_list_store = [];
    own_neighbors_info_entry = [client_id, 0, 0, 0, -1];
    
    document.getElementById("popup-report").classList.remove("active");
    document.getElementById("popup-reset").classList.remove("active");
    document.getElementById("popup-stats").classList.remove("active");
    document.getElementById("popup-stats-content").innerHTML = ""; //"<h1>Stats</h1><br>";
    document.getElementById("command_text").disabled = false;
    document.getElementById("send_command_button").disabled = false;
    
    document.getElementById("report-button").disabled = false;
    document.getElementById("report-button").innerText = "Submit";
    document.getElementById('report-list').innerHTML = "";
    document.getElementById("popup-1").classList.add("active");
    
    var popup_text = document.getElementById("popup_text");
    
    //popup_text.innerHTML = "<h1>Succesful reset, you can start playing now!</h1><span class='br'></span><ol><li>Sense objects and bring only the dangerous ones into the goal area.</li><li>When you sense an object you will see whether the object is dangerous or benign according to the color of the object ID, and in your sidebar you will be able to consult the object, as well as the accuracy of the prediction.</li><li>Objects have a weight associated with them, and you will only be able to carry those that match your strength level. Strength level is modified by the amount of robots that are close to you and you may only be able to carry a heavy object when you have a given quantity of robots around you.</li><li>You need to compare estimates of danger level in order to get the correct dangerous objects. You can talk with your fellow robots through the sidebar chat. You can send messages by writing the text in the input field and then pressing Enter</li><li>To search for an object, input the pattern in the left-hand sidebar and then press Enter.</li><li>Maximum distance for carrying objects: <b>" + strength_distance_limit + " m</b></li><li>Maximum distance for communication: <b>" + communication_distance_limit + " m</b></li><li>Maximum distance for sensing: <b>" + sensing_distance_limit + " m</b></li><li><p style='color: blue;'> Benign objects are associated with blue color</p></li><li><p style='color: red;'>Dangerous objects are associated with red color</p></li></ol><span class='br'></span><h1> Controls </h1><span class='br'></span><b>Arrow Keys</b> to advance <br><b>A</b> to grab/drop object with left arm<br><b>D</b> to grab/drop object with right arm <br><b>S</b> to rotate camera downwards <br><b>W</b> to rotate camera upwards <br><b>Q</b> to take sensing action <br><b>E</b> to focus on an object (you need this to then grab an object)<span class='br'></span><h1>Scenario Map</h1><span class='br'></span><img src='/media/layout.png' style='width:30%;height:auto;'/>";
    
    if(! (previous_scenario && previous_scenario == 2 && map_config["scenario"] == 1)){
        popup_history = [];
        document.getElementById('tutorial_progress').innerHTML = "";
        popup_current_index = 0;
    }
    
    if(map_config["scenario"] != 2){
        tutorial_popup("session");
    }
    
    previous_scenario = map_config["scenario"];
    //document.getElementById("reset_button").disabled = false;
    //document.getElementById("reset_button").innerText = "Reset Game";
    
    var neighbor_info_div = document.getElementById("collapsible_nearby_team_members");
    
    neighbor_info_div.innerHTML = '';

    var collapsible_tag = document.getElementById("collapsible_nearby_team_members_tag");
    
    
    var object_info_div = document.getElementById("object_entries");
    
    object_info_div.innerHTML = '';
    
    //var text_search = document.createElement("input");
    //text_search.setAttribute("type", "text");
    //text_search.setAttribute("placeholder", "Search...");
    
    
    var text_search = document.getElementById("search_input");
    text_search.addEventListener("change",function (event) {
		
		   
    	var object_info_div = document.getElementById("object_entries");

        list_child_nodes = object_info_div.childNodes;
        const pattern = new RegExp('^' + event.target.value);
        
        for (let i = 0; i < list_child_nodes.length; i++) {
            if(! pattern.test(list_child_nodes[i].getAttribute("value"))){
                if(list_child_nodes[i].style.display != "none"){
                    list_child_nodes[i].style.display = "none";
                }
            } else{
                if(list_child_nodes[i].style.display == "none"){
                    list_child_nodes[i].style.display = "";
                }
            }
        }
		
		
	});
	
	
    //object_info_div.appendChild(text_search);
    
    
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
        
        const tbl = document.createElement('table');
        const tr1 = tbl.insertRow();
	    const td1 = tr1.insertCell();
	    td1.appendChild(document.createTextNode("Agent " + String(map_config['all_robots'][um_idx][0]) + " (type: " + map_config['all_robots'][um_idx][1] + ")"));
	    
	    const tr2 = tbl.insertRow();
	    const td2 = tr2.insertCell();
	    td2.classList.add("rr-block");
	    td2.appendChild(document.createTextNode("(location: Out of Range)"));
	
        label_element.appendChild(tbl);
        
        label_element.style.color = "red";
        
        div_element.appendChild(input_element);	
        div_element.appendChild(label_element);
        neighbor_info_div.appendChild(div_element);


        
    }
    
    collapsible_tag.innerHTML = "Team Members (" + removeTags(String(neighbors_list_store.length)) + ")";


}

socket.on("agent_reset", (config_options) => {
    reset(config_options);
});

socket.on("sim_crash", () => {
    document.getElementById("popup-warning").classList.add("active");
    reset(map_config);
});


socket.on("stats", (stats_dict, final) => {

    document.getElementById("popup-report").classList.remove("active");
    document.getElementById("popup-stats").classList.add("active");

    
    
    const popup_content = document.getElementById("popup-stats-content");
    
    popup_content.innerHTML = "";
    
    const title = document.createElement('h1');
    
    title.textContent = "Session Finished: ";
    
    if(stats_dict["failed"]){
        title.textContent += "Picked Up/Dropped Heavy Dangerous Object";
    } else {
        title.textContent += "User Ended Control";
    }
    
    popup_content.appendChild(title);
    
        
    const subtitle1 = document.createElement('h1');
    
    subtitle1.textContent = "Individual Statistics";
    
    popup_content.appendChild(subtitle1);
    
    const tbl = document.createElement('table');
    
    const key_to_text = {"distance_traveled": "Distance traveled (m):",
        "grabbed_objects": "Number of grabbed objects:",
        "grab_attempts":"Number of grab attempts:", 
        "dropped_outside_goal": "Number of times forced to drop object:",
        "objects_sensed":"Number of objects sensed:",
        "objects_in_goal":"Number of objects brought to goal area:",
        "dangerous_objects_in_goal":"Truly dangerous objects brought to goal area (list of object IDs):",
        "num_messages_sent":"Number of messages sent:",
        "average_message_length":"Average length of messages sent (characters):",
        "failed": "Picked Up/Dropped Heavy Dangerous Object:",
        "time_with_teammates":"Time passed next to teammates:",
        "end_time":"End time:",
        "sensor_activation":"Number of times sensor was used:",
        "quality_work":"Quality of work:",
        "effort": "Effort:",
        "individual_payment": "Payment ($):",
        "token": "Your token:"};
    
    Object.keys(stats_dict).forEach(function(key) {
        
        if(key == "objects_in_goal" || key == "dropped_outside_goal"){
        
            const tr1 = tbl.insertRow();
            var td1 = tr1.insertCell();
        
            td1.appendChild(document.createTextNode(key_to_text[key]));
            var td1 = tr1.insertCell();
            
            td1.appendChild(document.createTextNode(String(stats_dict[key].length)));
        } else if(key == "dangerous_objects_in_goal") {
        
            const tr1 = tbl.insertRow();
            var td1 = tr1.insertCell();
            
            td1.appendChild(document.createTextNode(key_to_text[key]));
            var td1 = tr1.insertCell();
        
        
            var final_string = "";
        
            /*
            for(ob_idx = 0; ob_idx < stats_dict[key].length; ob_idx++){
            
                if(ob_idx > 0){
                    final_string += ",";
                }
                final_string += String(stats_dict[key][ob_idx]);
            }
            */
            final_string = stats_dict[key].toString();
            td1.appendChild(document.createTextNode(final_string));
        } else if(key == "failed"){
        
            const tr1 = tbl.insertRow();
            var td1 = tr1.insertCell();
        
            td1.appendChild(document.createTextNode(key_to_text[key]));
            var td1 = tr1.insertCell();
            
            td1.appendChild(document.createTextNode(Boolean(stats_dict[key]).toString()));
        } else if(key == "time_with_teammates"){
        
            const tr1 = tbl.insertRow();
            var td1 = tr1.insertCell();
        
            td1.appendChild(document.createTextNode(key_to_text[key]));
            var td1 = tr1.insertCell();
        
            var final_string = "";
            
            Object.keys(stats_dict[key]).forEach(function(key2) {
            
                if(final_string){
                    final_string += ",";
                }
                
                const divmod_results = divmod(stats_dict[key][key2], 60);
                const divmod_results2 = divmod(divmod_results[1],1);
                
                final_string += String(key2) + ' -> ' + pad(String(divmod_results[0]),2) + ":" + pad(String(divmod_results2[0]),2);
            });
            
            //final_string = JSON.stringify(stats_dict[key]);
            td1.appendChild(document.createTextNode(final_string));
            
        } else if(key == "end_time"){
        
            const tr1 = tbl.insertRow();
            var td1 = tr1.insertCell();
        
            td1.appendChild(document.createTextNode(key_to_text[key]));
            var td1 = tr1.insertCell();
        
            var final_string = "";
            
            const divmod_results = divmod(stats_dict[key], 60);
            const divmod_results2 = divmod(divmod_results[1],1);
            
            final_string = pad(String(divmod_results[0]),2) + ":" + pad(String(divmod_results2[0]),2);
            td1.appendChild(document.createTextNode(final_string));
        } else if(key == "distance_traveled" || key == "individual_payment" || key == "effort" || key == "quality_work"){
        
            const tr1 = tbl.insertRow();
            var td1 = tr1.insertCell();
        
            td1.appendChild(document.createTextNode(key_to_text[key]));
            var td1 = tr1.insertCell();
            td1.appendChild(document.createTextNode(String(stats_dict[key].toFixed(2))));
        } else if(key == "token"){
        
            if(stats_dict[key]){
                const tr1 = tbl.insertRow();
                var td1 = tr1.insertCell();
            
                var bold_el = document.createElement('strong');
                bold_el.setAttribute("class", "token");
                bold_el.appendChild(document.createTextNode(key_to_text[key]));
                td1.appendChild(bold_el);
                var td1 = tr1.insertCell();
                var bold_el = document.createElement('strong');
                bold_el.setAttribute("class", "token");
                bold_el.appendChild(document.createTextNode(stats_dict[key]));
                td1.appendChild(bold_el);
            }
        } else if(Object.keys(key_to_text).includes(key)){
        
            const tr1 = tbl.insertRow();
            var td1 = tr1.insertCell();
        
            td1.appendChild(document.createTextNode(key_to_text[key]));
            var td1 = tr1.insertCell();
            td1.appendChild(document.createTextNode(String(stats_dict[key])));
        }
    });
    
    popup_content.appendChild(tbl);
    

    if(final){
    
        const subtitle2 = document.createElement('h1');
        
        
        subtitle2.textContent = "Team Statistics";
        
        popup_content.appendChild(subtitle2);
        
        const tbl_team = document.createElement('table');
        
        const key_to_text_team = {"team_dangerous_objects_in_goal":"Number of dangerous objects brought to goal area: ",
        "team_end_time": "End Time: ",
        "team_failure_reasons": "Final Team Status: ",
        "total_dangerous_objects": "Total number of dangerous objects: ",
        "human_team_effort": "Team effort: ",
        "team_quality_work": "Team quality of work: ",
        "team_speed_work": "Team speed of work: ",
        "team_achievement": "Team achievement: ",
        "team_payment": "Team total payment ($): "
        };
        
        Object.keys(stats_dict).forEach(function(key) {
            
            if(key == "team_dangerous_objects_in_goal"){
            
                const tr1 = tbl_team.insertRow();
                var td1 = tr1.insertCell();
            
                td1.appendChild(document.createTextNode(key_to_text_team[key]));
                var td1 = tr1.insertCell();
                
                td1.appendChild(document.createTextNode(String(stats_dict[key])));
            
            } else if(key == "team_failure_reasons"){
            
                const tr1 = tbl_team.insertRow();
                var td1 = tr1.insertCell();
            
                td1.appendChild(document.createTextNode(key_to_text_team[key]));
                var td1 = tr1.insertCell();
            
                var final_string = "";
                
                Object.keys(stats_dict[key]).forEach(function(key2) {
                
                    if(final_string){
                        final_string += ",";
                    }
                    
                    var motive = "";
                    
                    if(stats_dict[key][key2] == 0){
                        motive = "Control Ended";
                    } else{
                        motive = "Dropped/Picked Up Dangerous Heavy Object";
                    }
                    
                    final_string += String(key2) + ' -> ' + motive;
                });
                
                //final_string = JSON.stringify(stats_dict[key]);
                td1.appendChild(document.createTextNode(final_string));
                
            } else if(key == "team_end_time"){
            
                const tr1 = tbl_team.insertRow();
                var td1 = tr1.insertCell();
            
                td1.appendChild(document.createTextNode(key_to_text_team[key]));
                var td1 = tr1.insertCell();
            
                var final_string = "";
                
                const divmod_results = divmod(stats_dict[key], 60);
                const divmod_results2 = divmod(divmod_results[1],1);
                
                final_string = pad(String(divmod_results[0]),2) + ":" + pad(String(divmod_results2[0]),2);
                td1.appendChild(document.createTextNode(final_string));
            } else if (key == "total_dangerous_objects"){
               
                const tr1 = tbl_team.insertRow();
                var td1 = tr1.insertCell();
            
            	td1.appendChild(document.createTextNode(key_to_text_team[key]));
                var td1 = tr1.insertCell();
                
                td1.appendChild(document.createTextNode(String(stats_dict[key])));
            } else if(key == "human_team_effort" || key == "team_quality_work" || key == "team_speed_work" || key == "team_achievement" || key == "team_payment"){
            
                    const tr1 = tbl_team.insertRow();
                    var td1 = tr1.insertCell();
            
                    console.log(key, stats_dict[key])	    
            
		    td1.appendChild(document.createTextNode(key_to_text_team[key]));
		    var td1 = tr1.insertCell();
		    td1.appendChild(document.createTextNode(String(stats_dict[key].toFixed(2))));
            } else if(Object.keys(key_to_text_team).includes(key)){
            
                const tr1 = tbl_team.insertRow();
                var td1 = tr1.insertCell();
            
                td1.appendChild(document.createTextNode(key_to_text_team[key]));
                var td1 = tr1.insertCell();
                td1.appendChild(document.createTextNode(String(stats_dict[key])));
           }
            
        });
        
        
        popup_content.appendChild(tbl_team);
        /*
        const bt = document.createElement('button');
        bt.setAttribute("id", "reset_button");
        bt.textContent = "Reset Game";
        bt.onclick = resetGame;
        popup_content.appendChild(bt);
        */
    } else{
        const p_waiting = document.createElement('p');
        p_waiting.textContent = "Waiting for other players...";
        popup_content.appendChild(p_waiting);
    }
    

});


socket.on("watcher", (robot_id_r, config_options) => {

    pass_input_text.removeEventListener("keydown", process_input_code);

    document.getElementById("popup-pass").classList.toggle("active");

    client_id = robot_id_r;
    
    
    reset(config_options);
});

simulator_timer  = -1;

socket.on("human_output", (location, item_info, neighbors_info, timer, disable) => {


    simulator_timer = timer;


    if(disable){
        document.getElementById("command_text").disabled = true;
        document.getElementById("send_command_button").disabled = true;
    }

    
    /*
    if(Object.keys(item_info).length){
        Object.keys(object_html_store).forEach(function(object_key) {
            object_html_store[object_key].style.borderWidth = "thin" ;
        });
    }
    */
    
    var make_visible = true;
    
    Object.keys(item_info).forEach(function(object_key) {
    	
        thick_element = update_objects_info(object_key, item_info[object_key]['time'], item_info[object_key]['sensor'], [item_info[object_key]['location'][0],item_info[object_key]['location'][2]], item_info[object_key]['weight'], false);
        
        var object_html = document.getElementById("div_" + String(object_key));
        
        if(thick_element && object_html.style.borderWidth != "thick"){
            object_html.style.borderWidth = "thick";
            
            if(make_visible) {
        		make_visible = false;
        		object_html.scrollIntoView({block: "nearest", inline: "nearest"});
    		}
        } else if (! thick_element && object_html.style.borderWidth == "thick"){
            object_html.style.borderWidth = "thin";
        }
        
        
        
    });
    
    
    Object.keys(neighbors_info).forEach(function(neighbor_key) {
        update_neighbors_info(neighbor_key, timer, neighbors_info[neighbor_key][1], false);
        
    });
    
    nearby_keys = Object.keys(neighbors_info);
    
    for(ob_idx = 0; ob_idx < neighbors_list_store.length; ob_idx++){
    
        const text_node = document.getElementById(neighbors_list_store[ob_idx][0] + '_entry');
        
        
        const n_idx = Object.keys(neighbors_info).indexOf(neighbors_list_store[ob_idx][0]);
        
        const distance_string = "location: ";
        const distance_idx = text_node.children[0].rows[1].cells[0].textContent.indexOf(distance_string);
        
        //if(Object.keys(neighbors_info).includes(neighbors_list_store[ob_idx][0])){
        //if(n_idx > -1){
        if(Object.keys(neighbors_info).includes(neighbors_list_store[ob_idx][0])){
 
            
            const x = Math.pow(neighbors_info[neighbors_list_store[ob_idx][0]][1][0] - location[0],2);
            const y = Math.pow(neighbors_info[neighbors_list_store[ob_idx][0]][1][1] - location[2],2);
            
            var distance = Math.sqrt(x+y);
            
            
            const disabled = neighbors_info[neighbors_list_store[ob_idx][0]][3];
            
            if(disabled){
                if(text_node.children[0].rows[1].cells[0].textContent != "Disabled"){
                    text_node.style.color = "red";
                    text_node.children[0].rows[1].cells[0].textContent = "Disabled";
                }
            } else {
                
                if(text_node.style.color != "black"){
                    text_node.style.color = "black";
                }
            
            
            
                const divmod_results = divmod(neighbors_list_store[ob_idx][4], 60);
                const divmod_results2 = divmod(divmod_results[1],1);
                
                text_node.children[0].rows[1].cells[0].textContent = "Last seen in location (" + removeTags(String(neighbors_info[neighbors_list_store[ob_idx][0]][1][0].toFixed(1))) + "," + removeTags(String(neighbors_info[neighbors_list_store[ob_idx][0]][1][1].toFixed(1))) + ") (Distance: "+ String(distance.toFixed(1)) +" m) at time " + removeTags(pad(String(divmod_results[0]),2) + ":" + pad(String(divmod_results2[0]),2));
            }
            
            
            
            
            neighbors_list_store[ob_idx][5] = true;
             
        } else {
        
            if(neighbors_list_store[ob_idx][5]){
            
                if(text_node.style.color != "red"){
		            text_node.style.color = "red";
		        }
		        
		        //text_node.children[0].rows[1].cells[0].textContent = "Last seen: " + text_node.children[0].rows[1].cells[0].textContent;
		        //text_node.children[0].rows[1].cells[0].textContent = "location: " + " Last Seen in" + removeTags(String(neighbors_info[neighbors_list_store[ob_idx][0]][1][0].toFixed(1))) + "," + removeTags(String(neighbors_info[neighbors_list_store[ob_idx][0]][1][1].toFixed(1))) + " at " + removeTags(String(neighbors_info[neighbors_list_store[ob_idx][0]][4].toFixed(1))) +  ")"; //"Out of Range)";
		        neighbors_list_store[ob_idx][5] = false;
            }
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
//play_area.onkeydown = function(evt) {

play_area.addEventListener("keydown", (evt) => {
    
    
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
});

document.body.addEventListener("keydown", (evt) => {
    
    var kkey = evt.key;
    key_events.push({"key":kkey, "time": Date.now()});
    
    
});



play_area.addEventListener('mouseover', function() {
    play_area.focus();
    const ct = document.getElementById("active_text");
    
    if(ct.innerHTML != "Active"){
        ct.innerHTML = "Active";
        ct.style.color = "green";
    }
});

play_area.addEventListener('click', function() {
    play_area.focus();
    const ct = document.getElementById("active_text");
    
    if(ct.innerHTML != "Active"){
        ct.innerHTML = "Active";
        ct.style.color = "green";
    }
});

play_area.addEventListener('focusout', function() {
    const ct = document.getElementById("active_text");
    
    if(ct.innerHTML != "Not active"){
        ct.innerHTML = "Not active";
        ct.style.color = "red";
    }
});





function togglePopup(element){
	document.getElementById(element).classList.toggle("active");
}

function activatePopup(element){
	document.getElementById(element).classList.add("active");
}


function convert_to_real_coordinates(position){

    min_pos = map_config['edge_coordinate']
    multiple = map_config['cell_size']
    pos_new = [position[0]*multiple - Math.abs(min_pos), position[1]*multiple - Math.abs(min_pos)]

    
    return pos_new
}


function update_danger_estimate(danger_data){

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
	    txt_danger = 'benign';
    }
    else{
	    color = 'red';
	    txt_danger = 'dangerous';
    }

    var label_string =  ' <p style="color:' + color + ';margin:0;"> Status Danger: ' + txt_danger + ',  Prob. Correct: ' + removeTags(String((sensor_user.confidence*100).toFixed(1)))+"% </p>"; //" <div style=\"color:" + color + "\">&#9632;</div> "+ String((sensor_user.confidence*100).toFixed(2))+"%";
    
    return label_string;

}

const divmod = (x, y) => [Math.floor(x / y), x % y];

function pad(num,size) {
    var s = "00000" + num;
    return s.substr(s.length-size);
}


function update_objects_info(object_key, timer, danger_data, position, weight, convert_coordinates){

	var known_object = false;
	
    var collapsible_tag = document.getElementById("collapsible_object_tag");
	
	if(convert_coordinates){
		position = convert_to_real_coordinates(position);
	}
	
	if((weight >= map_config['all_robots'].length+2 || (tutorial_mode && weight >= 3)) && ! heaviest_objects.includes(object_key)){
    	heaviest_objects.push(object_key);
	}
	
	var danger_changed = false;
	
	var thick_element = false;
	
	for(ob_idx = 0; ob_idx < object_list_store.length; ob_idx++){
 		if(object_key == object_list_store[ob_idx][0]){ 
 			if(Object.keys(danger_data).length > 0){
 			    object_list_store[ob_idx][2] = Object.assign({}, object_list_store[ob_idx][2], danger_data); //TODO update estimation in ui
 			    danger_changed = true;
 			    /*
 			    const tbl = document.createElement('table');
 			    const tr1 = tbl.insertRow();
 			    const td1 = tr1.insertCell();
 			    td1.innerHTML = removeTags(String(object_list_store[ob_idx][0]) + " (weight: " + String(object_list_store[ob_idx][1]) + ")");
 			    const tr2 = tbl.insertRow();
 			    const td2 = tr2.insertCell();
 			    td2.innerHTML = update_danger_estimate(object_list_store[ob_idx][2]);
 			    //label_element = document.getElementById("label_" + String(object_list_store[ob_idx][0]));
 			    label_element = object_html_store[object_key].children[1]
 			    label_element.appendChild(tbl);
 			    */
 			}
 			
 			
 			if(object_list_store[ob_idx][5]	< timer || danger_changed){
 			
 			    if(object_list_store[ob_idx][5] < timer){
	 			    object_list_store[ob_idx][3] = position[0]
	 			    object_list_store[ob_idx][4] = position[1]
	 			    object_list_store[ob_idx][5] = timer
	 			    
	 			    //object_html_store[object_key].style.color = "red" ;
	 			    thick_element = true;
	 			    //object_html_store[object_key].style.borderWidth = "thick";
	 			    
	 			    
 			    }
 			    
 			    
 			    var label_element = document.getElementById("div_" + String(object_key)).children[1]
 			    
 			    danger_changed = false;
 			    
 			    //label_element.children[0].rows[0].cells[0].textContent = "";
 			    
 			    var in_goal = false;
 			    for(let midx in map_config["goal_radius"]){
                    const x = Math.pow(position[0]-map_config["goal_radius"][midx][1][0],2);
                    const y = Math.pow(position[1]-map_config["goal_radius"][midx][1][1],2);
                
                    var distance = Math.sqrt(x+y);
     			    
     			    if(distance < map_config["goal_radius"][midx][0]){
     			        in_goal = true;
     			        break;
     			    }
 			    }
 			    
 			    
 			    
 			    if(in_goal && label_element.children[0].rows[0].cells[0].textContent[0] != "*"){
 			        label_element.children[0].rows[0].cells[0].textContent = "*" + label_element.children[0].rows[0].cells[0].textContent;
 			    } else if (! in_goal && label_element.children[0].rows[0].cells[0].textContent[0] == "*"){
 			        label_element.children[0].rows[0].cells[0].textContent = label_element.children[0].rows[0].cells[0].textContent.substring(1);
 			    }
 			    //label_element.children[0].rows[0].cells[0].textContent += "Object " + removeTags(String(object_list_store[ob_idx][0]) + " (weight: " + String(object_list_store[ob_idx][1]) + ")");
 			    const divmod_results = divmod(object_list_store[ob_idx][5], 60);
	            const divmod_results2 = divmod(divmod_results[1],1);

 			    label_element.children[0].rows[1].cells[0].textContent = removeTags("Last seen in (" + String(object_list_store[ob_idx][3].toFixed(1)) + "," + String(object_list_store[ob_idx][4].toFixed(1)) + ") at " + removeTags(pad(String(divmod_results[0]),2) + ":" + pad(String(divmod_results2[0]),2)));
 			    
 			    if(Object.keys(object_list_store[ob_idx][2]).length > 0){
 			        var danger_string = update_danger_estimate(object_list_store[ob_idx][2]);
 			        
 			        if(label_element.children[0].rows[2].cells[0].innerHTML != danger_string){
	 			        label_element.children[0].rows[2].cells[0].innerHTML = danger_string;
	 			    }
 			    }
	                    
	                    
	                    
	                    
 			    
 			    //label_element.appendChild(tbl);
 			    
 			    
 			    //var label_string = object_html_store[object_key].children[1].innerHTML;
 			    //label_string += label_string;
 			}
 			
 			known_object = true;
 			
 			break;
 			
		}
		
 	}
 	
 	if(! known_object){
	    object_list_store.push([object_key,weight,danger_data,position[0],position[1],timer]);
	    
	    var div_element = document.createElement("div");
	    div_element.setAttribute("class", "wrapper");
	    div_element.setAttribute("id", "div_" + String(object_key));
	    div_element.setAttribute("value", String(object_key));
	    var input_element = document.createElement("input");
	    input_element.setAttribute("type", "radio");
	    input_element.setAttribute("id", String(object_key));
	    input_element.setAttribute("name", "objects");
	    input_element.setAttribute("value", String(object_key));
	    var label_element = document.createElement("label");
	    label_element.setAttribute("for", String(object_key));
	    label_element.setAttribute("id", "label_" + String(object_key));
	    
	    
	    const tbl = document.createElement('table');
	    const tr1 = tbl.insertRow();
	    const td1 = tr1.insertCell();
        td1.innerHTML = "";  
 			    
	    for(let midx in map_config["goal_radius"]){
            const x = Math.pow(position[0]-map_config["goal_radius"][midx][1][0],2);
            const y = Math.pow(position[1]-map_config["goal_radius"][midx][1][1],2);
                
            var distance = Math.sqrt(x+y);
     			    
     		if(distance < map_config["goal_radius"][midx][0]){
     		    td1.innerHTML = "*";
  			    break;
 		    }
        }
                
	    td1.innerHTML += "Object " + removeTags(String(object_key) + " (weight: " + String(weight) + ")");
	    

	    
	    const divmod_results = divmod(object_list_store[ob_idx][5], 60);
        const divmod_results2 = divmod(divmod_results[1],1);
            
        const tr3 = tbl.insertRow();
	    const td3 = tr3.insertCell();
	    
	    td3.classList.add("rr-block");
	    
	    td3.innerHTML = removeTags("Last seen in (" + String(object_list_store[ob_idx][3].toFixed(1)) + "," + String(object_list_store[ob_idx][4].toFixed(1)) + ") at " + removeTags(pad(String(divmod_results[0]),2) + ":" + pad(String(divmod_results2[0]),2)));
	    
	    const tr2 = tbl.insertRow();
        const td2 = tr2.insertCell();
    	if(Object.keys(danger_data).length > 0){ 
	    	
	        td2.innerHTML = update_danger_estimate(danger_data);
	    }
	    
	    label_element.appendChild(tbl);
	    
	    
	    div_element.appendChild(input_element);	
	    div_element.appendChild(label_element);
	    
	    thick_element = true;
	    //object_html_store[object_key].style.borderWidth = "thick" ;
	    

        collapsible_tag.innerHTML = "Object Information (" + removeTags(String(object_list_store.length)) + ")";
        
        var object_info_div = document.getElementById("object_entries");
        
        list_child_nodes = object_info_div.childNodes;
        
        if(list_child_nodes.length){
            var inserted = false;
            for (let i = 0; i < list_child_nodes.length; i++) {
                if(parseInt(div_element.getAttribute("value")) < parseInt(list_child_nodes[i].getAttribute("value"))){

                    object_info_div.insertBefore(div_element, list_child_nodes[i]);
                    inserted = true;
                    break;
                }
            }
            
            if(! inserted){
                object_info_div.appendChild(div_element);
            }
        } else{
            object_info_div.appendChild(div_element);
        }
	}
 	
    return thick_element;
}

/*
var previous_visible = [];

function fill_info(){

    var object_info_div = document.getElementById("object_entries");

	const original_length = object_info_div.childNodes.length;

	for (let i = original_length-1; i > 0; i--) { 
		object_info_div.removeChild(object_info_div.childNodes[i]);
	}
    
    var make_visible = true;
    
    var current_visible = [];
    
    Object.keys(object_html_store).sort((a,b)=>parseInt(a)-parseInt(b)).forEach(function(object_key) {
    
    
    

        object_info_div.appendChild(object_html_store[object_key]);
        
    	if(object_html_store[object_key].style.borderWidth == 'thick'){ //Put into view the objects just sensed
    	    current_visible.push(object_key);
    	    if(make_visible && (! previous_visible.includes(object_key))) {
        		make_visible = false;
        		object_html_store[object_key].scrollIntoView({block: "nearest", inline: "nearest"});
    		}
    	}
    });
    
    previous_visible = current_visible;

}
*/


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

	var command_string, extra_info;
	for(i = 0; i < radio_elements.length; i++) {
		if(radio_elements[i].checked){
			command_string = radio_elements[i].value;
			if(pattern == 'Object to message'){
			    const child_table = document.getElementById("object_entries").children[i].children[1].children[0];
			    command_string = "";
			    for(j = 0; j < child_table.rows.length; j++) {
				    command_string += child_table.rows[j].cells[0].textContent + " ";
				}
			} else {
				if(i > 0){
					 command_string = "Agent " + command_string + " " + document.getElementById(neighbors_list_store[i-1][0] + '_entry').children[0].rows[1].cells[0].textContent;
				}
			}
			

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


function set_text(text_string){
    document.getElementById('command_text').value = text_string;
}

var help_requests = {};

//Set Command based on templates
function setCommand (num){

	var final_string = "";

	var ele = document.getElementsByName('command');
    final_string = ele[num].innerText;
    
	if(final_string.includes('Agent to message')){
		var agents = document.getElementsByName('neighbors');
		final_string = findCheckedRadio(agents,final_string,'Agent to message');
	
	}
	else if(final_string.includes('Object to message')){
		var objects = document.getElementsByName('objects');
		final_string = findCheckedRadio(objects,final_string,'Object to message');
	}
	
	if(final_string.length == 0){
		return;
	}

	
	document.getElementById('command_text').value = final_string;

	
}



function sendCommand() {

	final_string = document.getElementById('command_text').value;
	document.getElementById('command_text').value = "";
	
	if(final_string){
	
	
	    newMessage(final_string, client_id);
	    
	    var popup_text = document.getElementById("popup_text");

        switch(tutorial_client_side){
        
            case "ask_for_sensing":
            
                if(final_string.includes("Hey, what results did you get for object " + object_num + "?")){
                    tutorial_popup(tutorial_client_side);
                    tutorial_client_side = "send_object_info";
                    
                }
                
                break;
            case "send_object_info":
            
                if(final_string.includes(tutorial_object)){
                    tutorial_popup(tutorial_client_side);
                    tutorial_client_side = "ask_for_help";
                }
                break;
            case "ask_for_help":
                if(final_string.includes("Yes, can you help me carry that object?")){
                    tutorial_popup(tutorial_client_side);
                    tutorial_client_side = "exchange_info";
                }
                
                break;
            case "exchange_info":
                if(final_string.includes("Object " + String(heaviest_objects[0]))){
                    tutorial_popup(tutorial_client_side);
                    tutorial_client_side = "tutorial_end";
                    break;
                }
            default:
                break;
            
        }

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
	    
	    /*
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
	    */
	    
	    
	    socket.emit("message", final_string, simulator_timer, neighbors_dict);
	}
}



socket.on("message", (message, timestamp, id) => {
	console.log("Received message");
	newMessage(message, id);
	
	let object_info = tutorial_object.match(/Object ([0-9]+) \(weight: ([0-9]+)\)/);
	
	if(object_info && (parseInt(object_info[2]) >= map_config['all_robots'].length+2 || (tutorial_mode && parseInt(object_info[2]) >= 3)) && ! heaviest_objects.includes(object_info[1])){
	    heaviest_objects.push(object_info[1]);
	}

	/*
	if(message.includes("I need help with ")){
		help_requests[id] = message.substring(25);
	}
	else if(message.includes("Ask for object information to ")){
	    socket.emit("objects_update", String(id), object_list_store);
	}
	else if(message.includes("Ask for agent information to ")){
		socket.emit("neighbors_update", String(id), get_corrected_neighbors_info(String(id)));
		
	}
	*/
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

function reportObjects(){
    document.getElementById("popup-report").classList.add("active");
    
    const report_list = document.getElementById('report-list');
    
    for(ho_idx = 0; ho_idx < heaviest_objects.length; ho_idx++){
        var div_element = document.createElement("div");
        div_element.setAttribute("class", "wrapper");
        var input_element = document.createElement("input");
        input_element.setAttribute("type", "checkbox");
        input_element.setAttribute("id", "heavy_" + heaviest_objects[ho_idx]);
        input_element.setAttribute("name", "report_objects");
        input_element.setAttribute("value", heaviest_objects[ho_idx]);

        var label_element = document.createElement("label");
        label_element.setAttribute("for", "heavy_" + heaviest_objects[ho_idx]);
        label_element.style.color = "black";
        label_element.appendChild(document.createTextNode("Object " + heaviest_objects[ho_idx]));
        
        
        div_element.appendChild(input_element);	
        div_element.appendChild(label_element);
        report_list.appendChild(div_element);
    	
    }
    
}

function submitReport(){

    var report_objects = document.getElementsByName('report_objects');
    
    var report_array = [];
	for(i = 0; i < report_objects.length; i++) {
		if(report_objects[i].checked){
			report_array.push(report_objects[i].value);
		}
	}
	
	document.getElementById("report-button").disabled = true;
    document.getElementById("report-button").textContent = "Waiting for other players...";
	
	socket.emit("report", report_array)
	
	disableRobot();

    

}


function disableRobot(){
    socket.emit("disable");
}

function resetGame(){
    document.getElementById("reset_button").disabled = true;
    document.getElementById("reset_button").textContent = "Waiting for other players...";
    socket.emit("reset");
}

socket.on("ping", () => {
    socket.emit("pong", Date.now());

});



