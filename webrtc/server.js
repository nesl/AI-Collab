const express = require("express");
const app = express();

const commander = require('commander');

//TODO enforce communication radius

commander
  .version('1.0.0', '-v, --version')
  .usage('[OPTIONS]...')
  .option('--address <value>', 'IP address', '172.17.15.69')
  .option('--port <number>', 'Port', 4000)
  .option('--log', 'Log everything')
  .option('--message-loop', 'Send back messages sent')
  .option('--password <value>', 'Specify passwords separated by comma')
  .option('--logdir <value>', 'Specify alternate directory to write logs')
  .option('--replay <value>', 'Replay session using log file')
  .option('--wait', 'Create waiting room')
  .option('--cookies', 'Use cookies')
  .parse(process.argv);

const command_line_options = commander.opts();

let broadcaster;
let simulator;
var map_config;
const port = parseInt(command_line_options.port);
const host = command_line_options.address;//'172.17.15.69'; //'localhost';

const https = require("https");
const yaml = require('js-yaml');
const fs = require("fs");

const replay_file = command_line_options.replay;

var dir;

if(! command_line_options.logdir){
    dir = './log/';
}
else{
    dir = command_line_options.logdir + '/';
}



if (!fs.existsSync(dir)){
    fs.mkdirSync(dir);
}

var today = new Date();
var date = today.getFullYear()+'_'+(today.getMonth()+1)+'_'+today.getDate();
var time = today.getHours() + "_" + today.getMinutes() + "_" + today.getSeconds();
var dateTime; // = date+'_'+time;

// Creating object of key and certificate
// for SSL
const options = {
  key: fs.readFileSync("server.key"),
  cert: fs.readFileSync("server.cert"),
};
  
// Creating https server by passing
// options and app object
const server = https.createServer(options, app)
.listen(port,host, function (req, res) {
  console.log(`Server is running on port ${port}`);
});
//const server = https.createServer(app);

const io = require("socket.io")(server);

var yaml_doc = yaml.load(fs.readFileSync('team_structure.yaml', 'utf8'));

const cookieParser = require('cookie-parser');
app.use(cookieParser());

app.use(function (req, res, next) {
  // check if client sent cookie
  var cookie = req.cookies.simulator_cookie;
  if (cookie === undefined) {
    // no: set a new cookie
    var randomNumber=Math.random().toString();
    randomNumber=randomNumber.substring(2,randomNumber.length);
    res.cookie('simulator_cookie',randomNumber, { maxAge: 900000, httpOnly: true });
    console.log('cookie created successfully');
  }
  next(); // <-- important!
});

app.use("/rrweb",express.static( __dirname + "/node_modules/rrweb/dist/"));
app.use(express.static(__dirname + "/public"));


function getCookie(cookie, name){
    cookie = ";"+cookie;
    cookie = cookie.split("; ").join(";");
    cookie = cookie.split(" =").join("=");
    cookie = cookie.split(";"+name+"=");
    if(cookie.length<2){
        return null;
    }
    else{
        return decodeURIComponent(cookie[1].split(";")[0]);
    }
}


var Filter = require('bad-words'),
    filter = new Filter();

const { exec } = require("child_process");
var window_name = '';

var char_replacement = [{'Up':'Up','Down':'Down','Left':'Left','Right':'Right'},{'Up':'W','Down':'S','Left':'A','Right':'D'}];
var clients_ids = [], user_ids_list = [], ai_ids_list = [], ai_ids = [], all_ids = [], all_ids_list = [], stats = {}, saved_events = [], saved_key_strokes = [], saved_tutorial_state = [];
var wait_ids = [], redirect_ids = [], waiting_room = false, wait_cookies=[], redirect_cookies=[], redirect_questions=[], redirect_answers=[], use_cookies = command_line_options.cookies;
var init_xdotool = false;
var video_idx_broadcaster = 0;
var past_timer2 = 0, last_time = 0;
var message_sent = false;
const disable_list = [];

var reset_count = 0;

var passcode = [];

if(! command_line_options.wait){

    if(! command_line_options.password){
        passcode = [Math.random().toString(36).substring(2,7)];
    }
    else{
        passcode = command_line_options.password.split(",");
    }
} else{
    waiting_room = true;
}


console.log("Code: ", passcode);


function socket_to_simulator_id(socket_id){
  return all_ids_list[all_ids.indexOf(socket_id)];
}

function simulator_id_to_socket(simulator_id){
  return all_ids[all_ids_list.indexOf(simulator_id)];
}

function eliminate_from_waiting_room(socket, socket_id){

    if(wait_ids.includes(socket_id)){
        const wait_index = wait_ids.indexOf(socket_id);
		wait_ids.splice(wait_index, 1);
		wait_cookies.splice(wait_index, 1);
		
		for (let id_idx = 0; id_idx < wait_ids.length; id_idx++) {
  		    socket.to(wait_ids[id_idx]).emit("waiting_participants", wait_ids.length, user_ids_list.length);
	    }
    }
    
    if(redirect_ids.includes(socket_id)){
        const redirect_index = redirect_ids.indexOf(socket_id);
		redirect_ids.splice(redirect_index, 1);
		redirect_cookies.splice(redirect_index, 1);
		
		for (let id_idx = 0; id_idx < redirect_ids.length; id_idx++) {
  		    socket.to(redirect_ids[id_idx]).emit("redirecting_participants", redirect_ids.length);
  		}
		
    }
}

function get_waiting_time(start){

    var time_limit = map_config["timer_limit"];
    
    if(map_config["scenario"] == 2){
        time_limit *= 2;
    }
    
    if(! start){
        time_limit -= last_time;
    }
    
    console.log("Timer", time_limit, last_time, map_config["timer_limit"]);
    
    return time_limit;
    
}

io.sockets.on("error", e => console.log(e));
io.sockets.on("connection", socket => { //When a client connects

  //console.log("connected!!", socket.id);

  socket.on("broadcaster_load", () => {
    console.log("broadcaster_log", video_idx_broadcaster, socket.id)
    socket.emit("simulator", video_idx_broadcaster, map_config);
  });

  socket.on("broadcaster", () => { //When the broadcaster client connects
    broadcaster = socket.id;
    socket.broadcast.emit("broadcaster");
    
    /*  
    //Initiate key press forwarding to the simulator through xdotool by getting the simulators window name
    if(! init_xdotool){
        exec('xdotool search --name TDW', (error, stdout, stderr) => {
            console.log("window_name: " + stdout);
            window_name = stdout.trim();
        });
        init_xdotool = true;
    }
    */
  });

  socket.on("watcher", (client_number, code) => { //When a human client connects


    if(passcode.includes(code)){


	    socket.to(broadcaster).emit("watcher", socket.id, client_number);

	    if(client_number != 0){
	    
	    	console.log(all_ids, socket.id, client_number)
			clients_ids[client_number-1] = socket.id;
			all_ids[client_number-1] = socket.id;
		
		
			socket.emit("watcher", user_ids_list[client_number-1], map_config, yaml_doc);
			
			if(saved_tutorial_state[client_number-1]){
			    socket.emit("tutorial", saved_tutorial_state[client_number-1]);
			    saved_tutorial_state[client_number-1] = null;
			}
			
			time_sync[client_number-1] = {"offset":Date.now(), "latency":0};
			socket.emit("ping");
			
	    }
    } else{
    	socket.emit("passcode-rejected");
    }
        
    
    /*
    if (clients_ids.includes(socket.id) == false){
        clients_ids.push(socket.id);
        all_ids.push(socket.id);
    }
    */
  });
  
  socket.on("pong", (client_time) => {
    client_number = all_ids.indexOf(socket.id);
    time_sync[client_number]["latency"] = Date.now() - time_sync[client_number]["offset"];
    time_sync[client_number]["offset"] = time_sync[client_number]["offset"] - (client_time - time_sync[client_number]["latency"]/2)
    
    console.log(client_number, time_sync)
  });
  
  socket.on("watcher_ai", (client_number, use_occupancy, server_address, view_radius, centered, skip_frames) => { //When an ai client connects
    console.log("watcher_ai")
    
    if(client_number != 0){
    
        if(! use_occupancy){
            client_number_adapted = client_number + user_ids_list.length;
            socket.to(broadcaster).emit("watcher_ai", socket.id, client_number_adapted, server_address, ai_ids_list[client_number-1]);
        } else {
            socket.to(simulator).emit("watcher_ai", ai_ids_list[client_number-1], view_radius, centered, skip_frames)
        }
        ai_ids[client_number-1] = socket.id;
        all_ids[client_number-1+user_ids_list.length] = socket.id;

        socket.emit("watcher_ai", ai_ids_list[client_number-1], map_config, dateTime, yaml_doc);
        /*
        if (ai_ids.includes(socket.id) == false){
            ai_ids.push(socket.id);
            all_ids.push(socket.id);
        }
        */
    }

  });


  socket.on("occupancy_map", (client_number, object_type_coords_map, object_attributes_id, objects_held) => { //Occupancy maps forwarding
    //console.log(`Sending to ${client_number}`);
    if(client_number != 0){
        socket.to(all_ids[client_number]).emit("occupancy_map", object_type_coords_map, object_attributes_id, objects_held)
    }
  });

  socket.on("simulator", (user_ids, ai_agents_ids, video_idx, config, log_file_name, timer, true_time) => { //When simulator connects
    simulator = socket.id;
    user_ids_list = user_ids;
    ai_ids_list = ai_agents_ids;
    all_ids_list = user_ids.concat(ai_agents_ids);
    clients_ids = Array.apply(null, Array(user_ids_list.length));
    ai_ids = Array.apply(null, Array(ai_ids_list.length));
    all_ids = Array.apply(null, Array(ai_ids_list.length+user_ids_list.length));
    saved_events = Array.apply(null, Array(user_ids_list.length));
    time_sync = Array.apply(null, Array(ai_ids_list.length+user_ids_list.length));
    saved_tutorial_state = Array.apply(null, Array(user_ids_list.length));

    dateTime = log_file_name;
    
    for (let id_idx = 0; id_idx < all_ids_list.length; id_idx++) {
    	stats[all_ids_list[id_idx]] = {'average_message_length':0,'num_messages_sent':0, 'voted':false};
    }
    
    video_idx_broadcaster = video_idx;
    map_config = config;
    
    if(command_line_options.log){
    	fs.appendFile(dir + dateTime + '.txt', String(timer.toFixed(2)) + ',4,0,' + String(true_time.toFixed(3)) + '\n', err => {});
    }
    
    if(! command_line_options.wait){ //enable timer if we are not waiting for anyone
        socket.emit("enable_timer");
    }

  });
  
  
  socket.on("reset", () => {
  	
   if(all_ids.includes(socket.id)){
		var sim_id = socket_to_simulator_id(socket.id);
		if(! stats[sim_id]['voted']){
			stats[sim_id]['voted'] = true;
			reset_count += 1;
		}
	}
  	
  	console.log(stats, reset_count, all_ids.length);
  	
  	if(socket.id == broadcaster || reset_count == all_ids.length){

    	socket.to(simulator).emit("reset"); //, socket_to_simulator_id(socket.id));
    	
	} 
  });
  
  socket.on("reset_partial", () => {
  
    socket.to(simulator).emit("reset_partial");
  
  });
  
  
  socket.on("reset_tutorial", () =>{
    socket.to(simulator).emit("reset_tutorial");
  });
  
   socket.on("reset_ai", () => {

   	socket.to(simulator).emit("reset"); //, socket_to_simulator_id(socket.id));
  });

  //WEBRTC connection setup
  socket.on("offer", (id, message) => {
    socket.to(id).emit("offer", socket.id, message, user_ids_list[clients_ids.indexOf(id)]);
  });
  socket.on("answer", (id, message) => {
    socket.to(id).emit("answer", socket.id, message);
  });
  socket.on("offer_ai", (id, message, fn) => {
    console.log(fn)
    socket.to(id).emit("offer_ai", socket.id, message, fn);
  });
  socket.on("answer_ai", (message) => {
    socket.to(broadcaster).emit("answer_ai", socket.id, message);
  });
  socket.on("candidate", (id, message) => {
    socket.to(id).emit("candidate", socket.id, message);
  });

  socket.on("get_id", () => {
    socket.to(socket.id).emit("get_id", clients_ids.indexOf(socket.id));
  });
  socket.on("disconnect", () => {
    socket.to(broadcaster).emit("disconnectPeer", socket.id);
    
    eliminate_from_waiting_room(socket, socket.id);
    
  });

  
  socket.on("ai_action", (action_message) => {//AI action forwarding
    socket.to(simulator).emit("ai_action",action_message,socket_to_simulator_id(socket.id));
  });
  socket.on("ai_status", (idx, status) => {//AI status forwarding
    socket.to(all_ids[idx]).emit("ai_status",status);
  });
  
  socket.on("ai_output", (idx, object_type_coords_map, object_attributes_id, objects_held, sensing_results, ai_status, extra_status, strength, timer, disable, location, dropped_objects) => {//AI output forwarding
    socket.to(all_ids[idx]).emit("ai_output", object_type_coords_map, object_attributes_id, objects_held, sensing_results, ai_status, extra_status, strength, timer, disable, location, dropped_objects);
    //Sensing actions missing
    if((! disable_list.includes(all_ids_list[idx])) && disable){
    	disable_list.push(all_ids_list[idx]);
    }
    
    
    last_time = timer;
  });
  
  
  socket.on("human_output", (idx, location, item_info, neighbors_info, timer, disable, dropped_objects, objects_held) => {
    socket.to(all_ids[idx]).emit("human_output", location, item_info, neighbors_info, timer, disable, objects_held);
    
    if(command_line_options.log && Object.keys(item_info).length > 0){ //(timer - past_timer > 1 || Object.keys(item_info).length > 0)){
        fs.appendFile(dir + dateTime + '.txt', String(timer.toFixed(2)) +',' + '3' + ',' + socket_to_simulator_id(all_ids[idx]) + ',' + JSON.stringify(item_info) + '\n', err => {}); //+ ',' + JSON.stringify(neighbors_info) + '\n', err => {});

		/*
        if(disable){
        	fs.appendFile(dir + dateTime + '.txt', String(timer.toFixed(2)) + ',5,' + socket_to_simulator_id(all_ids[idx]) + '\n', err => {});
        }
        */
    }
    
    if((! disable_list.includes(all_ids_list[idx])) && disable){
    	disable_list.push(all_ids_list[idx]);
    }
    
    last_time = timer;
    
  });
  
  socket.on("log_output", (location_map, timer) => {

	if(command_line_options.log && (timer - past_timer2 > 0.1 || message_sent)){
		past_timer2 = timer;
		message_sent = false;
  		fs.appendFile(dir + dateTime + '.txt', String(timer.toFixed(2)) +',' + '0' + ',' + JSON.stringify(location_map) + '\n', err => {});
	}
  });
  
  socket.on("get_config", () => {
    yaml_doc = yaml.load(fs.readFileSync('team_structure.yaml', 'utf8'));
    socket.emit("agent_reset", map_config, yaml_doc);
  });
  
  socket.on("agent_reset", (magnebot_id, timer, true_time, config) => {
  
  	reset_count = 0;
  	
  	map_config = config;
  	
    if(disable_list.includes(magnebot_id)){

		const disable_index = disable_list.indexOf(magnebot_id);
		disable_list.splice(disable_index, 1);
    }

    console.log(magnebot_id, simulator_id_to_socket(magnebot_id))
    yaml_doc = yaml.load(fs.readFileSync('team_structure.yaml', 'utf8'));
    socket.to(simulator_id_to_socket(magnebot_id)).emit("agent_reset", map_config, yaml_doc);
    
    

	stats[magnebot_id] = {'average_message_length':0,'num_messages_sent':0, 'voted':false};
    
    
    if(command_line_options.log){
    	fs.appendFile(dir + dateTime + '.txt', String(timer.toFixed(2)) + ',4,' + magnebot_id + ',' + String(true_time.toFixed(3)) + ',' + JSON.stringify(yaml_doc) + '\n', err => {});
    }
    
    past_timer2 = 0;
  });
  
  socket.on("reset_announcement", (magnebot_id) => {
  
    socket.to(simulator_id_to_socket(magnebot_id)).emit("reset_announcement");
  
  });
  
  
  socket.on("message", (message, timestamp, neighbors_list, robot_state) => { //Forwarding messages between robots

    /*
    var neighbor_keys = Object.keys(neighbors_list);
    for(let c in clients_ids){
        if(! (clients_ids[c] === socket.id)){
            //console.log("really sent message")
            socket.to(clients_ids[c]).emit("message", message, user_ids_list[clients_ids.indexOf(socket.id)]);
        }
    }
    */
    //const origin_id = user_ids_list[clients_ids.indexOf(socket.id)]
    
    
    let source_id = socket_to_simulator_id(socket.id);
    
    
    
    if(socket.id == broadcaster){
    
        message = filter.clean(message); //censor
    
        console.log(timestamp,source_id,message, neighbors_list);
    
        if(Object.keys(neighbors_list).length == 0){
            for (let id_idx = 0; id_idx < clients_ids.length; id_idx++) {
                socket.to(clients_ids[id_idx]).emit("message", message, timestamp, "ADMIN", []);
            }
        } else{
            for (const [key, value] of Object.entries(neighbors_list)) {
                if(value === 'human'){
		            let c = clients_ids[user_ids_list.indexOf(key)]; 
		            
			        //console.log(c)
			        socket.to(c).emit("message", message, timestamp, "ADMIN", []);
		            
		        }
            }
        }
    } else{
    
        let whole_message = message["whole"];
        socket.to(broadcaster).emit("message", whole_message, timestamp, source_id, robot_state);
        
        //message = filter.clean(message); //censor
    
        
        //console.log(user_ids_list,ai_ids_list)
    
        if(! disable_list.includes(source_id)){
		    message_sent = true;
		    //console.log("not disabled 1")
		    
		    if(stats[source_id]["average_message_length"] > 0){
			    stats[source_id]["average_message_length"] = (stats[source_id]["average_message_length"] + whole_message.length)/2;
		    } else{
			    stats[source_id]["average_message_length"] = whole_message.length;
		    }
		    stats[source_id]["num_messages_sent"] += 1;
		    if(all_ids.indexOf(socket.id) >= 0){
		        //console.log("not disabled 2")
		        
		        //console.log(source_id)
		        //console.log(neighbors_list)
		        
		        var keys_neighbors = '"';
		        var debug_message = '';
		        
		        if (command_line_options.messageLoop){
			        socket.emit("message", message, timestamp, source_id, robot_state); //Emit message to itself
		        }
		        
		        //console.log(timestamp,source_id,message);
		        for (const [key, value] of Object.entries(message)) {
		            //console.log(key)
		            //console.log(value)
		            
		            debug_message += key + ": " + value
		            
		            if(key != "whole" && ! disable_list.includes(key) && value){
		                //console.log("not disabled 3")
				        keys_neighbors += key + ',';
				        
				        
				        
				        if(user_ids_list.includes(key)){
				            let c = clients_ids[user_ids_list.indexOf(key)]; 
				            
					        socket.to(c).emit("message", value, timestamp, source_id, robot_state);
				            
				        } else if(ai_ids_list.includes(key)){
				            let c = ai_ids[ai_ids_list.indexOf(key)];
				            socket.to(c).emit('message', value, timestamp, source_id, robot_state);
				        }
			        }
		            
		        }
		        
		        console.log(timestamp,source_id,debug_message);
		        
		        keys_neighbors += '"';
		        
		        
		        if(command_line_options.log){
		            fs.appendFile(dir + dateTime + '.txt', String(timestamp.toFixed(2)) +',' + '2' + ',' + socket_to_simulator_id(socket.id) + ',' + '"'+message.replace(/"/g, '\\"')+'"'+','+keys_neighbors+'\n', err => {});
		        }
		    }
        }
    }

  });
  //Every time a key is pressed by someone in their browser, emulate that keypress using xdotool
  socket.on("key", (key, timestamp) => {
  
  	var sim_id = socket_to_simulator_id(socket.id);
  	
    if(! disable_list.includes(sim_id)){
		//console.log(sim_id, socket.id, all_ids);
		socket.to(simulator).emit("key", key, sim_id);
		if(command_line_options.log){
		    fs.appendFile(dir + dateTime + '.txt', String(timestamp.toFixed(2)) +',' + '1' + ',' + sim_id + ',' + key +'\n', err => {});
		}
	}
    /*
    let idx = clients_ids.indexOf(socket.id);
    
    
    if (window_name && idx >= 0){
        key = key.replace('Arrow', '');
        
        //key = char_replacement[idx][key];
        exec('xdotool key --window ' + window_name + ' ' + key, (error, stdout, stderr) => {
            if (stdout)
                console.log("stdout: " + stdout);
            if (stderr)
                console.log("stderr: " + stderr);
            if (error !== null) {
                console.log("exec error: " + error);
            }
        });
    }
    */
  });
  socket.on("objects_update", (target_id, objects_list) => { //Every time someone discovers/shares an object
    //console.log("stdout: " + socket.id + " " + broadcaster);
    socket.to(simulator_id_to_socket(target_id)).emit("objects_update", objects_list, socket_to_simulator_id(socket.id));
  });
  


  socket.on("neighbors_update", (target_id, neighbors_list) => { //Evertime someone gets close to other robots
    socket.to(simulator_id_to_socket(target_id)).emit("neighbors_update", neighbors_list, socket_to_simulator_id(socket.id));
  });
  socket.on("set_goal", (obj_id) => { //Set visual goal
    socket.to(simulator).emit("set_goal",clients_ids.indexOf(socket.id), obj_id);
  });
  
  
  socket.on("disable", () => {
  	socket.to(simulator).emit("disable", socket_to_simulator_id(socket.id));
  	

  });
  
  socket.on("text_processing", (state) => {
  
    for (let id_idx = 0; id_idx < clients_ids.length; id_idx++) {
        socket.to(clients_ids[id_idx]).emit("text_processing", state, socket_to_simulator_id(socket.id));
    }
  	
  });
  
  socket.on("stats", (magnebot_id, stats_dict, timer, final) => {
  	stats_dict["average_message_length"] = stats[magnebot_id]["average_message_length"].toFixed(1);
  	stats_dict["num_messages_sent"] = stats[magnebot_id]["num_messages_sent"];
  	socket.to(simulator_id_to_socket(magnebot_id)).emit("stats", stats_dict, final);
  	
  	if(command_line_options.log){
    	fs.appendFile(dir + dateTime + '.txt', String(timer.toFixed(2)) + ',5,' + magnebot_id + ',' + JSON.stringify(stats_dict) + '\n', err => {});
    }
    
    if(map_config["scenario"] != 2 && final && command_line_options.wait){ //Rest experimental session
        passcode = [];

        var client_number = all_ids_list.indexOf(magnebot_id);

        console.log(client_number, ai_ids, all_ids)
        if(client_number < clients_ids.length){
            clients_ids[client_number] = null;
            all_ids[client_number] = null;
        }// else{
        //    ai_ids[client_number-clients_ids.length] = null;
        //}
        //all_ids[client_number] = null;
        
        waiting_room = true;
        
        
        if(wait_ids.length >= user_ids_list.length){
            for (let id_idx = 0; id_idx < wait_ids.length; id_idx++) {
    		    socket.to(wait_ids[id_idx]).emit("enable_button");
            }
        } else{
            for (let id_idx = 0; id_idx < wait_ids.length; id_idx++) {
    		    socket.to(wait_ids[id_idx]).emit("deactivate_timer");
            }
        }    
        
    }
    
  });
  
  socket.on("sim_crash", (timer) => {
  	//socket.to(simulator).emit("disable", socket_to_simulator_id(socket.id));
  	for (let id_idx = 0; id_idx < all_ids.length; id_idx++) {
  		stats[all_ids_list[id_idx]] = {'average_message_length':0,'num_messages_sent':0, 'voted':false};
  		socket.to(all_ids[id_idx]).emit("sim_crash");
	}
	
	if(command_line_options.log){
    	fs.appendFile(dir + dateTime + '.txt', String(timer.toFixed(2)) + ',6\n', err => {});
    }

  });
  
  socket.on("tutorial", (magnebot_id, state) => {
    if(simulator_id_to_socket(magnebot_id)){
  	    socket.to(simulator_id_to_socket(magnebot_id)).emit("tutorial", state);
  	} else{
  	    uidx = user_ids_list.indexOf(magnebot_id);
  	    saved_tutorial_state[uidx] = state;
  	}
  	

  });
  
  socket.on("log_user_events", (events) => {
  
    //console.log("Saving events!", events)
    const arr_idx = clients_ids.indexOf(socket.id);
  
    if(command_line_options.log && arr_idx >= 0){

        const next_events = JSON.parse(events);
        
        if(saved_events[arr_idx]){
            saved_events[arr_idx] = saved_events[arr_idx].concat(next_events["events"]);
            saved_key_strokes[arr_idx] = saved_key_strokes[arr_idx].concat(next_events["key_events"]);
        } else{
            saved_events[arr_idx] = next_events["events"];
            saved_key_strokes[arr_idx] = next_events["key_events"];
        }
        
        save_json = JSON.stringify({"events": saved_events, "key_events": saved_key_strokes, "time": time_sync});
        fs.writeFile(dir + dateTime + '_events.txt', save_json, err => {});
    }
    
  });
  
  socket.on("replay_user_events", (client_number) => {
  
  
    if(replay_file){
    
        fs.readFile(replay_file, (err,data) => {
        
        
            const contents = JSON.parse(data);
            arr_idx = client_number-1;
          
            string_events = JSON.stringify({"events": contents["events"][arr_idx], "key_events": contents["key_events"][arr_idx]});
          
            socket.emit("replay_user_events", string_events);
        });
    }
    
  });
  
  socket.on("report", (object_list) => {
  
    socket.to(simulator).emit("report", object_list, socket_to_simulator_id(socket.id));
  });
  
  socket.on("survey", (timer, survey_questions, survey_responses, token) => {
  
    if(command_line_options.log){
    	fs.appendFile(dir + dateTime + '.txt', String(timer.toFixed(2)) + ',8,' + JSON.stringify(survey_questions) + "," + JSON.stringify(survey_responses) + ',' + String(token) + '\n', err => {});
    }
    console.log("Survey:", survey_responses, token)
  });
  
  socket.on("join_wait", () => {
  
    var cookie = getCookie(socket.request.headers.cookie,'simulator_cookie');
  
    if(command_line_options.wait && cookie && ! (use_cookies && wait_cookies.includes(cookie))){
  
        wait_ids.push(socket.id);
        wait_cookies.push(cookie);
        
        console.log(socket.request.connection.remoteAddress, socket.request.connection.remotePort, getCookie(socket.request.headers.cookie,'simulator_cookie'), socket.handshake.headers.cookie)

        for (let id_idx = 0; id_idx < wait_ids.length; id_idx++) {
        
            if(wait_ids[id_idx] == socket.id){
                socket.emit("waiting_participants", wait_ids.length, user_ids_list.length);
            } else{
      		    socket.to(wait_ids[id_idx]).emit("waiting_participants", wait_ids.length, user_ids_list.length);
      		}
      		
      		if(wait_ids.length >= user_ids_list.length && waiting_room){
      		    if(wait_ids[id_idx] == socket.id){
      		        socket.emit("enable_button");
      		    } else{
      		        socket.to(wait_ids[id_idx]).emit("enable_button");
      		    }
      		    
      		}
        }
        
        if(! waiting_room){
            socket.emit("disable_button", get_waiting_time(false));
        }
        
    } else{
        socket.emit("error_message", "Sorry, we are not allowing any more participants at this time");
    }

    
  });
  
  socket.on("redirect_session", (demographics_questions, demographics_answers) => {

    var cookie = getCookie(socket.request.headers.cookie,'simulator_cookie');

    if(command_line_options.wait && waiting_room && ! (use_cookies && redirect_cookies.includes(cookie))){
    
        redirect_ids.push(socket.id);
        redirect_cookies.push(cookie);
        redirect_questions.push(demographics_questions);
        redirect_answers.push(demographics_answers);


        if(redirect_ids.length == user_ids_list.length){
            socket.to(simulator).emit("reset_tutorial");
            socket.to(simulator).emit("enable_timer");
            waiting_room = false;
            for (let id_idx = 0; id_idx < redirect_ids.length; id_idx++) {
                individual_passcode = Math.random().toString(36).substring(2,7);
                passcode.push(individual_passcode);
                
                if(command_line_options.log){
    	            fs.appendFile(dir + dateTime + '.txt', '0.00' + ',7,' + String(id_idx+1) + ',' + redirect_cookies[id_idx] + ',' + JSON.stringify(redirect_questions[id_idx]) + ',' + JSON.stringify(redirect_answers[id_idx]) + '\n', err => {});
                }
                
                console.log(id_idx, individual_passcode)
                if(redirect_ids[id_idx] == socket.id){
                    socket.emit("redirect_session", id_idx+1, individual_passcode);
                } else{
                    socket.to(redirect_ids[id_idx]).emit("redirect_session", id_idx+1, individual_passcode);
                }
            }
            
            redirect_questions = [];
            redirect_answers = [];
            
            for (let id_idx = 0; id_idx < wait_ids.length; id_idx++) {
            
                if(! redirect_ids.includes(wait_ids[id_idx])){ //For all those that are not admitted into the session
                    if(wait_ids[id_idx] == socket.id){
                        socket.emit("disable_button", get_waiting_time(true));
                    } else{
                        socket.to(wait_ids[id_idx]).emit("disable_button", get_waiting_time(true));
                    }
                } else{
                    eliminate_from_waiting_room(socket, wait_ids[id_idx]);
                }
            }
            
        } else{
            for (let id_idx = 0; id_idx < redirect_ids.length; id_idx++) {
                if(redirect_ids[id_idx] == socket.id){
                    socket.emit("redirecting_participants", redirect_ids.length);
                } else{
          		    socket.to(redirect_ids[id_idx]).emit("redirecting_participants", redirect_ids.length);
          		}
          		
      		}
        }
    
    } else{
        socket.emit("error_message", "Sorry, we are not allowing any more participants at this time");
    }
    
  });
  
  socket.on("agent_delete", (magnebot_id) => {
  
    socket.to(simulator_id_to_socket(magnebot_id)).emit("agent_delete");

  });
  
  
});

//server.listen(port, host, () => console.log(`Server is running on port ${port}`));


