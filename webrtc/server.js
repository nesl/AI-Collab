const express = require("express");
const app = express();

const commander = require('commander');

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
  .parse(process.argv);

const command_line_options = commander.opts();

let broadcaster;
let simulator;
var map_config;
const port = parseInt(command_line_options.port);
const host = command_line_options.address;//'172.17.15.69'; //'localhost';

const https = require("https");

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

app.use("/rrweb",express.static( __dirname + "/node_modules/rrweb/dist/"));

app.use(express.static(__dirname + "/public"));




var Filter = require('bad-words'),
    filter = new Filter();

const { exec } = require("child_process");
var window_name = '';

var char_replacement = [{'Up':'Up','Down':'Down','Left':'Left','Right':'Right'},{'Up':'W','Down':'S','Left':'A','Right':'D'}];
var clients_ids = [], user_ids_list = [], ai_ids_list = [], ai_ids = [], all_ids = [], all_ids_list = [], stats = {}, saved_events = [], saved_key_strokes = [], saved_tutorial_state = [];
var init_xdotool = false;
var video_idx_broadcaster = 0;
var past_timer = 0, past_timer2 = 0;
var message_sent = false;
const disable_list = [];

var reset_count = 0;

var passcode;

if(! command_line_options.password){
    passcode = [Math.random().toString(36).substring(2,7)];
}
else{
    passcode = command_line_options.password.split(",");
}


console.log("Code: ", passcode);


function socket_to_simulator_id(socket_id){
  return all_ids_list[all_ids.indexOf(socket_id)];
}

function simulator_id_to_socket(simulator_id){
  return all_ids[all_ids_list.indexOf(simulator_id)];
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
		
		
			socket.emit("watcher", user_ids_list[client_number-1], map_config);
			
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

        socket.emit("watcher_ai", ai_ids_list[client_number-1], map_config);
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

  });
  
  
  socket.on("reset", () => {
  	
   if(all_ids.includes(socket.id)){
		var sim_id = socket_to_simulator_id(socket.id);
		if(! stats[sim_id]['voted']){
			stats[sim_id]['voted'] = true;
			reset_count += 1;
		}
	}
  	
  	if(socket.id == broadcaster || reset_count == all_ids.length){

    	socket.to(simulator).emit("reset"); //, socket_to_simulator_id(socket.id));
	} 
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
  });

  
  socket.on("ai_action", (action_message) => {//AI action forwarding
    socket.to(simulator).emit("ai_action",action_message,socket_to_simulator_id(socket.id));
  });
  socket.on("ai_status", (idx, status) => {//AI status forwarding
    socket.to(all_ids[idx]).emit("ai_status",status);
  });
  
  socket.on("ai_output", (idx, object_type_coords_map, object_attributes_id, objects_held, sensing_results, ai_status, extra_status, strength, timer, disable) => {//AI output forwarding
    socket.to(all_ids[idx]).emit("ai_output", object_type_coords_map, object_attributes_id, objects_held, sensing_results, ai_status, extra_status, strength, timer, disable);
    
    if((! disable_list.includes(all_ids_list[idx])) && disable){
    	disable_list.push(all_ids_list[idx]);
    }
    
  });
  
  
  socket.on("human_output", (idx, location, item_info, neighbors_info, timer, disable) => {
    socket.to(all_ids[idx]).emit("human_output", location, item_info, neighbors_info, timer, disable);
    
    if(command_line_options.log && Object.keys(item_info).length > 0){ //(timer - past_timer > 1 || Object.keys(item_info).length > 0)){
        past_timer = timer;
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
    
  });
  
  socket.on("log_output", (location_map, timer) => {

	if(command_line_options.log && (timer - past_timer2 > 0.1 || message_sent)){
		past_timer2 = timer;
		message_sent = false;
  		fs.appendFile(dir + dateTime + '.txt', String(timer.toFixed(2)) +',' + '0' + ',' + JSON.stringify(location_map) + '\n', err => {});
	}
  });
  
  socket.on("agent_reset", (magnebot_id, timer, true_time, config) => {
  
  	reset_count = 0;
  	
  	map_config = config;
  	
    if(disable_list.includes(magnebot_id)){

		const disable_index = disable_list.indexOf(magnebot_id);
		disable_list.splice(disable_index, 1);
    }

    socket.to(simulator_id_to_socket(magnebot_id)).emit("agent_reset", map_config);
    
    

	stats[magnebot_id] = {'average_message_length':0,'num_messages_sent':0, 'voted':false};
    
    
    if(command_line_options.log){
    	fs.appendFile(dir + dateTime + '.txt', String(timer.toFixed(2)) + ',4,' + magnebot_id + ',' + String(true_time.toFixed(3)) + '\n', err => {});
    }
  });
  
  socket.on("reset_announcement", (magnebot_id) => {
  
    socket.to(simulator_id_to_socket(magnebot_id)).emit("reset_announcement");
  
  });
  
  
  socket.on("message", (message, timestamp, neighbors_list) => { //Forwarding messages between robots

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
    
    
    var sim_id = socket_to_simulator_id(socket.id);
    message = filter.clean(message); //censor
    
    console.log(timestamp,sim_id,message);
    
    
    let source_id = socket_to_simulator_id(socket.id)
    
    
    if(socket.id == broadcaster){
        if(Object.keys(neighbors_list).length == 0){
            for (let id_idx = 0; id_idx < clients_ids.length; id_idx++) {
                socket.to(clients_ids[id_idx]).emit("message", message, timestamp, "ADMIN");
            }
        } else{
            for (const [key, value] of Object.entries(neighbors_list)) {
                if(value === 'human'){
		            let c = clients_ids[user_ids_list.indexOf(key)]; 
		            
			        //console.log(c)
			        socket.to(c).emit("message", message, timestamp, "ADMIN");
		            
		        }
            }
        }
    } else{
        socket.to(broadcaster).emit("message", message, timestamp, source_id);
        
    
    
        if(! disable_list.includes(sim_id)){
		    message_sent = true;
		    //console.log("not disabled 1")
		    
		    if(stats[sim_id]["average_message_length"] > 0){
			    stats[sim_id]["average_message_length"] = (stats[sim_id]["average_message_length"] + message.length)/2;
		    } else{
			    stats[sim_id]["average_message_length"] = message.length;
		    }
		    stats[sim_id]["num_messages_sent"] += 1;
		    if(all_ids.indexOf(socket.id) >= 0 && neighbors_list){
		        //console.log("not disabled 2")
		        
		        //console.log(source_id)
		        //console.log(neighbors_list)
		        
		        var keys_neighbors = '"';
		        
		        if (command_line_options.messageLoop){
			        socket.emit("message", message, timestamp, source_id); //Emit message to itself
		        }
		        
		        for (const [key, value] of Object.entries(neighbors_list)) {
		            //console.log(key)
		            //console.log(value)
		            
		            if(! disable_list.includes(key)){
		                //console.log("not disabled 3")
				        keys_neighbors += key + ',';
				        if(value === 'human'){
				            let c = clients_ids[user_ids_list.indexOf(key)]; 
				            
					        //console.log(c)
					        socket.to(c).emit("message", message, timestamp, source_id);
				            
				        } else if(value === 'ai'){
				            let c = ai_ids[ai_ids_list.indexOf(key)];
				            socket.to(c).emit('message', message, timestamp, source_id);
				        }
			        }
		            
		        }
		        
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
  
  socket.on("stats", (magnebot_id, stats_dict, timer, final) => {
  	stats_dict["average_message_length"] = stats[magnebot_id]["average_message_length"].toFixed(1);
  	stats_dict["num_messages_sent"] = stats[magnebot_id]["num_messages_sent"];
  	socket.to(simulator_id_to_socket(magnebot_id)).emit("stats", stats_dict, final);
  	
  	if(command_line_options.log){
    	fs.appendFile(dir + dateTime + '.txt', String(timer.toFixed(2)) + ',5,' + magnebot_id + ',' + JSON.stringify(stats_dict) + '\n', err => {});
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
  
});

//server.listen(port, host, () => console.log(`Server is running on port ${port}`));


