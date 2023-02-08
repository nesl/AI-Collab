const express = require("express");
const app = express();

const commander = require('commander');

commander
  .version('1.0.0', '-v, --version')
  .usage('[OPTIONS]...')
  .option('--address <value>', 'IP address', '172.17.15.69')
  .option('--port <number>', 'Port', 4000)
  .parse(process.argv);

const command_line_options = commander.opts();

let broadcaster;
let simulator;
var map_config;
const port = parseInt(command_line_options.port);
const host = command_line_options.address;//'172.17.15.69'; //'localhost';

const https = require("https");

const fs = require("fs");

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
app.use(express.static(__dirname + "/public"));



const { exec } = require("child_process");
var window_name = '';

var char_replacement = [{'Up':'Up','Down':'Down','Left':'Left','Right':'Right'},{'Up':'W','Down':'S','Left':'A','Right':'D'}];
var clients_ids = [], user_ids_list = [], ai_ids_list = [], ai_ids = [], all_ids = [], all_ids_list = [];
var init_xdotool = false;
var video_idx_broadcaster = 0;


io.sockets.on("error", e => console.log(e));
io.sockets.on("connection", socket => { //When a client connects
  socket.on("broadcaster_load", () => {
    console.log("broadcaster_log", video_idx_broadcaster, socket.id)
    socket.emit("simulator", video_idx_broadcaster);
  });

  socket.on("broadcaster", () => { //When the broadcaster client connects
    broadcaster = socket.id;
    socket.broadcast.emit("broadcaster");
    
    
    //Initiate key press forwarding to the simulator through xdotool by getting the simulators window name
    if(! init_xdotool){
        exec('xdotool search --name TDW', (error, stdout, stderr) => {
            console.log("window_name: " + stdout);
            window_name = stdout.trim();
        });
        init_xdotool = true;
    }
  });

  socket.on("watcher", (client_number) => { //When a human client connects

    socket.to(broadcaster).emit("watcher", socket.id, client_number);

    clients_ids[client_number-1] = socket.id;
    all_ids[client_number-1] = socket.id;
        
    
    /*
    if (clients_ids.includes(socket.id) == false){
        clients_ids.push(socket.id);
        all_ids.push(socket.id);
    }
    */
  });
  socket.on("watcher_ai", (client_number, use_occupancy, server_address, view_radius, centered) => { //When an ai client connects
    console.log("watcher_ai")
    if(! use_occupancy){
        client_number_adapted = client_number + user_ids_list.length;
        socket.to(broadcaster).emit("watcher_ai", socket.id, client_number_adapted, server_address, ai_ids_list[client_number-1]);
    } else {
        socket.to(simulator).emit("watcher_ai", ai_ids_list[client_number-1], view_radius, centered)
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


  });


  socket.on("occupancy_map", (client_number, object_type_coords_map, object_attributes_id, objects_held) => { //Occupancy maps forwarding
    //console.log(`Sending to ${client_number}`);
    socket.to(all_ids[client_number]).emit("occupancy_map", object_type_coords_map, object_attributes_id, objects_held)
  });

  socket.on("simulator", (user_ids, ai_agents_ids, video_idx, config) => { //When simulator connects
    simulator = socket.id;
    user_ids_list = user_ids;
    ai_ids_list = ai_agents_ids;
    all_ids_list = user_ids.concat(ai_agents_ids);
    clients_ids = Array.apply(null, Array(user_ids_list.length));
    ai_ids = Array.apply(null, Array(ai_ids.length));
    all_ids = Array.apply(null, Array(ai_ids.length+user_ids_list.length));
    video_idx_broadcaster = video_idx;
    map_config = config;

  });

  //WEBRTC connection setup
  socket.on("offer", (id, message) => {
    socket.to(id).emit("offer", socket.id, message, user_ids_list[clients_ids.indexOf(id)]);
  });
  socket.on("answer", (id, message) => {
    socket.to(id).emit("answer", socket.id, message);
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

  
  socket.on("ai_action", (action_message, source_id) => {//AI action forwarding
    socket.to(simulator).emit("ai_action",action_message,source_id);
  });
  socket.on("ai_status", (idx, status) => {//AI status forwarding
    socket.to(all_ids[idx]).emit("ai_status",status);
  });
  socket.on("message", (message,neighbors_list, source_id) => { //Forwarding messages between robots

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
    console.log(source_id)
    console.log(neighbors_list)
    for (const [key, value] of Object.entries(neighbors_list)) {
        console.log(key)
        console.log(value)
        if(value === 'human'){
            let c = clients_ids[user_ids_list.indexOf(key)]; 
            console.log(c)
            socket.to(c).emit("message", message, source_id);
        } else if(value === 'ai'){
            let c = ai_ids[ai_ids_list.indexOf(key)];
            socket.to(c).emit('message', message, source_id);
        }
    }

  });
  //Every time a key is pressed by someone in their browser, emulate that keypress using xdotool
  socket.on("key", key => {
    //socket.to(broadcaster).emit("key", socket.id, key);
    if (window_name){
        key = key.replace('Arrow', '');
        let idx = clients_ids.indexOf(socket.id);
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
  });
  socket.on("objects_update", (idx, objects_list) => { //Every time someone discovers/shares an object
    //console.log("stdout: " + socket.id + " " + broadcaster);
    socket.to(all_ids[idx]).emit("objects_update", objects_list);
  });

  socket.on("neighbors_update", (idx, neighbors_list) => { //Evertime someone gets close to other robots
    socket.to(all_ids[idx]).emit("neighbors_update", neighbors_list);
  });
  socket.on("set_goal", (obj_id) => { //Set visual goal
    socket.to(simulator).emit("set_goal",clients_ids.indexOf(socket.id), obj_id);
  });
});

//server.listen(port, host, () => console.log(`Server is running on port ${port}`));


