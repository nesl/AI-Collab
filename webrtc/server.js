const express = require("express");
const app = express();

let broadcaster;
let simulator;
const port = 4000;
const host = '172.17.15.69'; //'localhost';

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
var clients_ids = [];
var user_ids_list = [];
var init_xdotool = false;


io.sockets.on("error", e => console.log(e));
io.sockets.on("connection", socket => {
  socket.on("broadcaster", () => {
    broadcaster = socket.id;
    socket.broadcast.emit("broadcaster");
    if(! init_xdotool){
        exec('xdotool search --name TDW', (error, stdout, stderr) => {
            console.log("window_name: " + stdout);
            window_name = stdout.trim();
        });
        init_xdotool = true;
    }
  });
  socket.on("watcher", (client_number) => {

    socket.to(broadcaster).emit("watcher", socket.id, client_number);
    if (clients_ids.includes(socket.id) == false){
        clients_ids.push(socket.id);
    }
  });
  socket.on("simulator", (user_ids) => {
    simulator = socket.id;
    user_ids_list = user_ids;

  });
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
  socket.on("message", (message,neighbors_list, source_id) => {

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
            socket.to(simulator).emit('ai_message', message, source_id, key);
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
  socket.on("objects_update", (idx, objects_list) => {
    //console.log("stdout: " + socket.id + " " + broadcaster);
    socket.to(clients_ids[idx]).emit("objects_update", objects_list);
  });

  socket.on("neighbors_update", (idx, neighbors_list) => {
    socket.to(clients_ids[idx]).emit("neighbors_update", neighbors_list);
  });
  socket.on("set_goal", (obj_id) => {
    socket.to(simulator).emit("set_goal",clients_ids.indexOf(socket.id), obj_id);
  });
});

//server.listen(port, host, () => console.log(`Server is running on port ${port}`));


