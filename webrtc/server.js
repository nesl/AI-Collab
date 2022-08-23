const express = require("express");
const app = express();

let broadcaster;
const port = 4000;

const http = require("http");
const server = http.createServer(app);

const io = require("socket.io")(server);
app.use(express.static(__dirname + "/public"));

const { exec } = require("child_process");
var window_name = '';

exec('xdotool search --name TDW', (error, stdout, stderr) => {
    console.log("window_name: " + stdout);
    window_name = stdout.trim();
});

io.sockets.on("error", e => console.log(e));
io.sockets.on("connection", socket => {
  socket.on("broadcaster", () => {
    broadcaster = socket.id;
    socket.broadcast.emit("broadcaster");
  });
  socket.on("watcher", () => {
    socket.to(broadcaster).emit("watcher", socket.id);
  });
  socket.on("offer", (id, message) => {
    socket.to(id).emit("offer", socket.id, message);
  });
  socket.on("answer", (id, message) => {
    socket.to(id).emit("answer", socket.id, message);
  });
  socket.on("candidate", (id, message) => {
    socket.to(id).emit("candidate", socket.id, message);
  });
  socket.on("disconnect", () => {
    socket.to(broadcaster).emit("disconnectPeer", socket.id);
  });
  socket.on("key", key => {
    //socket.to(broadcaster).emit("key", socket.id, key);
    if (window_name){
        key = key.replace('Arrow', '');
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
});
server.listen(port, () => console.log(`Server is running on port ${port}`));
