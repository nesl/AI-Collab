const socket = io.connect(window.location.origin);

const original_title = document.title;

const audioElement = new Audio("/media/notification.mp3");

var out_of_focus = false;
var complete_title = original_title;

socket.on("connect", () => {
  socket.emit("join_wait");
});

socket.on("enable_button", () => {
  document.getElementById('ready-btn').disabled = false;
  document.getElementById('ready-btn').textContent = "Click here to join!";

});

socket.on("error_message", (msg) => {
  document.getElementById('error-msg').textContent = msg;
});

function readyFunction(){
    socket.emit("redirect_session");
    document.getElementById('ready-btn').remove();
    
    document.getElementById('waiting-txt').textContent = "Waiting for other players...";
}

socket.on("redirect_session", (client_id, passcode) => {
    console.log("redirecting...")
    window.location.replace('https://128.97.92.77:5683/?client=' + String(client_id) + '&pass=' + String(passcode));
});

socket.on("waiting_participants", (num_participants, total) => {
  document.getElementById('wait-count').textContent = String(num_participants);
  
  var left = parseInt(total) - parseInt(num_participants);
  
  if(left < 0){
    left = 0;
  }
  
  if(document.getElementById('ready-btn')){
      document.getElementById('ready-btn').textContent = "Waiting for " + String(left) + " more";
      document.getElementById('ready-btn').disabled = true;
  }
  
  audioElement.play();
  
  complete_title = "(" + String(num_participants) + ") " + original_title;
  
  document.title = complete_title;
  
  if(out_of_focus){
    document.title = "*" + document.title;
  }
  
});

socket.on("redirecting_participants", (num_participants) => {
  document.getElementById('redirect-count').textContent = "(Ready: " + String(num_participants) + ")";
});

// when the user loses focus
window.addEventListener("blur", () => {
    out_of_focus = true;
});

// when the user's focus is back to your tab (website) again
window.addEventListener("focus", () => {
    out_of_focus = false;
    
    if(document.title != complete_title){
        document.title = complete_title;
    }
});

