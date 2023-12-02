const queryString = window.location.search;
const urlParams = new URLSearchParams(queryString);
const client_number = urlParams.get('client');

const socket = io.connect(window.location.origin);

socket.emit("replay_user_events", client_number);

socket.on("replay_user_events", (string_events) => {

    const all_events = JSON.parse(string_events)
    const events = all_events["events"]
    const replayer = new rrweb.Replayer(events);
    replayer.play();

});
