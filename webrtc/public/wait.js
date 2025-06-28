const socket = io.connect(window.location.origin);

const original_title = document.title;

const audioElement = new Audio("/media/notification.mp3");

var out_of_focus = false;
var complete_title = original_title;
var session_ongoing = false;
var countdown_ongoing = null;
var countDownDate = null;


const likertQuestions = [
    'Generally, I trust automated agents',
    'Automated agents help me solve many problems',
    'I think it’s a good idea to rely on automated agents for help',
    'I don’t trust the information I get from automated agents',
    'Automated agents are reliable',
    'I rely on automated agents'
];

// 2. Grab the <ul> container
const list = document.getElementById('question_list');

// 3. For each question, build and append an <li>
likertQuestions.forEach((text, idx) => {
    // li wrapper
    const li = document.createElement('li');
    li.style.marginTop = '0.8vw';

    // the question text
    const p = document.createElement('p');
    const b = document.createElement('b');
    p.className = 'demographics_question';
    b.textContent = text;
    p.appendChild(b);
    li.appendChild(p);

    // container for the radio row
    const row = document.createElement('div');
    Object.assign(row.style, {
      display: 'flex',
      alignItems: 'center',
      gap: '0.5vw',
      marginTop: '0.5vw',
      justifyContent: 'center',
    });

    // left label
    const leftLabel = document.createElement('span');
    leftLabel.textContent = 'Strongly Disagree';
    row.appendChild(leftLabel);

    // the 5 radio buttons
    for (let val = 1; val <= 5; val++) {
      const label = document.createElement('label');
      label.style.display = 'flex';
      label.style.alignItems = 'center';
      label.style.gap = '0.2vw';

      const input = document.createElement('input');
      input.type = 'radio';
      input.name = 'demographics' + String(idx+4);
      input.value = String(val);

      label.appendChild(input);
      label.appendChild(document.createTextNode(val));
      row.appendChild(label);
    }

    // right label
    const rightLabel = document.createElement('span');
    rightLabel.textContent = 'Strongly Agree';
    row.appendChild(rightLabel);

    // assemble and append
    li.appendChild(row);
    list.appendChild(li);
});




function pad(num,size) {
    var s = "00000" + num;
    return s.substr(s.length-size);
}

function countdown_func() {

  // Get today's date and time
  var now = new Date().getTime();
    
  // Find the distance between now and the count down date
  var distance = countDownDate - now;
    
  var minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
  var seconds = Math.floor((distance % (1000 * 60)) / 1000);
  var time_str = pad(String(minutes),2) + ":" + pad(String(seconds),2);
    
  // If the count down is over, write some text 
  if (distance < 0) {
    document.getElementById("ready-btn").innerHTML = "Current session is ending";
    session_ongoing = false;
    clearInterval(countdown_ongoing);
    countdown_ongoing = null;
  } else{
    document.getElementById("ready-btn").innerHTML = "Waiting for current session to end (" + time_str + ")";
  }
}

socket.on("connect", () => {
  socket.emit("join_wait");
});

socket.on("enable_button", () => {
  document.getElementById('ready-btn').disabled = false;
  document.getElementById('ready-btn').textContent = "Click here to join!";
  session_ongoing = false;
  
  if(countdown_ongoing){
    clearInterval(countdown_ongoing);
    countdown_ongoing = null;
  }

});

socket.on("disable_button", (time_to_finish) => {
  document.getElementById('ready-btn').disabled = true;
  session_ongoing = true;
  
  console.log(time_to_finish)
  
  var t = new Date();
  t.setSeconds(t.getSeconds() + time_to_finish);
  countDownDate = t;
  
  countdown_ongoing = setInterval(countdown_func, 1000);
  

});

socket.on("deactivate_timer", () => {

  session_ongoing = false;
  
  
  if(countdown_ongoing){
    clearInterval(countdown_ongoing);
    countdown_ongoing = null;
  }
  
  document.getElementById('ready-btn').textContent = "Waiting for participants";

});

socket.on("error_message", (msg) => {
  document.getElementById('error-msg').textContent = msg;
});


function readyFunction(){

    survey_name = document.getElementById('survey_name').value;
    survey_age = document.getElementById('survey_age').value;
    
    
    var questions = ["Name", "Age"];
    var array_values = [survey_name, survey_age];
    
    const demographicQuestionsElements = document.getElementsByClassName("demographics_question");
    
    for(i = 0; i < demographicQuestionsElements.length; i++) {
    
        questions.push(demographicQuestionsElements[i].textContent);
    
	    var survey_question = document.getElementsByName('demographics' + String(i+1));
	    for(j = 0; j < survey_question.length; j++) {
	        if(survey_question[j].checked){
	            array_values.push(survey_question[j].value);
	            break;
	        }

	    }
	    if(array_values.length <= i+2){
	        array_values.push("");
	    }
	}

    socket.emit("redirect_session", questions, array_values);
    document.getElementById('ready-btn').remove();
    
    document.getElementById('waiting-txt').textContent = "Waiting for other players...";
}

socket.on("redirect_session", (client_id, passcode) => {
    console.log("redirecting...")
    window.location.replace(window.location.origin + '/?client=' + String(client_id) + '&pass=' + String(passcode));
});

socket.on("waiting_participants", (num_participants, total) => {
  document.getElementById('wait-count').textContent = String(num_participants);
  
  var left = parseInt(total) - parseInt(num_participants);
  
  if(left < 0){
    left = 0;
  }
  
  if(document.getElementById('ready-btn') && ! session_ongoing){
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

