const socket = io.connect(window.location.origin);

const original_title = document.title;

const audioElement = new Audio("/media/notification.mp3");

var out_of_focus = false;
var complete_title = original_title;
var session_ongoing = false;
var countdown_ongoing = null;
var countDownDate = null;

var popup_history = [];
var content_popup_1;



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
}

var popup_current_index = 0;

function help_toggle(){

    document.getElementById("popup-1").classList.add("active");
    
    progress_bar = document.getElementById('tutorial_progress');
    progress_bar.childNodes[popup_current_index].style.background = "gray";
    content_popup_1.scrollTo({ top: 0, behavior: 'smooth' });
    
    //popup_text.innerHTML = popup_history[popup_current_index];
    
}

function togglePopup(element){
	document.getElementById(element).classList.toggle("active");
}

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

function build_team_strategy2(version){

    var slide_text = "";

    var teammates_type = [[],[]];
   
    var text_content = "";
    if(version == 0 || version == 1){
        
        text_content = "<b>There will be a human leader</b>. <ol>";
        if(version == 0){
            text_content += "<li>They will be in full command: they’ll determine every action the team takes, the other human and the two AI-controlled robots should simply carry out their instructions—AI will not offer any suggestions or pushback. </li><li>They may issue, revise or override orders for anyone at any time, and should communicate with all agents via chat. </li>";
        } else{
            text_content += "<li>They’re in charge: the other human and the two AI-driven robots should execute their orders. </li><li>They may issue, revise, or cancel commands for any agent at any moment. </li><li>From time to time, the AI robots will propose more efficient actions—recommendations generated by their optimization algorithm—but it’s entirely the leader's decision whether to act on them. </li>";
        }
        
        
        if(version == 1){
            text_content += "<li>The AI agents' algorithm crafts plans that minimize total travel distance, reduce uncertainty about object danger levels, favor dependable agents, and balance overall workload. </li><li>They gauge reliability by past task success, and will offer suggestions more or less frequently based on how often you incorporate their feedback. </li><li>You should always try sharing any object information you have with the AI agents so they can give better suggestions! </li>";
        }
        
        text_content += "<li>Human Leader Instructions:<ul><li><u>Decide who senses an object or room.</u> <b>Example:</b> <i>\"Hey D and B, sense room 1, C you can sense object 3\"</i></li><li><u>Decide who carries an object and with whom.</u> When ordering agents to carry a heavy object, you should indicate who is going to \"lead\" the team (carry the object) in such an effort. <b>Example: </b><i>\"Hey D and B, please carry object 3. D should lead the team. C you can go carry object 2 by yourself\"</i></li></ul></li>";
        
    } else{
        text_content = "<b>There is an AI leader</b>. <ol>";
        
        if(version == 2){
            text_content += "<li>No human may propose changes to the plan. </li><li>The AI leader issues commands to all agents, and those directives must be followed. </li>";
        } else{
            text_content += "<li>The AI leader issues directives to all agents, and those orders must be carried out. </li><li>Human teammates may submit suggestions, which the AI leader will assess and incorporate into its plan only if they’re predicted to improve overall performance. </li>";
        }
        text_content += "<li>It leverages an optimization algorithm to craft the best possible plan given current data—minimizing total travel distance, reducing uncertainty about each object’s danger level, limiting reliance on less reliable agents, and maximizing overall agent deployment. </li><li>Reliability is judged by past task success, and the plan adapts in real time based on how quickly you carry out those tasks. </li><li>You should always be in contact communication with the leader, reporting back when it asks for it! </li>";
        
        if(version == 2){
            text_content += "<li>AI Leader Instructions:<ul><li><u>Decide who senses an object or room.</u> <b>AI Example:</b> <i>\"A, sense room 1\"</i></li><li><u>Decide who carries an object and with whom.</u> When ordering agents to carry a heavy object, you should indicate who is going to \"lead\" the team (carry the object) in such an effort. <b>AI Example: </b><i>\"Team leader: A. Carry object 8 with teammates: [B,C]\"</i></li></ul></li>";
        }else{
            text_content += "<li>Human Suggestions/AI Leader Instructions:<ul><li><u>Decide who senses an object or room.</u> <b>Human Example:</b> <i>\"Hey D and B, sense room 1, C you can sense object 3\"</i>. <b>AI Example:</b> <i>\"A, sense room 1\"</i></li><li><u>Decide who carries an object and with whom.</u> When ordering agents to carry a heavy object, you should indicate who is going to \"lead\" the team (carry the object) in such an effort. <b>Human Example: </b><i>\"Hey D and B, please carry object 3. D should lead the team. C you can go carry object 2 by yourself\".</i> <b>AI Example: </b><i>\"Team leader: A. Carry object 8 with teammates: [B,C]\"</i></li></ul></li>";
        }
        
   
        
    }
    
    text_content += "</ol>"
    
    slide_text += "<p>" + text_content + "</p>";
    
    return slide_text;
    
}

function tutorial_popup(){

  var popup_text = document.getElementById("popup_text");
  popup_text.style.fontSize="0.7vw";
  
  
  popup_text.innerHTML = "";
  
  const scene_title = ["Human leader, no AI suggestions","Human leader with AI suggestions","AI leader, no human suggestions","AI leader with human suggestions"];
  
  const versions = 4;
  
  for (let i = 0; i < versions; i++) {
  
  
  
    

        
        
        if(popup_text.innerHTML){
            popup_history.push(popup_text.innerHTML);
        
        
            if(popup_current_index == popup_history.length-1){
                add_dot(true);
            } else{
                add_dot(false);
            }
            popup_text.innerHTML = "";
        }

          
        popup_text.innerHTML += "<h1>" + scene_title[i] + "</h1>";


        popup_text.innerHTML += build_team_strategy2(i);
        
            
  }
  

  
  
  popup_history.push(popup_text.innerHTML);
  

  
  if(popup_current_index == popup_history.length-1){
    add_dot(true);
  } else{
    add_dot(false);
  }
  
  
  popup_text.innerHTML = popup_history[popup_current_index];
  content_popup_1.scrollTo({ top: 0, behavior: 'smooth' });
}

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
	
	agent_id = document.getElementById('agent_id_question').value;

    socket.emit("redirect_session", questions, array_values, agent_id);
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


tutorial_popup();
