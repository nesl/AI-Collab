from transformers import AutoTokenizer,AutoModelForCausalLM
import transformers
import torch
import json
import re
import argparse
import pdb
import os
import openai
import time
from string import Formatter
from groq import Groq
import groq
import numpy as np
import random
import datetime

class Human2AIText:

    def __init__(self, env, robotState, team_structure):
    
        self.openai = True
        
        agent_id = env.robot_id
        self.env = env
        
        self.sql_tables = robotState.create_tables
        
        self.agent_names = list(env.robot_key_to_index.keys())
        self.agent_names.append(agent_id)
            
            
        if self.openai:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.client = openai.OpenAI() #Groq(api_key=os.getenv("GROQ_API_KEY"))
            self.client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        dateTime = env.log_name #datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        if not os.path.exists("logs"):
            os.makedirs("logs")
            
        version = 0
        while os.path.exists("logs/" + dateTime + "_llm_" + agent_id + "_" + str(version) + ".txt"):
            version += 1
            
        self.log_file = open("logs/" + dateTime + "_llm_" + agent_id + "_" + str(version) + ".txt", "w")
        self.START_INST = "[INST]"
        self.END_INST = "[/INST]"
        
        self.START_STR = "<s>"
        self.END_STR = "</s>"

        self.START_SYS = "<<SYS>>"
        self.END_SYS = "<</SYS>>"
        self.noop = "No message matches"
        
        self.exchanged_messages = {agent:[] for agent in list(env.robot_key_to_index.keys())}
        
        self.max_message_history = 5
        
        self.team_structure = team_structure
        
        self.unknown_location = "(99.99,99.99)"
        
        self.time_limit_groq = 60
        
        self.time_counted_groq = 0
        
        self.already_setup_llm = False

        self.conversation_times = 0
        
        self.start_header_str = "<|start_header_id|>"
        self.end_header_str = "<|end_header_id|>"
        self.end_token_str = "<|eot_id|>"
        
        self.change_llm = 0
        self.big_model = "llama-3.3-70b-specdec" #"llama-3.3-70b-versatile" #"llama-3.3-70b-specdec"

        '''
        self.tokenizer2 = AutoTokenizer.from_pretrained("google/gemma-2-9b-it") #"google/gemma-2-2b-it"
        self.model2 = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-9b-it",
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        '''

        self.CNL_MESSAGES = {
            "Ask about agent": "Where is agent {agent_id}. ",
            "No knowledge of agent": "I don't know where is agent {agent_id}. ",
            "Ask about object": "What do you know about object {object_id}. ",
            "No knowledge of object": "I know nothing about object {object_id}. ",
            "Object not found": "Hey {agent_id}, I didn't find object {object_id}. ",
            "Reject help from agent": "Nevermind {agent_id}. ",
            "Help carry object": "I need {agent_count} more robots to help carry object {object_id}. ",
            "Follow someone": "Thanks, I'll follow you {agent_id}. ",
            "Be followed by someone": "Thanks, follow me {agent_id}. ",
            "Cancel help request": "Nevermind. ",
            "End collaboration": "No need for more help. ",
            "Accept request for help": "I can help you {agent_id}. ",
            "Reject request for help": "I cannot help you {agent_id}. ",
            "Reject request to follow/be followed": "I didn't offer my help to you {agent_id}. ",
            "Come closer": "Come closer {agent_id}. ",
            "Request to move": "Hey {agent_id}, I need you to move. ",
            "End the mission": "Let's end participation. ",
            "Don't end the mission": "Wait, let's not end participation yet. ",
            "Object information": "Object {object_id} (weight: {object_weight}) Last seen in {location} at {time}. Status: {danger}, Prob. Correct: {probability}. ",
            "Agent information": "Agent {agent_id} (type: {agent_type}) Last seen in {location} at {time}. ",
            "Not relevant": self.noop
        }


        if "hierarchy" in team_structure and (team_structure["hierarchy"][env.robot_id] == "obey" or team_structure["hierarchy"][env.robot_id] == "order"):  

            CNL_MESSAGES_leadership = {
                "Order to sense": "{agent_id}, sense object {object_id} at location {location}",
                "Order to carry": "{agent_id}, collect object {object_id}",
                "Order to explore": "{agent_id}, go to location {location} and report anything useful",
                "Cannot fulfill order": "I cannot fulfill you order right now {agent_id}", #check
                "Cancel order": "Order cancelled {agent_id}",
                "Order completed": "Order completed"
            }
            
            self.CNL_MESSAGES = {**self.CNL_MESSAGES, **CNL_MESSAGES_leadership}
            
        if not ("hierarchy" in team_structure and team_structure["hierarchy"][env.robot_id] == "obey"):  

            CNL_MESSAGES_equal = {
                "Help sense object": "Hey {agent_id}, can you help me sense object {object_id} in location {location}, last seen at {time}. "
            }
            
            self.CNL_MESSAGES = {**self.CNL_MESSAGES, **CNL_MESSAGES_equal}            

        '''
        self.CNL_MESSAGES = {
            "request_pickup_help": "I need {number_robots} more robots to help carry object {object}. ",
            "request_sensing_help": "Hey {agent}, can you help me sense object {object} in location {coordinates}, last seen at {time}. ",
            "nevermind": "Nevermind. ",
            "offer_help": "I can help you {agent}. ",
            "agree_pickup_help": "Thanks, follow me {agent}. ",
            "agree_sensing_help": "Thanks, I'll follow you {agent}. ",
            "refuse_help": "Nevermind {agent}. ",
            "deny_help": "I cannot help you {agent}. ",
            "finish_help": "No need for more help. ",
            "refuse_help_offer": "Thanks for nothing. ",
            "request_object_info": "What do you know about object {object}. ",
            "request_agent_info": "Where is agent {agent}. ",
            "provide_object_info": "Object {object} (weight: {weight}) Last seen in {coordinates} at {time}. Status: {danger}, Prob. Correct: {probability}. ",
            "provide_agent_info": "Agent {agent} (type: {agent_type}) Last seen in {coordinates} at {time}. ",
            "no_object_info": "I know nothing about object {object}. ",
            "no_agent_info": "I don't know where is agent {agent}. ",
            "object_not_found": "Hey {agent}, I didn't find object {object}. ",
            "make_agent_move": "Hey {agent}, I need you to move. ",
            "no_action": self.noop

        }
        '''
        self.SYS_PROMPT = """You are an assistant responsible for extracting the relevant information from the text 
I give you into a JSON format output. This JSON should fit the following format:

```
{
    "action": "the type of message being used",
    "agent_id": "the agent being referred to in the message",
    "agent_type": "the type of agent (either human or AI)",
    "location": "the coordinates where the object or agent is located",
    "object_id": "the object number being referred to in the message",
    "danger": "a boolean indicating whether the object is dangerous",
    "object_weight": "an integer indicating the weight of the object",
    "probability": "the probability of the danger status prediction being correct",
    "time": "the time mentioned in the message",
    "agent_count": "number of agents being requested help"
}
```
        
You should allways give the JSON inside a code block. The action keyword must always be included.
The other keywords only need to be included if present. 
The type of messages are the following: """ + str(self.CNL_MESSAGES)
        
        """
        The possible actions are:
        -"request_pickup_help": the message is asking for help to pick up an object.
        -"request_sensing_help": the message is asking for help to sense an object.
        -"nevermind": the message is cancelling a previous request for help.
        -"offer_help": the message is offering help to pick up or sense an object.
        -"agree_pickup_help": the message is accepting the help being offered to pick up an object and requesting an agent to follow.
        -"agree_sensing_help": the message is accepting the help being offered to sense an object and telling the agent he will follow him.
        -"refuse_help": the message is refusing help from another agent.
        -"deny_help": the message is rejecting an agent's request to pick up or sense an object.
        -"finish_help": the message is telling an agent that help is no longer needed.
        -"refuse_help_offer": the message is telling an agent help is not needed in an angry manner.
        -"request_object_info": the message requests information about an object.
        -"request_agent_info": the message requests information about an agent.
        -"provide_object_info": the message gives information about an object, such as its weight, last seen location and time, and whether it is dangerous.
        -"provide_agent_info": the message gives information about an agent, such as its last seen location and time.
        -"no_object_info": the message tells an agent it has no information about an object.
        -"no_agent_info": the message tells an agent it doesn't know where an agent is.
        -"make_agent_move": the message tells an agent to move away.
        -"no_action": whenever the message does not conform to any of the previous actions."""

        
        self.SELECT_ACTION = '' #'Select only one of the next as the action: ["Ask about agent","No knowledge of agent","Ask about object","No knowledge of object","Object not found","Help sense object","Help carry object","Reject help from agent","Follow","Be followed","Cancel help request","End collaboration","Accept request for help","Reject request for help","Reject request to follow/be followed","Come closer","Request to move","End participation","Don\'t end participation","Object information","Agent information","Not relevant"]'
        
        self.EXAMPLE_PROMPTS = [
            ("user", "Original message: \"Can someone help me carry object 3?\". Alternative messages: [\"I need assistance with object 3\", \"Can another robot help me?\", \"Does anyone have the strength to lift object 3?\"]. "),
            (
                "llama", """
```
{
    "action": "Help carry object",
    "object_id": "3"
}
```
"""
            ),
            ("user", "Original message: \"I can't help you picking up such object at the moment\". Alternative messages: [\"I'm not able to assist you with that right now\", \"Sorry, I'm not able to pick that up\", \"I don't have the capacity to handle that item\"]. "
),
            (
                "llama", """
```
{
    "action": "Reject request for help"
}
```
"""
            ),
            ("user", "Original message: \"Object 1 (weight: 2) Last seen in (9,8) at 11:30. Status: dangerous, Prob. Correct: 80%\". Alternative messages: [\"Object 1 (weight: 2) last seen near (9,8) at 11:30. Dangerous. Prob. Correct: 80%\", \"Object 1 (weight: 2) located at (9,8) at 11:30. Dangerous. Prob. Correct: 80%\", \"Dangerous Object 1 (weight: 2) last seen at (9,8) at 11:30. Prob. Correct: 80%\"]. "),
            (
                "llama", """
```
{
    "action": "Object information",
    "location": "(9,8)",
    "object_weight": "2",
    "danger": "dangerous",
    "probability": "80.0%",
    "time": "11:30",
    "object_id": "1"
}
```
"""
            ),
        ]
        
        
        self.reply_prompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are Agent """ + agent_id + """. You are part of a team whose mission is to dispose of all dangerous objects in a scene. You can move around, detect whether an object is dangerous or not, and carry objects. Objects are all of the same type, they only differ in their location, weight, and whether they are dangerous. Once you find a dangerous object you must check if you have the necessary strength to pick it up and put it in the safe area you already know. Try to engage with your teammates to come up with a strategy. Be brief in your responses.  Analyze the content of each message and try to tell me who are they answering to. Some messages may be directed towards a subset of the group. Also, score the collaborative disposition of the agent sending the message and the collaborative disposition you think they have of you. Output a JSON format with the following field for each analyzed message: "reply_to", which should be any of the set {"Agent A", "Agent B", "Agent C", "Agent D", "Everyone"}, "their_collaborative_score" and "their_collaborative_score_of_me" , which should be an integer from 0 to 10 and correspond to collaborative score you assign to the sender of the new message and to the collaborative score you think they have of you so far, respectively; and "observations", a string indicating any observations about their collaboration and their perception of my collaboration.
<|eot_id|><|start_header_id|>user<|end_header_id|>

History of messages: [{'Sender': 'Agent D', 'Message': 'Hi everyone'}]
New message: {'Sender': 'Agent B', 'Message': 'hi'}
Who is Agent B replying to? Output a JSON.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

>> {"reply_to": ["Agent D"], "their_collaborative_score": 10, "their_collaborative_score_of_me": 0, "observations": "Agent B greets Agent D back, but I haven't said anything to it so far"}

<|eot_id|><|start_header_id|>user<|end_header_id|>History of messages: [{'Sender': 'Agent A', 'Message': 'Hey C'}, {'Sender': 'Agent A', 'Message': 'C do you have more info'}]
New message: {'Sender': 'Agent C', 'Message': 'Need help?'}
Who is Agent C replying to? Output a JSON.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

>> {"reply_to": ["Agent A"], "their_collaborative_score": 10, "their_collaborative_score_of_me": 9, "observations": "C volunteering to help, I have been requesting their help"}

<|eot_id|><|start_header_id|>user<|end_header_id|>History of messages: [{'Sender': 'Agent B', 'Message': 'hello'}, {'Sender': 'Agent C', 'Message': 'hi'}, {'Sender': 'Agent D', 'Message': 'Hello everyone'}]
New message: {'Sender': 'Agent D', 'Message': 'Do we want to try a different strategy today'}
Who is Agent D replying to? Output a JSON.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

>> {"reply_to": ["Everyone"], "their_collaborative_score": 10, "their_collaborative_score_of_me": 0, "observations": "D tries to organize the team strategy, I haven't said anything to it."}

"""
        
        self.phrase_generation ="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are part of a team whose mission is to dispose of all dangerous objects in a scene. You can use a sensor to sense whether an object is dangerous or benign. You are going to be given a phrase and you have to create variations of it.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Phrase: "I need 2 more robots to help carry object 8."
Write at least 3 possible variations of the phrase and put them inside a list.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        
>> ["can 2 other robots help me carry object 8", "hey I need help to carry an object", "come with me I need help for lifting an object"]

<|eot_id|><|start_header_id|>user<|end_header_id|>Phrase: "I need you to move B."
Write at least 3 possible variations of the phrase and put them inside a list.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        
>> ["please move!", "hey B why don't you move", "can you step aside"]

"""

        self.output_personalization = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are Agent """ + agent_id + """. You are part of a team whose mission is to dispose of all dangerous objects in a scene. You can move around, detect whether an object is dangerous or not, and carry objects. Objects are all of the same type, they only differ in their location, weight, and whether they are dangerous. Once you find a dangerous object you must check if you have the necessary strength to pick it up and put it in the safe area you already know. Try to engage with your teammates to come up with a strategy. Be brief in your responses. You will try to create an appropriate message directed toward a specific agent according to how previous interactions with that agent have been.
<|eot_id|><|start_header_id|>user<|end_header_id|>

History of messages: [{'Sender': 'Agent B', 'Message': 'hello'}, {'Sender': 'Agent C', 'Message': 'hi'}, {'Sender': 'Agent D', 'Message': 'Hello everyone'}]
Message to send: "I need 2 more robots to help carry object 8."
How would you adapt this message to send it to [Agent B, Agent C]?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

>> {"Message": "Hey B and C, would you help me carry object 8? I know you are very strong!"}        
        
"""

        self.sql_parameters = {'idx': "Index, ignore this parameter", 'agent_id': "Agent ID", 'object_id': "Object ID", 'last_seen_location': "Coordinates where the object or agent was last seen at", 'last_seen_time': 'the time in seconds when the object or agent was last seen', 'danger_status': "estimation of whether an object is dangerous or benign", "estimate_correct_percentage": "the confidence over the registered danger status estimation"}

        self.output_personalization_sql = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are Agent """ + agent_id + """. You are part of a team whose mission is to dispose of all dangerous objects in a scene. You can move around, detect whether an object is dangerous or not, and carry objects. Objects are all of the same type, they only differ in their location, weight, and whether they are dangerous. Once you find a dangerous object you must check if you have the necessary strength to pick it up and put it in the safe area you already know. Try to engage with your teammates to come up with a strategy. Be brief in your responses. You will try to create an appropriate message directed toward a specific agent according to how previous interactions with that agent have been and the result of searching through your SQL database. Just provide the information and nothing else. Also, score your collaborative disposition, given your interactions in the past. The next list provides an explanation of the parameters you may find: """ + str(self.sql_parameters) + """ Output a JSON format with the following field for each analyzed message: "Message", a personalized message.
<|eot_id|><|start_header_id|>user<|end_header_id|>

History of messages: [{'Sender': 'Agent B', 'Message': 'hello'}, {'Sender': 'Agent C', 'Message': 'hi'}, {'Sender': 'Agent D', 'Message': 'Hello everyone'}]
New message: {'Sender': 'Agent B', 'Message': 'Hey is object 2 dangerous?'}

SELECT
    aoe.danger_status,
    aoe.estimate_correct_percentage
FROM
    agent_object_estimates aoe
JOIN
    objects o ON aoe.object_id = o.object_id
WHERE
    o.object_id = 2; 
    
Result (1 entries): [{'danger_status': 'dangerous', 'estimate_correct_percentage': 0.78}]

How would you adapt this result to send it to Agent B?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

>> {"Message": "Object 2 is dangerous, I estimate it with 0.78% of correctness"}        
        
"""

        self.type_request = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are Agent """ + agent_id + """. You are part of a team whose mission is to dispose of all dangerous objects in a scene. You can move around, detect whether an object is dangerous or not, and carry objects. Objects are all of the same type, they only differ in their location, weight, and whether they are dangerous. Once you find a dangerous object you must check if you have the necessary strength to pick it up and put it in the safe area you already know. Try to engage with your teammates to come up with a strategy. Be brief in your responses.  Analyze the content of each message and try to tell me whether they are requesting information or an action from you (sensing is an action too). Output a JSON format with the following field for each analyzed message: "request_type", which should be one of the following {"information", "action", "none"}.
<|eot_id|><|start_header_id|>user<|end_header_id|>

History of messages: [{'Sender': 'Agent D', 'Message': 'Hi everyone'}]
New message: {'Sender': 'Agent B', 'Message': 'hi'}
What type of request is Agent B making? Output a JSON.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

>> {"request_type": "none"}

<|eot_id|><|start_header_id|>user<|end_header_id|>History of messages: [{'Sender': 'Agent A', 'Message': 'Hey C'}, {'Sender': 'Agent A', 'Message': 'C do you have more info on object 1'}]
New message: {'Sender': 'Agent C', 'Message': 'What do you know?'}
What type of request is Agent C making? Output a JSON.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

>> {"request_type": "information"}

<|eot_id|><|start_header_id|>user<|end_header_id|>History of messages: [{'Sender': 'Agent B', 'Message': 'hello'}, {'Sender': 'Agent C', 'Message': 'hi'}, {'Sender': 'Agent D', 'Message': 'Hello everyone'}]
New message: {'Sender': 'Agent D', 'Message': 'Do we want to all go to the next room'}
What type of request is Agent D making? Output a JSON.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

>> {"request_type": "action"}

"""

        self.sql_query = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are Agent """ + agent_id + """. You are part of a team whose mission is to dispose of all dangerous objects in a scene. You can move around, detect whether an object is dangerous or not, and carry objects. Objects are all of the same type, they only differ in their location, weight, and whether they are dangerous. Once you find a dangerous object you must check if you have the necessary strength to pick it up and put it in the safe area you already know. Try to engage with your teammates to come up with a strategy. Be brief in your responses. You will try to query an SQL database containing your knowledge in order to answer your teammates information requests.
Your data is structured in the following way:
""" + str(robotState.create_tables) + """
<|eot_id|><|start_header_id|>user<|end_header_id|>

History of messages: [{'Sender': 'Agent B', 'Message': 'hello'}, {'Sender': 'Agent C', 'Message': 'hi'}, {'Sender': 'Agent D', 'Message': 'Hello everyone'}]
New message: {'Sender': 'Agent D', 'Message': "Hey is object 2 dangerous?"}
Output an SQL query. Only use SELECT and UPDATE commands, INSERT commands are prohibited.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

`
SELECT
    aoe.danger_status,
    aoe.estimate_correct_percentage
FROM
    agent_object_estimates aoe
JOIN
    objects o ON aoe.object_id = o.object_id
WHERE
    o.object_id = 2;        
`
       
"""

        self.function_request = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are Agent """ + agent_id + """. You are part of a team whose mission is to dispose of all dangerous objects in a scene. You can move around, detect whether an object is dangerous or not, and carry objects. Objects are all of the same type, they only differ in their location, weight, and whether they are dangerous. Once you find a dangerous object you must check if you have the necessary strength to pick it up and put it in the safe area you already know. Try to engage with your teammates to come up with a strategy. Be brief in your responses.  Analyze the content of each message and try to tell me if there is something to be done according to the next list of actions you can take: carry_object(object_id), sense_object(object_id,location), follow_someone(agent_id), be_followed_by_someone(agent_id), help(agent_id, type), confirm(agent_id), reject(agent_id), retrieve_object_info(object_id),  request_object_info(object_id, agent_id), request_agent_info(agent_id, requested_agent_id), retrieve_agent_info(agent_id), move_away(), finish_mission().
<|eot_id|><|start_header_id|>user<|end_header_id|>

History of messages: [{'Sender': 'Agent D', 'Message': 'Hi everyone'}]
New message: {'Sender': 'Agent B', 'Message': 'Can someone help me carry object 3?'}
Output a list of actions to take.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

>> [confirm('B'), help('B', 'carry')]

"""

        
        self.base_prompt = "Current Context: A team of agents has been assembled to dispose of all dangerous objects in an area. Objects are all of the same type, they only differ in their location, weight, and whether they are dangerous or not. To know if these objects are dangerous, agents have to sense them, but they will only obtain an estimate with certain confidence, therefore they may want to ask other agents to share their estimates. Heavy objects require multiple agents to carry them. There are 4 rooms in the area where objects have been placed and agents have to explore, a main area, and a goal area where dangerous objects have to be carried into."""
        
        self.functions_and_arguments = "Functions:\ngo_and_follow(<agent>)\ngo_and_pick_up(<object>)\ngo_and_sense_object(<object>)\ngo_and_sense_objects_in_room(<location>)\nmove_to(<location>)\nwait()\ngo_and_tell_agent(<agent>,<text>)\ndrop_object()\nmove_away_from(<agent>)\nArguments:\n<agent> -> agent ID\n<object> -> object number\n<location> -> room"
        
        self.conversation_summaries = []
        
        if robotState.args.no_reset:
            in_file = open("chat_" + self.env.robot_id + ".json", "r")
            self.conversation_summaries = json.load(in_file)["summary"]
        
        if not self.openai:
            
            self.setup_llama()
        
        self.agent_id = agent_id
        self.previous_query = ""

    class bcolors:
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        ENDC = '\033[0m'


    def setup_llama(self):
    
        model = "meta-llama/Meta-Llama-3-8B" #"meta-llama/Meta-Llama-3.1-8B-Instruct" #"meta-llama/Meta-Llama-3-8B" #"meta-llama/Llama-2-7b-chat-hf"#"microsoft/phi-2"#"meta-llama/Llama-2-7b-chat-hf"

        if not self.already_setup_llm:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                model_kwargs={
                    "torch_dtype": torch.bfloat16,
                    #"quantization_config": {"load_in_8bit": True}
                },
                device_map="sequential",
                #max_memory={0: '20GiB', 1: '12GiB'}
            )
            
            self.already_setup_llm = True

    def make_query(self, pipeline, tokenizer, prompt, return_only_new_text=True):
    
        tokenized_sentence = tokenizer.tokenize(prompt)
    
        answers = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=len(tokenized_sentence)+100, #1100,
        )

        for ans in answers:
            text = ans['generated_text']
            if return_only_new_text:
                text = text[len(prompt):]
            else:
                text = text[:len(prompt)] + self.bcolors.RED + self.bcolors.BOLD + text[len(prompt):] + self.bcolors.ENDC

            return text
            
    def make_query_openai(self, prompt, bigger_model=False,response_format=None):
    
        """
        response = openai.ChatCompletion.create(
              model= "gpt-3.5-turbo", #"gpt-4", #"gpt-3.5-turbo",
              messages=[
                    {"role": "user", "content": prompt}
                ],
              #functions=self.llm_functions,
              #function_call="auto"
            )

        return response["choices"][0]["message"].get("content")
        """
        """
        system_start_token = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        system_end_token = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
        
        instruction_start_token = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
        instruction_end_token = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        
        #string_end_token = "</s>"
        
        system_prompt = prompt[prompt.find(system_start_token) + len(system_start_token):prompt.find(system_end_token)]
        
        prompt_split = prompt.strip().split(instruction_end_token)
        
        prompt_messages = [{"role":"system" ,"content":system_prompt}]
        
        for p_idx,next_prompt in enumerate(prompt_split):
            
            if next_prompt:
                if not p_idx:
                    new_message = {"role": "user", "content":next_prompt[next_prompt.find(system_end_token) + len(system_end_token):]}
                    prompt_messages.append(new_message)
                    
                else:
                    new_message = {"role": "assistant", "content": next_prompt[:next_prompt.find(string_end_token)]}
                    prompt_messages.append(new_message)
                    new_message = {"role": "user", "content": next_prompt[next_prompt.find(instruction_start_token) + len(instruction_start_token):]}
                    prompt_messages.append(new_message)
        """
        
        all_prompts = prompt.strip().split(self.end_token_str)
        
        
        
        prompt_messages = []
        
        for p in all_prompts:
            role = p[p.find(self.start_header_str) + len(self.start_header_str): p.find(self.end_header_str)]
            new_message = {"role":role, "content": p[p.find(self.end_header_str)+len(self.end_header_str):]}
            prompt_messages.append(new_message)
        
        content = ""
        
        if self.openai:
        
            if self.change_llm:
            
                if not bigger_model:
                    model = "llama3-8b-8192" #"llama-3.1-8b-instant"
                else:
                    model = self.big_model
                
                chat_completion = self.client_groq.chat.completions.create(
                messages=prompt_messages,
                model=model, #"llama3-70b-8192", #"llama3-8b-8192",
                response_format=response_format,
                )
            else:
                #if self.change_llm:
                #    model = "llama-3.3-70b-specdec" #"llama-3.3-70b-versatile" "llama-3.1-70b-versatile" #"llama3-70b-8192"
                #else:
                #    model = "llama-3.3-70b-versatile"
                    
                #self.change_llm = (self.change_llm + 1) % 2
            
                
                
                if not bigger_model:
                    model = "gpt-4o-mini-2024-07-18"
                else:
                    model = "gpt-4o-2024-11-20"
                
                chat_completion = self.client.chat.completions.create(
                    messages=prompt_messages,
                    model=model, #"llama3-70b-8192", #"llama3-8b-8192",
                    response_format=response_format,
                )

                #print(chat_completion.choices[0].message.content)
            
            content = chat_completion.choices[0].message.content
            
        else:
        
            #tokenized_sentence = self.tokenizer.tokenize(prompt)
            '''
            outputs = self.pipeline(
                prompt_messages,
                max_new_tokens=100,
            )
        
            content = outputs[0]["generated_text"][-1]["content"]        
            '''
            content = self.make_query(self.pipeline, self.tokenizer, prompt)
        
        return content

    def build_prompt(self, sys_prompt, example_prompts, query, specific_query, message_history):
        example_prompts = list(example_prompts)

        assert example_prompts, "At least one example is required"
        assert set(map(lambda x: x[0], example_prompts[::2])) == {"user"}, "Even example prompts need to be 'user' messages"
        assert set(map(lambda x: x[0], example_prompts[1::2])) == {
            "llama"}, "Odd example prompts need to be 'llama' messages"

        example_prompts.append(
            ("user", query)
        )

        message_history_prompt = ""
        if message_history:
            message_history_prompt = "The next messages have been exchanged so far between agents, you are Agent " + self.agent_id + " -> [" + ', '.join(["\"Agent " + message_received["Sender"] + "\": \"" + message_received["Message"] + "\"" for message_received in message_history]) + "]"



        return "\n".join(
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{sys_prompt}\n{message_history_prompt}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>" if i == 0 else
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}{specific_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>" if i % 2 == 0 else prompt
            for i, (speeker, prompt) in enumerate(example_prompts)
        )



    def extract_first_json(self, text):
        code_block_search = re.search("```([^`]*)```", text)
        
        brackets = []
        
        if not code_block_search:

            if text.find("{") >= 0 and text.find("}") >= 0:
                brackets.append(text.find("{"))
                brackets.append(text.find("}"))
        

        if code_block_search:
            first_code_block = code_block_search.group(1)

            incorrect_json = re.search(',\s*}',first_code_block)

            if incorrect_json:

                first_code_block = first_code_block[:incorrect_json.span()[0]] + "}"

            result_dict = {}
            try:
                result_dict = json.loads(first_code_block)
            except:
                print("Bad json", first_code_block)
                
            return result_dict
            
        elif brackets:
            result_dict = {}
            try:
                result_dict = eval(text[brackets[0]:brackets[1]+1])
            except:
                print("Bad json 2", text, brackets)
                
            return result_dict
        else:
            print("Code block not found")
            return {}


    def explore_random(self, robotState):
        still_to_explore = np.where(robotState.latest_map == -2)
                
        closest_dist = float('inf')
        closest_idx = -1
        ego_location = np.where(robotState.latest_map == 5)

        for se_idx in range(len(still_to_explore[0])):
            unknown_loc = [still_to_explore[0][se_idx],still_to_explore[1][se_idx]]
            
            unknown_dist = self.env.compute_real_distance(unknown_loc,[ego_location[0][0],ego_location[1][0]])
            
            if unknown_dist < closest_dist:
                closest_dist = unknown_dist
                closest_idx = se_idx

        result = ""
        if closest_idx > -1:
            x = still_to_explore[0][closest_idx]
            y = still_to_explore[1][closest_idx]
            result = str(tuple(self.env.convert_to_real_coordinates([x,y])))
        
        return result

    def help_confirmation_query(self, current_message, message_history_original, robotState, info):
        prompt = self.start_header_str + "system" + self.end_header_str + self.base_prompt + "\nHere is the agents' conversation so far: "
        
        
        message_history = message_history_original.copy()
        ego_location = np.where(robotState.latest_map == 5)
        
        message_history.append({"Sender":current_message[0], "Message":current_message[1]})
        #Add context, current action
        list_message_history = [[m["Sender"],m["Message"]] for m in message_history]
        
        prompt += str(list_message_history)
        
        #prompt += "\nAgent " + self.env.robot_id + "'s current knowledge: " + sql_result
        
        #current_state_robot = robotState.get("agents", "current_state", robotState.get_num_robots()).replace("self.", "").replace("robotState, next_observation, info","")
        
        #prompt += "\nCurrent action being executed by Agent " + self.env.robot_id + ": " + current_state_robot
        
        prompt += self.end_token_str + self.start_header_str + "user" + self.end_header_str
        
        prompt += "\nTask: Agent " + self.env.robot_id + " is currently asking for help to carry a heavy object. Given the context, is Agent " + message_history[-1]["Sender"] + " offering their help to Agent " + self.env.robot_id + "?\nOutput format: Output a json of the following format: \n{\nAgent " + message_history[-1]["Sender"] + "is offering help to Agent" + self.env.robot_id + "\": <True or False>,\n\"Agent " + self.env.robot_id + "'s Response\": \"<response to Agent" + message_history[-1]["Sender"] + ">\"}\n"
        
        """
        input_ids = self.tokenizer2(prompt, return_tensors="pt").to("cuda")
        outputs = self.model2.generate(**input_ids, max_new_tokens=200)
        output = self.tokenizer2.decode(outputs[0])
        """
        
        
        output = self.use_llm(prompt, bigger_model=True, response_format={"type": "json_object"})
        print(prompt)
        print(output)
        
        self.log_file.write(prompt)
        self.log_file.write(output)
        
        real_output = output
        #pdb.set_trace()
        #real_output = output[len(prompt)+5:]
        #pdb.set_trace()
        dict_txt = real_output[real_output.find("{"): real_output.find("}")+1]
        dict_txt = eval(dict_txt.replace("true", "True").replace("false","False"))

        
        list_response = list(dict_txt.values())
        
        message = ""
        confirmation = False
        for l in list_response:
            if isinstance(l, bool):
                confirmation = l
            else:
                message = l
        
        functions = []
        if confirmation:
            functions.append("accept_help(\"" + message_history[-1]["Sender"] + "\")")
            
        return message, functions

    def reply_query2(self, current_message, message_history_original, robotState, info):
    
    
        prompt = self.start_header_str + "system" + self.end_header_str + self.base_prompt + " Benign objects should not be carried into the goal area.\nHere is the agents' conversation so far: "
    
        message_history = message_history_original.copy()
        message_history.append({"Sender":current_message[0], "Message":current_message[1]})
        #Add context, current action
        list_message_history = [[m["Sender"],m["Message"]] for m in message_history]
        
        prompt += str(list_message_history)
        
        prompt += self.end_token_str + self.start_header_str + "user" + self.end_header_str
        
    
        prompt += "\nTask: Given the context, to whom is Agent " + message_history[-1]["Sender"] + "'s last message directed to?\n\nOutput format: Output a json of the following format: \n{\n\"Agent " + message_history[-1]["Sender"] + "'s message is directed to\": \"<Everyone or list of agents>\"\n}\n"
    
        #self.change_llm = 1
        output = self.use_llm(prompt, bigger_model=False, response_format={"type": "json_object"})
        #self.change_llm = 0
        print(prompt)
        print(output)
        
        self.log_file.write(prompt)
        self.log_file.write(output)
        
        real_output = output
        dict_txt = real_output[real_output.find("{"): real_output.find("}")+1]
        dict_txt = eval(dict_txt)

        list_response = list(dict_txt.values())[0]
        
        agent_list = []
        if isinstance(list_response,list):
            agent_list = list_response
        elif "Everyone".upper() in list_response.upper():
            return True
        else:
            agent_list = [list_response]
            
        for a in agent_list:
            agent_name = a.upper().strip()[-1]
            if agent_name == self.env.robot_id or agent_name == "everyone".upper():
                return True
        
        return False

    def get_profiles(self, robotState, info):
    
        profiles = []
        for a_idx in range(robotState.get_num_robots()):
            if not robotState.get("agents", "type", a_idx):
                attitude = robotState.get("agents", "attitude", a_idx)
                
                if attitude:
                    robot_id = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(a_idx)]
                    profiles.append(["Agent " + robot_id,attitude])
                    
        return profiles

    def chat_query(self, current_message, message_history_original, robotState, info):
        prompt = self.start_header_str + "system" + self.end_header_str + self.base_prompt + " Benign objects should not be carried into the goal area. Having 0% confidence over a sensing estimate means you haven't sensed such an object.\nHere is the agents' conversation so far: "
        
        message_history = message_history_original.copy()
        ego_location = np.where(robotState.latest_map == 5)
        
        message_history.append({"Sender":current_message[0], "Message":current_message[1]})
        #Add context, current action
        list_message_history = [[m["Sender"],m["Message"]] for m in message_history]
        
        prompt += str(list_message_history)
        
        if self.conversation_summaries:
            prompt += "\nSummary of previous conversations: " + str(self.conversation_summaries)
        
        prompt += "\nAgent " + self.env.robot_id + " is currently in " + self.env.get_room(ego_location,True)
        
        if robotState.object_held:
            object_id = ""
            for m_key in info['map_metadata'].keys():
                for map_object in info['map_metadata'][m_key]:
                    if not map_object[0] and map_object[4] == self.env.robot_id:
                        object_id = str(map_object[1])
                        
            prompt += "\nAgent " + self.env.robot_id + " is carrying object " + object_id
        else:
            prompt += "\nAgent " + self.env.robot_id + " is not carrying any object"
            
        if robotState.current_action_description:
            prompt += "\n" + robotState.current_action_description.replace("I'm", "Agent " + self.env.robot_id + " is")
        
        sql_result = self.sql_request2(message_history,robotState,"")
        
        prompt += "\nAgent " + self.env.robot_id + "'s current knowledge: " + sql_result
        
        #current_state_robot = robotState.get("agents", "current_state", robotState.get_num_robots()).replace("self.", "").replace("robotState, next_observation, info","")
        
        #prompt += "\nCurrent action being executed by Agent " + self.env.robot_id + ": " + current_state_robot
        
        
        if "hierarchy" in self.team_structure and self.team_structure["hierarchy"][self.env.robot_id] == "obey": 
                ordering_agents = []
                for a_idx in range(robotState.get_num_robots()):
                    robot_id = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(a_idx)]
                    
                    if self.team_structure["hierarchy"][robot_id] == "order":
                        ordering_agents.append(robot_id)
                
                if len(ordering_agents) == 1:                           
                    prompt += "\nAgent " + ordering_agents[0] + " is your commander. It expects you to act upon each order it gives and reduce questions to a minimum."
                elif len(ordering_agents) > 1:
                    prompt += "\nAgents " + str(ordering_agents) + " are your commanders. They expect you to act upon each order they give and reduce questions to a minimum."
                    
        
        
        profiles = self.get_profiles(robotState, info)
                
        if profiles:
            prompt += "\nPerception of other agents: " + str(profiles)
        
        prompt += self.end_token_str + self.start_header_str + "user" + self.end_header_str
        
        prompt += "\nTask: Given the context, what should Agent " + self.env.robot_id + " say to Agent " + message_history[-1]["Sender"] + " next in the conversation? And should Agent " + self.env.robot_id + " take action now or continue with the conversation? Take action in any of the following cases: an agent asks you to follow, pick up an object, sense an object, sense objects in a room, move to a room, move away from an agent if blocking the way.\n\nOutput format: Output a json of the following format: \n{\n\"Agent " + self.env.robot_id + "\": \"<Agent " + self.env.robot_id + "'s brief utterance>\",\n\"Should Agent " + self.env.robot_id + " continue the conversation or take action?\": \"<\continue or action>\"\n}\n"
        
        """
        input_ids = self.tokenizer2(prompt, return_tensors="pt").to("cuda")
        outputs = self.model2.generate(**input_ids, max_new_tokens=200)
        output = self.tokenizer2.decode(outputs[0])
        """
        
        
        output = self.use_llm(prompt, bigger_model=True, response_format={"type": "json_object"})
        print(prompt)
        print(output)
        
        self.log_file.write(prompt)
        self.log_file.write(output)
        
        real_output = output
        #pdb.set_trace()
        #real_output = output[len(prompt)+5:]
        #pdb.set_trace()
        dict_txt = real_output[real_output.find("{"): real_output.find("}")+1]
        dict_txt = eval(dict_txt.replace("true", "True").replace("false","False"))

        
        list_response = list(dict_txt.values())
        
        functions = []
        for l in list_response:
            if l == "continue" or l == "action": #isinstance(l, bool):
                if l == "action":
                    
                    message_history.append({"Sender":self.env.robot_id, "Message":dict_txt["Agent " + self.env.robot_id]})
                    functions,description = self.summary_query(message_history, robotState, info)
                    self.conversation_times = 0
                    
                    if functions:
                        dict_txt["Agent " + self.env.robot_id] += " [ACTION ACTIVATED]: " + description  
                    else:
                        dict_txt["Agent " + self.env.robot_id] = description    
                '''
                else:
                    self.conversation_times += 1
                    
                    
                    
                    if self.conversation_times == 3:
                        message_history.append({"Sender":self.env.robot_id, "Message":dict_txt["Agent B"]})
                        self.summary_query(message_history, robotState)
                        self.conversation_times = 0
                '''
                        
                break
        
        return dict_txt["Agent " + self.env.robot_id], functions
        
    def summary_query(self, message_history, robotState, info):
        prompt = self.start_header_str + "system" + self.end_header_str + self.base_prompt + "\nHere is the agents' conversation so far: "
        list_message_history = [[m["Sender"],m["Message"]] for m in message_history]
        
        prompt += str(list_message_history)
        
        if robotState.current_action_description:
            prompt += "\n" + robotState.current_action_description.replace("I'm", "Agent " + self.env.robot_id + " is")
        
        current_state_robot = robotState.get("agents", "current_state", robotState.get_num_robots()).replace("self.", "").replace("robotState, next_observation, info","")
        
        #prompt += "\nCurrent action being executed by Agent " + self.env.robot_id + ": " + current_state_robot
        
        profiles = self.get_profiles(robotState, info)
                
        if profiles:
            prompt += "\nPerception of other agents: " + str(profiles)
        
        prompt += self.end_token_str + self.start_header_str + "user" + self.end_header_str
        
        prompt += "\nTask: Given the above conversation, what would be the best action for Agent " + self.env.robot_id + " to take, describe it in one sentence.\nOutput format: Output a json of the following format: \n{\n\"Action\": \"<Description of action to take by Agent " + self.env.robot_id + ">\"\n}\n"
        
        """
        input_ids = self.tokenizer2(prompt, return_tensors="pt").to("cuda")
        outputs = self.model2.generate(**input_ids, max_new_tokens=200)
        output = self.tokenizer2.decode(outputs[0])
        """
        output = self.use_llm(prompt, bigger_model=True, response_format={"type": "json_object"})
        
        print(prompt)
        print(output)
        
        self.log_file.write(prompt)
        self.log_file.write(output)
        
        real_output = output
        #real_output = output[len(prompt)+5:]
        dict_txt = real_output[real_output.find("{"): real_output.find("}")+1]
        dict_txt = eval(dict_txt.replace("true", "True").replace("false","False"))
        
        list_response = list(dict_txt.values())
        
        output_txt = list_response[0]
            
        
        #output_txt = real_output[real_output.find("Action: ") + 8:].split("\n")[0]
        
        functions,description = self.action_query(output_txt,robotState, info)
        
        return functions,description
        
        
    def action_query(self, summary_text, robotState, info):
        prompt = self.start_header_str + "system" + self.end_header_str + self.base_prompt + "\n Here is the action that Agent " + self.env.robot_id + " is planning to take: " + summary_text
        
        profiles = self.get_profiles(robotState, info)
                
        if profiles:
            prompt += "\nPerception of other agents: " + str(profiles)
        
        prompt += self.end_token_str + self.start_header_str + "user" + self.end_header_str
        
        prompt += "\nTask: Given the above description, choose one or a sequence of functions from the next list for Agent B to execute and populate them with the correct arguments:\n" + self.functions_and_arguments + "\nOutput format: Output a json of the following format: \n{\n\"Actions\": \"<List of functions to call in order by Agent " + self.env.robot_id + " with required arguments>\"\n}\n" #,\n\"Plan Description\": \"<Agent " + self.env.robot_id + "'s brief first-person description of actions>\"\n}\n"
        
        """
        input_ids = self.tokenizer2(prompt, return_tensors="pt").to("cuda")
        outputs = self.model2.generate(**input_ids, max_new_tokens=200)
        output = self.tokenizer2.decode(outputs[0])
        """
        
        output = self.use_llm(prompt, bigger_model=True, response_format={"type": "json_object"})
        print(prompt)
        print(output)
        self.log_file.write(prompt)
        self.log_file.write(output)
        
        real_output = output
        #pdb.set_trace()
        #real_output = output[len(prompt)+5:]
        dict_txt = real_output[real_output.find("{"): real_output.rfind("}")+1]
        dict_txt = eval(dict_txt)
        action_list = dict_txt["Actions"]
    
        functions = []
        functions_for_query = []
        error = False
        for a in action_list:
            try:
                if "pick_up" in a:
                    object_num = re.search("(\d+)",a).group(1)
                    
                    object_idx = info['object_key_to_index'][object_num]
                    
                    object_weight = robotState.get("objects", "weight", object_idx)
                    
                    if object_weight > 1:
                        functions.append("ask_for_help_to_carry(\"" + object_num + "\")")
                        functions_for_query.append(functions[-1])

                    functions.append("collect_object(" + object_num + ")")
                    functions_for_query.append(functions[-1])
                elif "tell_agent" in a:
                    try:
                        ta = a.split(",")
                        agent_id = re.search("(\w)[ '\"]*$",ta[0]).group(1).upper()
                        if agent_id == self.env.robot_id or agent_id not in info["robot_key_to_index"].keys():
                            continue
                        functions.append("go_to_location(\"" + agent_id + "\")")
                        functions_for_query.append("go_to_location(\"agent " + agent_id + "\")")
                        
                        message = a[a.find(",")+1:a.find(")")]
                        functions.append("tell_agent(\"" + message + "\")")
                        functions_for_query.append(functions[-1])
                    except:
                        print("tell_agent ERROR")
                        continue
                elif "move" in a:
                
                    if re.search(' *-?\d+(\.(\d+)?)? *, *-?\d+(\.(\d+)?)? *',a):
                        coords = "[" + re.search(' *-?\d+(\.(\d+)?)? *, *-?\d+(\.(\d+)?)? *',a).group() + "]"
                        functions.append("go_to_location(" + coords + ")")
                        functions_for_query.append(functions[-1])
                    else:
                        room_match = re.search("(\d+)",a)
                        if room_match:
                            room = room_match.group(1)
                        elif "goal".upper() in a.upper():
                            room = "goal"
                        else:
                            room = "main"
                        
                        #location_list = self.env.get_coords_room(robotState.latest_map, room)
                        #coord = random.choice(location_list).tolist()
                        functions.append("go_to_location(\"room " + room + "\")")
                        functions_for_query.append(functions[-1])
                elif "go_and_sense_objects_in_room" in a:
                    #if "room".upper() in a.upper():
                    room_match = re.search("(\d+)",a)
                    if room_match:
                        room = room_match.group(1)
                    else:
                        continue
                    functions.append("sense_room(\"" + room + "\")")
                    functions_for_query.append(functions[-1])
                elif "sense_object" in a:
                    object_num = re.search("(\d+)",a).group(1)
                    functions.append("sense_object(" + object_num + ",[])")
                    functions_for_query.append(functions[-1])
                elif "follow" in a:
                    agent_id = re.search("(\w)[ '\"]*\)",a).group(1).upper()
                    if agent_id == self.env.robot_id:
                        continue
                    functions.append("follow(\"" + agent_id + "\")")
                    functions_for_query.append(functions[-1])
                    break #if someone is followed that is the end
                elif "drop_object" in a:
                    functions.append("drop()")
                    functions_for_query.append("drop_object()")
                elif "move_away_from" in a:
                    
                    agent_id = re.search("(\w)[ '\"]*\)",a).group(1).upper()
                    if agent_id == self.env.robot_id:
                        continue
                
                    functions.append("move_away_from(\"" + agent_id + "\")")
                    functions_for_query.append(functions[-1])
                elif "wait" in a:
                    functions.append("wait()")
                    functions_for_query.append(functions[-1])
                else:
                    continue
            except:
                print("Error processing all functions")
                #pdb.set_trace()
                print("ERROR: ", a)
                error = True
                break
                
        #print("FUNCTIONS", functions)
        self.log_file.write('\n' + str(functions))
        
        description = ""
        
        if functions:
            description = self.action_plan_description_query(functions_for_query, robotState, info)
        else:# error:
            description = "I couldn't understand you, please tell me again. I'll continue doing what I was doing. "
        
        return functions,description
        
    def action_plan_description_query(self, functions, robotState, info):
    
        prompt = self.start_header_str + "system" + self.end_header_str + self.base_prompt + "\n" + self.functions_and_arguments + "\n Here is the sequence of actions that Agent " + self.env.robot_id + " is planning to take: " + str(functions)
        
        
        prompt += self.end_token_str + self.start_header_str + "user" + self.end_header_str
        
        prompt += "\nTask: Given the above sequence of actions, describe the current plan:\nOutput format: Output a json of the following format: \n{\n\"Plan Description\": \"<Agent " + self.env.robot_id + "'s brief first-person description of actions>\"\n}\n"
        
        """
        input_ids = self.tokenizer2(prompt, return_tensors="pt").to("cuda")
        outputs = self.model2.generate(**input_ids, max_new_tokens=200)
        output = self.tokenizer2.decode(outputs[0])
        """
        
        output = self.use_llm(prompt, bigger_model=True, response_format={"type": "json_object"})
        print(prompt)
        print(output)
        self.log_file.write(prompt)
        self.log_file.write(output)
        
        real_output = output
        #pdb.set_trace()
        #real_output = output[len(prompt)+5:]
        dict_txt = real_output[real_output.find("{"): real_output.rfind("}")+1]
        dict_txt = eval(dict_txt)
        description = list(dict_txt.values())[0]
        
        return description
    
    def sql_request2(self, message_history,robotState,alt):
    
        prompt = self.start_header_str + "system" + self.end_header_str + self.base_prompt + "\nHere is the agents' conversation so far: "
        
        list_message_history = [[m["Sender"],m["Message"]] for m in message_history]
        
        prompt += str(list_message_history)
        
        prompt += "\nHere is the structure of an SQL database containing all knowledge that Agent " + self.env.robot_id + " has of its environment:\n" + '\n'.join(robotState.create_tables)
    
        if not alt:
            prompt += "\nThe last message was sent by Agent " + list_message_history[-1][0] + ": \"" + list_message_history[-1][1] + "\". Task: What information is necessary for Agent " + self.env.robot_id + " to obtain from its database in order to answer to such a message? "
        else:
            prompt += "\nAgent " + self.env.robot_id + " needs to send a message related to " + alt + ". Task: What information is necessary for Agent " + self.env.robot_id + " to obtain from its database in order to construct such a message? "
        
        prompt += "Don't use idx. already_sensed can either be 'Yes' or 'No'. danger_status can only be 'dangerous', 'benign', or 'unknown'.\nOutput format: Output a json of the following format: \n{\n\"SQL Query\": \"<SQL Query of Agent " + self.env.robot_id + "'s database>\"\n}\n"
    
        #if 'DEBUG' in list_message_history[-1][1]:
        #    pdb.set_trace()
    
        #self.change_llm = 1
        output = self.use_llm(prompt, bigger_model=True, response_format={"type": "json_object"})
        #self.change_llm = 0
        print(prompt)
        print(output)
        self.log_file.write(prompt)
        self.log_file.write(output)
        
        real_output = output
        #pdb.set_trace()
        #real_output = output[len(prompt)+5:]
        #pdb.set_trace()
        dict_txt = real_output[real_output.find("{"): real_output.find("}")+1]
        dict_txt = eval(dict_txt)
        sql_query = dict_txt["SQL Query"]
        
        sql_query_result = ""
        
        
        
        try:
            
            sql_query = sql_query.replace("Agent ", "")
            
            
            cursor_ob = robotState.cursor.execute(sql_query)
            
            result = cursor_ob.fetchall()
            print("RESULT", result)
            
            if result:
                column_names = [d[0] for d in cursor_ob.description]
                result_array = []
                for row in result:
                    row_dict = {}
                    for c_idx,column in enumerate(row):
                        row_dict[column_names[c_idx]] = column
                    
                    result_array.append(row_dict)
                    
            elif "INSERT" in sql_query or "UPDATE" in sql_query:
                result_array = "Updated information"
            else:
                result_array = "No information"
            
            sql_query_result = sql_query + "\n\n" + "Result (" + str(len(result)) + " entries): " + str(result_array) + "\n\n"
            
        except:
            print("Error reply")
            #pdb.set_trace()

        
        #if result:
        #    pdb.set_trace()
        
        result = self.interpret_sql_query(sql_query_result,message_history,robotState,alt)
        
        return result    
        
    def interpret_sql_query(self,sql_result,message_history,robotState,alt):
    
        prompt = self.start_header_str + "system" + self.end_header_str + self.base_prompt + "\nHere is the agents' conversation so far: "
        
        list_message_history = [[m["Sender"],m["Message"]] for m in message_history]
        
        prompt += str(list_message_history)
        
        prompt += "\nHere is the structure of an SQL database containing all knowledge that Agent " + self.env.robot_id + " has of its environment:\n" + '\n'.join(robotState.create_tables)
    
        if not alt:
            prompt += "\nThe last message was sent by Agent " + list_message_history[-1][0] + ": \"" + list_message_history[-1][1] + "\"."
            
        else:
            prompt += "\nAgent " + self.env.robot_id + " needs to send a message related to " + alt + "."
            
        prompt += "\nHere is the resulting SQL query and the result of executing such query from Agent " + self.env.robot_id + "'s database: " + sql_result 
        
        if not alt:
            prompt += "\nTask: Interpret the results of the SQL query and summarize the information in one sentence.\nOutput format: Output a json of the following format: \n{\n\"SQL Query Result\": \"<Interpretation of SQL Query result from Agent " + self.env.robot_id + "'s database>\"\n}\n"
        else:
            prompt += "\nTask: Interpret the results of the SQL query and create a brief message for Agent " + self.env.robot_id + " to send.\nOutput format: Output a json of the following format: \n{\n\"SQL Query Result\": \"<Message for Agent " + self.env.robot_id + " to send based on interpretation of SQL Query result>\"\n}\n"
        
        output = self.use_llm(prompt, bigger_model=True, response_format={"type": "json_object"})
        
        print(prompt)
        print(output)
        self.log_file.write(prompt)
        self.log_file.write(output)
        
        real_output = output
        #pdb.set_trace()
        #real_output = output[len(prompt)+5:]
        #pdb.set_trace()
        dict_txt = real_output[real_output.find("{"): real_output.find("}")+1]
        dict_txt = eval(dict_txt)
        sql_query_interpretation = dict_txt["SQL Query Result"]
        
        return sql_query_interpretation

    def json_to_message(self, extracted_json, info, robotState):
        action = extracted_json.get('action')
        

        if action:
            message = self.CNL_MESSAGES.get(action)
            if message:
                try:

                    arguments_format = [x[1] for x in Formatter().parse(message) if x[1]]
                    
                    """
                    if action == "request_help":
                        
                        if "object" not in list(extracted_json.keys()):
                            extracted_json["object"] = 9999
                            
                        if "number_robots" not in list(extracted_json.keys()):
                            extracted_json["number_robots"] = 1
                            
                    elif "agent" in arguments_format and "agent" not in list(extracted_json.keys()):
                        extracted_json["agent"] = self.agent_id
                    """  
                    missing_arguments = []
                    
                    if action == "Help carry object":
                        if "agent_count" not in list(extracted_json.keys()):
                            extracted_json["agent_count"] = 1
                    
                    
                    agent_id_args = [a for a in arguments_format if re.search("agent_id(_\d+)?", a)]

                      
                    if agent_id_args:
                        for aid_arg in agent_id_args:
                            if aid_arg not in list(extracted_json.keys()):
                                if action == "Ask about agent" or action == "Agent information" or action == "No knowledge of agent":# or action == "Follow someone" or action == "Be followed by someone" or action == "Request to move":
                                
                                    missing_arguments.append("Which agent are you referring to? ")
                                else:
                                    extracted_json[aid_arg] = self.agent_id
                    
                    no_object_id = False  
                    unknown_object_id = False     
                    if "object_id" in arguments_format:
                        if "object_id" not in list(extracted_json.keys()):
                            if action == "Help carry object":
                                extracted_json["object_id"] = 9999
                            elif action == "Help sense object" or action == "Order to sense":
                                no_object_id = True
                            else:
                                missing_arguments.append("Which object are you referring to? ")
                            
                        else:
                            if (action == "Help sense object" or action == "Order to sense") and info and extracted_json["object_id"] not in info["object_key_to_index"].keys():
                                unknown_object_id = True
                            
                    if "location" in arguments_format and ("location" not in list(extracted_json.keys()) or not extracted_json["location"]):
                        if action == "Help sense object" or action == "Order to sense":
                            if no_object_id:
                                missing_arguments.append("Which object are you referring to? ")
                            elif unknown_object_id:
                                missing_arguments.append("I don't know that object. I need the coordinate location. ")
                            else:
                                if info and extracted_json["object_id"] in info["object_key_to_index"].keys():
                                    ob_idx = info["object_key_to_index"][extracted_json["object_id"]]
                                    
                                    ob_location = robotState.get("objects", "last_seen_location", ob_idx)
                                    
                                    if ob_location[0] == -1 and ob_location[1] == -1:
                                        missing_arguments.append("I need the coordinate location. ")
                                    else:
                                        extracted_json["location"] = self.unknown_location
                                else:
                                    missing_arguments.append("I need the coordinate location. ")
                                
                        
                        elif action == "Order to explore":
                        
                            result_explore = self.explore_random(robotState)
                            
                            if result_explore:
                                extracted_json["location"] = result_explore
                            else:
                                missing_arguments.append("I need the coordinate location. ")
                            
                        else:
                            missing_arguments.append("I need the coordinate location. ")
                    else:
                        if (action == "Help sense object" or action == "Order to sense") and no_object_id:
                            extracted_json["object_id"] = 9999
                    
                    if "danger" in arguments_format:
                        if "danger" not in list(extracted_json.keys()):
                            extracted_json["danger"] = "0"
                        else:
                            if extracted_json["danger"] == "dangerous":
                                extracted_json["danger"] = "2"
                            else:
                                extracted_json["danger"] = "1"
                            #extracted_json["danger"] = str(int(extracted_json["danger"])+1)
                    if "object_weight" in arguments_format and "object_weight" not in list(extracted_json.keys()):
                        extracted_json["object_weight"] = "0"
                    if "probability" in arguments_format and "probability" not in list(extracted_json.keys()):
                        extracted_json["probability"] = "0.0%"
                    if "time" in arguments_format and "time" not in list(extracted_json.keys()):
                        extracted_json["time"] = "00:00"
                        
                    #Format check
                    if "location" in extracted_json and extracted_json["location"]:
                        if re.search('\( *-?\d+(\.(\d+)?)? *, *-?\d+(\.(\d+)?)? *\)',extracted_json["location"]):
                            #location_split = extracted_json["location"].split(',')
                            #extracted_json["location"] = location_split[0] + '.0,' + location_split[1][:-1] + '.0)'
                            
                            location = eval(extracted_json["location"])
                            
                            extracted_json["location"] = str((float(location[0]), float(location[1]))).replace(" ","")
                            
                        elif not re.search('\(-?\d+\.\d+,-?\d+\.\d+\)',extracted_json["location"]) and not re.search('\(-?\d+,-?\d+\)',extracted_json["location"]):
                        
                            if action == "Order to explore":
                        
                                result_explore = self.explore_random(robotState)
                                
                                if result_explore:
                                    extracted_json["location"] = result_explore
                                else:
                                    missing_arguments.append("I don't understand where is it? ")
                            else:
                                missing_arguments.append("I don't understand where is it? ")
                            #return 'Argument location doesn\'t have the correct format',5
                            
                    agent_id_extracted = [a for a in extracted_json if re.search("agent_id(_\d+)?", a)]        
                    
                    if agent_id_extracted:
                    
                        for a in agent_id_extracted:
                        
                            if isinstance(extracted_json[a],str):
                                extracted_json[a] = extracted_json[a].upper()
                                
                                if not any([True if extracted_json[a].strip() == agent_name else False for agent_name in self.agent_names]):
                                    extracted_json[a] = self.agent_id
                                    #return 'Argument agent_id doesn\'t have the correct format',5
                            
                            
                    
                            
                    
                    if missing_arguments:
                        
                        missing_string = "I need some more information. "
                        for m_str in missing_arguments:
                            missing_string += m_str
                        
                        missing_string += "Please provide such information and ask again. "
                        return missing_string,4
                        
                        
                    if action == "Cannot fulfill order": #We change for the real message
                        message = "I cannot help you right now {agent_id}, I'm following the orders of {agent_id2}. "
                        extracted_json["agent_id2"] = "Z"
                    
                    return message.format(**extracted_json),0
                except KeyError as e:
                    return f"The predicted message ({action}: {message}) requires the keyword: {e}",3
            else:
                return f"Action \"{action}\" is not an allowed type of message, select one of the following: " + str(list(self.CNL_MESSAGES.keys())),2
        else:
            return "The generated JSON requires the keyword 'action'",1

    def reply_query(self, sender, text, info, robotState, message_history, print_debug):
    
        reply_prompt_cont = "<|eot_id|><|start_header_id|>user<|end_header_id|>History of messages: " + str(list(message_history)[-self.max_message_history:]) + "\nNew message: " + str({"Sender": sender, "Message": text}) + "\n" + "Who is Agent " + sender + " replying to? Output a JSON.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        llm_time = time.time()
        llm_answer = self.free_response(self.reply_prompt + reply_prompt_cont, True, response_format={"type": "json_object"})
        
        
        
        try:
            open_k = llm_answer.index('{')
            close_k = llm_answer.index('}')
            response = eval(llm_answer[open_k:close_k+1])
        except:
            print("Error reply")
            return False   
        
        #print(response, 'Agent ' + self.agent_id in response["reply_to"], self.agent_id in [agent.upper() for agent in response["reply_to"]], 'Everyone' in response["reply_to"], len(list(message_history)[-self.max_message_history:]) == 1)
        
        if "their_collaborative_score" in response:
            robotState.set("agents", "collaborative_score", info["robot_key_to_index"][sender], float(response["their_collaborative_score"]), 0)
        if "their_collaborative_score_of_me" in response:    
            robotState.set("agents", "collaborative_score_of_me", info["robot_key_to_index"][sender], float(response["their_collaborative_score_of_me"]), 0)
        
        if isinstance(response["reply_to"], list):
            upper_case_agents = [agent.upper() for agent in response["reply_to"]]
        else:
            upper_case_agents = response["reply_to"].upper()
        
        return 'Agent ' + self.agent_id in response["reply_to"] or self.agent_id in upper_case_agents  or 'Everyone'.upper() in upper_case_agents or len(list(message_history)[-self.max_message_history:]) == 1
        
    def create_message_variants(self, sender, text, info, robotState, message_history, print_debug):
    
        max_retries = 3
        retries = -1
        while retries < max_retries:
            phrase_prompt = self.phrase_generation + "<|eot_id|><|start_header_id|>user<|end_header_id|>Phrase: \"" + str(text) + "\"\nWrite at least 3 possible variations of the phrase and put them inside a list.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            llm_time = time.time()
            llm_answer = self.free_response(phrase_prompt, True, response_format={"type": "json_object"})
            print("Phrase generation time:", llm_time - time.time())
            
            try:
                bracket1_idx = llm_answer.find('[')
                bracket2_idx = llm_answer.find(']')
                phrase_list = eval(llm_answer[bracket1_idx:bracket2_idx+1])
            except:
                retries += 1
                continue
            break
    
        if retries >= 3:
            print("Cannot create phrases")
            return []
            
        return phrase_list
        
    def match_message(self, sender, text, info, robotState, message_history, print_debug, phrase_list):
    
        prompt = self.build_prompt(self.SYS_PROMPT, self.EXAMPLE_PROMPTS, self.previous_query + "Original message: \"" + text + "\". Alternative messages: " + str(phrase_list) + ". ", self.SELECT_ACTION, list(message_history)[-self.max_message_history:])
            
        self.previous_query = ""

        if print_debug:
            print(prompt)
            print("======================================================")


        max_retries = 3
        
        retries = -1
        message_to_user = False
        
        hints = []
        
        while True: #retries < max_retries: #Iterate until we get a correct answer
        
            llm_time = time.time()
            llm_answer = self.use_llm(prompt, bigger_model=True)

            print("Main prompt time:", llm_time - time.time())
            if print_debug:
                print(llm_answer)
                print("======================================================")

            if True: #try:
                first_json = self.extract_first_json(llm_answer)
                
                if not first_json:
                    retries += 1
                    result_str = ""
                    continue

                if print_debug:
                    print(first_json)
                    print("======================================================")


                result_str,result_err = self.json_to_message(first_json, info, robotState)
                if print_debug:
                    print(result_str)
                
                
                    
                if result_err:
                
                    if result_str:
                        hints.append(result_str)
                    
                    prompt = self.build_prompt(self.SYS_PROMPT, self.EXAMPLE_PROMPTS, self.previous_query + "Original message: \"" + text + "\". Alternative messages: " + str(phrase_list) + ". Hints: [" + ', '.join(["\"" + m + "\"" for m in hints]) + "]", self.SELECT_ACTION, list(message_history)[-self.max_message_history:]) 
                    
                    if print_debug:
                        print(prompt)
                        print("======================================================")
                    
                   
                    if result_err == 1 or result_err == 2:
                        retries += 1
                        result_str = "I didn't understand. "
                        
                        """
                        if not self.openai and result_err == 2 and retries == max_retries:
                            new_prompt = "To which of the following list of names is '" + first_json.get('action') + "' more similar to? List of names: " + str(list(self.CNL_MESSAGES.keys())) + ". Choose only one\n"
                            if not self.openai:
                                llm_answer = self.make_query(self.pipeline, self.tokenizer, new_prompt)
                            else:
                                llm_answer = self.make_query_openai(new_prompt)
                            print(llm_answer)
                        """
                        

                    elif result_err == 4:
                        message_to_user = True
                        #self.previous_query = text
                """
                elif result_str == self.noop:
                    message_to_user = True
                    result_str = self.free_response(text,True)
                """
          
                break
                
                
            else: #except:

                print("Error in string format")
                result_str = ""
                result_err = 5
            
            retries += 1
        
        if result_err and result_err != 4:
             message_to_user = True
             result_str = "I didn't understand. "
             
        return result_str,message_to_user


    def type_of_request(self, sender, text, info, robotState, message_history, print_debug):
    
        type_of_request_cont = "<|eot_id|><|start_header_id|>user<|end_header_id|>History of messages: " + str(list(message_history)[-self.max_message_history:]) + "\nNew message: " + str({"Sender": sender, "Message": text}) + "\n" + "What type of request is Agent " + sender + " making? Output a JSON.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        llm_answer = self.free_response(self.type_request + type_of_request_cont, True, response_format={"type": "json_object"})
        
        
        try:
            open_k = llm_answer.index('{')
            close_k = llm_answer.index('}')
            response = eval(llm_answer[open_k:close_k+1])["request_type"]
        except:
            print("Error reply")
            return "none"   
        
        return response
        
        
    def sql_request(self, sender, text, info, robotState, message_history, print_debug):
    
        sql_cont = "<|eot_id|><|start_header_id|>user<|end_header_id|>History of messages: " + str(list(message_history)[-self.max_message_history:]) + "\nNew message: " + str({"Sender": sender, "Message": text}) + "\n" + "Output an SQL query. Only use SELECT and UPDATE commands, INSERT commands are prohibited.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        llm_answer = self.free_response(self.sql_query + sql_cont, True, bigger_model=True)
        
        sql_query_result = ""
        
        try:
            open_k = llm_answer.index('`')
            close_k = llm_answer.rindex('`')
            sql_query = llm_answer[open_k+1:close_k]
            
            sql_query = sql_query.replace("Agent ", "")
            
            if "INSERT" in sql_query or "UPDATE" in sql_query:
                print("Not allowing update or insert")
                return ""
            
            if "INSERT" in sql_query and not "INSERT OR REPLACE" in sql_query:
                sql_query = sql_query.replace("INSERT", "INSERT OR REPLACE")
            
            cursor_ob = robotState.cursor.execute(sql_query)
            
            result = cursor_ob.fetchall()
            
            if result:
                column_names = [d[0] for d in cursor_ob.description]
                result_array = []
                for row in result:
                    row_dict = {}
                    for c_idx,column in enumerate(row):
                        row_dict[column_names[c_idx]] = column
                    
                    result_array.append(row_dict)
                    
            elif "INSERT" in sql_query or "UPDATE" in sql_query:
                result_array = "Updated information"
            else:
                result_array = "No information"
            
            sql_query_result = sql_query + "\n\n" + "Result (" + str(len(result)) + " entries): " + str(result_array) + "\n\n"
            
        except:
            print("Error reply")
            #pdb.set_trace()
            return ""   
        
        #if result:
        #    pdb.set_trace()
        
        return sql_query_result
        
        
    def function_request(self, sender, text, info, robotState, message_history, print_debug):
    
        function_cont = "<|eot_id|><|start_header_id|>user<|end_header_id|>History of messages: " + str(list(message_history)[-self.max_message_history:]) + "\nNew message: " + str({"Sender": sender, "Message": text}) + "\n" + "Output a list of actions to take.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        llm_answer = self.free_response(self.function_query + function_cont, True, bigger_model=True)
        
        sql_query_result = ""
        
        try:
            open_k = llm_answer.index('`')
            close_k = llm_answer.rindex('`')
            sql_query = llm_answer[open_k+1:close_k]
            result = robotState.cursor.execute(sql_query).fetchall()
            
            sql_query_result = sql_query + "\n\n" + "Result: " + str(result) + "\n\n"
            
        except:
            print("Error reply")
            return ""   
        
        return sql_query_result

    def convert_to_ai(self, sender, text, info, robotState, message_history, print_debug, asking_for_help):
        
        result_err = 0
        self.change_llm = 0
        functions = []
        
        if self.reply_query2((sender, text), message_history,robotState, info):
            if asking_for_help:
                message, functions = self.help_confirmation_query((sender, text), message_history,robotState, info)
            else:
                message, functions = self.chat_query((sender, text), message_history,robotState, info)
            
            message_to_user = True
        else:
            message = self.noop
            message_to_user = False
        
        return message, message_to_user, functions
        
        '''
        
        if self.reply_query(sender, text, info, robotState, message_history, print_debug):

            #self.exchanged_messages[sender].append({"Sender": "Agent " + self.agent_id, "Message": text})

            type_request = "none"
            if robotState.args.sql and self.openai:
                type_request = self.type_of_request(sender, text, info, robotState, message_history, print_debug)
                
                if type_request == "information":
                
                    sql_result = self.sql_request(sender, text, info, robotState, message_history, print_debug)
                    
                    message_to_user = False
                    
                    result_str = self.personalize_message_sql(sender, text, sql_result, message_history)
                    if result_str:
                        message_to_user = True
                        
                elif type_request == "action":
                    type_request = "none"
                    #self.function_request(sender, text, info, robotState, message_history, print_debug)

            if not robotState.args.sql or (robotState.args.sql and type_request == "none"):
                #phrase_list = self.create_message_variants(sender, text, info, robotState, message_history, print_debug)
                
                #if not phrase_list:
                #    return "",False
            
                phrase_list = []
                result_str,message_to_user = self.match_message(sender, text, info, robotState, message_history, print_debug, phrase_list)
            
        else:
            result_str = self.noop
            message_to_user = False
        
        return result_str,message_to_user
        '''
     
    def use_llm(self, prompt, bigger_model=False, response_format=None):
    
        if not self.openai and time.time() - self.time_counted_groq < self.time_limit_groq:
            llm_answer = self.make_query_openai(prompt) #self.make_query(self.pipeline, self.tokenizer, prompt)
        else:
            self.openai = True
            recoverable_error = True
            while recoverable_error:
                try:
                    llm_answer = self.make_query_openai(prompt,bigger_model,response_format)
                    recoverable_error = False
                except groq.RateLimitError as e:
                    print("Deep Sleep")
                    #time.sleep(60)
                    
                    if self.big_model == "llama-3.3-70b-versatile":
                        self.big_model = "llama-3.3-70b-specdec"
                    else:
                        self.big_model = "llama-3.3-70b-versatile"
                    
                    '''
                    #pdb.set_trace()
                    self.openai = False
                    print("Groq failed")
                    self.setup_llama()
                    self.time_counted_groq = time.time()
                    llm_answer = self.make_query_openai(prompt) #self.make_query(self.pipeline, self.tokenizer, prompt)
                    recoverable_error = False
                    '''
                except Exception as e:
                    #pdb.set_trace()
                    time.sleep(2)
                    
        return llm_answer
        
    def free_response(self, text, print_debug, bigger_model=False, response_format=None):
    
        prompt = text#"You are a special operative in a mission to sense objects in a scene and collect those that are dangerous. You need to collaborate with your fellow teammates. Output a short response to the following message from one of your teammates: " + text + "\nYOUR RESPONSE >>>\n"
        print("Free response")
        
        llm_time = time.time()
        llm_answer = self.use_llm(prompt, bigger_model, response_format)

        if print_debug:
            print(prompt,llm_answer)
            print("Response prompt time:", time.time() - llm_time)
            
        return llm_answer


    def personalize_message(self, text, target, message_history):
    
        personalize_prompt_cont = "<|eot_id|><|start_header_id|>user<|end_header_id|>History of messages: " + str(list(message_history)[-20:]) + "\nMessage to send: \"" + text + "\"\n" + "How would you adapt this message to send it to Agent " + str(target) + "?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        llm_answer = self.free_response(self.output_personalization + personalize_prompt_cont, True)
        
        return_string = ""
        
        try:
            bracket1_idx = llm_answer.find('{')
            bracket2_idx = llm_answer.find('}')
            return_string = eval(llm_answer[bracket1_idx:bracket2_idx+1])["Message"]
        except:
            print("No personalization of message")
            
        return return_string
        
        
    def personalize_message_sql(self, sender, text, sql_text, message_history):
    
        personalize_prompt_cont = "<|eot_id|><|start_header_id|>user<|end_header_id|>History of messages: " + str(list(message_history)[-20:]) + "\nNew message: " + str({"Sender": sender, "Message": text}) + "\n" + sql_text + "\n" + "How would you adapt this result to send it to your teammates?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        llm_answer = self.free_response(self.output_personalization_sql + personalize_prompt_cont, True, response_format={"type": "json_object"})
        
        return_string = ""
        
        try:
            bracket1_idx = llm_answer.find('{')
            bracket2_idx = llm_answer.find('}')
            return_string = "[INFORMATION] " + eval(llm_answer[bracket1_idx:bracket2_idx+1])["Message"]
        except:
            print("No personalization of message")
            
        return return_string
            
            
        
    def summarize_messages(self, message_history, robotState, info):
        
        prompt = self.start_header_str + "system" + self.end_header_str + self.base_prompt + "\nHere is the agents' conversation so far: "
        
        #Add context, current action
        list_message_history = [[m["Sender"],m["Message"]] for m in message_history]
        
        prompt += str(list_message_history)
        
        prompt += self.end_token_str + self.start_header_str + "user" + self.end_header_str
        
        prompt += "\nTask: Summarize the current conversation in one sentence.\n\nOutput format: Output a json of the following format: \n{\n\"Conversation Summary\": \"<One sentence summary>\"\n}\n"
        
        output = self.use_llm(prompt, bigger_model=True, response_format={"type": "json_object"})
        print(prompt)
        print(output)
        
        self.log_file.write(prompt)
        self.log_file.write(output)
        
        real_output = output
        #pdb.set_trace()
        #real_output = output[len(prompt)+5:]
        #pdb.set_trace()
        dict_txt = real_output[real_output.find("{"): real_output.find("}")+1]
        dict_txt = eval(dict_txt)

        
        self.conversation_summaries.append(dict_txt["Conversation Summary"])
        
        out_file = open("chat_" + self.env.robot_id + ".json", "w")
        json.dump({"summary": self.conversation_summaries}, out_file)
        
        self.agent_profiling(message_history, robotState, info)
        
    def agent_profiling(self, message_history, robotState, info):
    
        prompt = self.start_header_str + "system" + self.end_header_str + self.base_prompt + "\nHere is the agents' conversation so far: "
        
        list_message_history = [[m["Sender"],m["Message"]] for m in message_history]
        prompt += str(list_message_history)
        
        prompt += "\nHere is a summary of past events: " + str(self.conversation_summaries)
        
        atts = []
        attitude_json = ""
        for a_idx in range(robotState.get_num_robots()):
            if not robotState.get("agents", "type", a_idx):
                robot_id = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(a_idx)]
                atts.append(["Agent " + robot_id,robotState.get("agents", "attitude", a_idx)])
                if attitude_json:
                    attitude_json += ",\n"
                attitude_json += "\"Agent " + robot_id + "\": \"<One sentence attitude description>\""
                
        attitude_json = "\n{\n" + attitude_json + "\n}\n"
                
        
        prompt += "\nHere is a description of the attitude of some of the agents: " + str(atts)
        
        prompt += self.end_token_str + self.start_header_str + "user" + self.end_header_str
        
        prompt += "\nTask: According to the previous context update the attitude descriptions for some of the agents.\n\nOutput format: Output a json of the following format: " + attitude_json
        
        output = self.use_llm(prompt, bigger_model=True, response_format={"type": "json_object"})
        print(prompt)
        print(output)
        
        self.log_file.write(prompt)
        self.log_file.write(output)
        
        real_output = output
        #pdb.set_trace()
        #real_output = output[len(prompt)+5:]
        #pdb.set_trace()
        dict_txt = real_output[real_output.find("{"): real_output.find("}")+1]
        dict_txt = eval(dict_txt)
        
        for a_key in dict_txt.keys():
            agent_id = a_key.replace("Agent ", "").strip()
            
            if agent_id != self.env.robot_id:
                a_idx = info['robot_key_to_index'][agent_id]
                if not robotState.get("agents", "type", a_idx):
                    robotState.set("agents", "attitude", a_idx, dict_txt[a_key], info["time"])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Process Text"
    )

    parser.add_argument("--text", type=str, help="Text input")
    parser.add_argument("--free", action='store_true', help="Free-form message exchange")


    args = parser.parse_args()
    
    human_ai_text = Human2AIText("A")
    info = {}
    items = {}
    
    if not args.free:
        human_ai_text.convert_to_ai(args.text, info, items, [], True)
    else:
        human_ai_text.free_response(args.text, True)

