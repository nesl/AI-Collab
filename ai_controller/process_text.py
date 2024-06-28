from transformers import AutoTokenizer
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

class Human2AIText:

    def __init__(self, env):
    
        self.openai = False
        
        agent_id = env.robot_id
        
        self.agent_names = list(env.robot_key_to_index.keys())
        self.agent_names.append(agent_id)
            
        if self.openai:
            openai.api_key = os.getenv("OPENAI_API_KEY")
    
        self.START_INST = "[INST]"
        self.END_INST = "[/INST]"
        
        self.START_STR = "<s>"
        self.END_STR = "</s>"

        self.START_SYS = "<<SYS>>"
        self.END_SYS = "<</SYS>>"
        self.noop = "No message matches"


        self.CNL_MESSAGES = {
            "Ask about agent": "Where is agent {agent_id}. ",
            "No knowledge of agent": "I don't know where is agent {agent_id}. ",
            "Ask about object": "What do you know about object {object_id}. ",
            "No knowledge of object": "I know nothing about object {object_id}. ",
            "Object not found": "Hey {agent_id}, I didn't find object {object_id}. ",
            "Help sense object": "Hey {agent_id}, can you help me sense object {object_id} in location {location}, last seen at {time}. ",
            "Help carry object": "I need {agent_count} more robots to help carry object {object_id}. ",
            "Reject help from agent": "Nevermind {agent_id}. ",
            "Follow someone": "Thanks, I'll follow you {agent_id}. ",
            "Be followed by someone": "Thanks, follow me {agent_id}. ",
            "Cancel help request": "Nevermind. ",
            "End collaboration": "No need for more help. ",
            "Accept request for help": "I can help you {agent_id}. ",
            "Reject request for help": "I cannot help you {agent_id}. ",
            "Reject request to follow/be followed": "I didn't offer my help to you {agent_id}. ",
            "Come closer": "Come closer {agent_id}. ",
            "Request to move": "Hey {agent_id}, I need you to move. ",
            "End participation": "Let's end participation. ",
            "Don't end participation": "Wait, let's not end participation yet. ",
            "Object information": "Object {object_id} (weight: {object_weight}) Last seen in {location} at {time}. Status: {danger}, Prob. Correct: {probability}. ",
            "Agent information": "Agent {agent_id} (type: {agent_type}) Last seen in {location} at {time}. ",
            "Not relevant": self.noop
        }


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
<s>[INST]<<SYS>>
You are Agent """ + agent_id + """. You are part of a team whose mission is to dispose of all dangerous objects in a scene. You can move around, detect whether an object is dangerous or not, and carry objects. Objects are all of the same type, they only differ in their location, weight, and whether they are dangerous. Once you find a dangerous object you must check if you have the necessary strength to pick it up and put it in the safe area you already know. Try to engage with your teammates to come up with a strategy. Be brief in your responses.  Analyze the content of each message and try to tell me who are they answering to. Some messages may be directed towards a subset of the group. Output a JSON format with the following field for each analyzed message: "reply_to", which should be any of the set {"Agent A", "Agent B", "Agent C", "Agent D", "Everyone"}.
<</SYS>>

History of messages: [{'Sender': 'Agent D', 'Message': 'Hi everyone'}]
New message: {'Sender': 'Agent B', 'Message': 'hi'}
Who is Agent B replying to? Output a JSON.[/INST]

>> {"reply_to": ["Agent D"]}

</s><s>[INST]History of messages: [{'Sender': 'Agent A', 'Message': 'Hey C'}, {'Sender': 'Agent A', 'Message': 'A do you have more info'}]
New message: {'Sender': 'Agent C', 'Message': 'Need help?'}
Who is Agent C replying to? Output a JSON.[/INST]

>> {"reply_to": ["Agent A"]}

</s><s>[INST]History of messages: [{'Sender': 'Agent B', 'Message': 'hello'}, {'Sender': 'Agent C', 'Message': 'hi'}, {'Sender': 'Agent D', 'Message': 'Hello everyone'}]
New message: {'Sender': 'Agent D', 'Message': 'Do we want to try a different strategy today'}
Who is Agent D replying to? Output a JSON.[/INST]

>> {"reply_to": ["Everyone"]}

"""
        
        self.phrase_generation ="""
<s>[INST]<<SYS>>
You are part of a team whose mission is to dispose of all dangerous objects in a scene. You can use a sensor to sense whether an object is dangerous or benign. You are going to be given a phrase and you have to create variations of it.
<</SYS>>

Phrase: "I need 2 more robots to help carry object 8."
Write at least 3 possible variations of the phrase and put them inside a list.[/INST]
        
>> ["can 2 other robots help me carry object 8", "hey I need help to carry an object", "come with me I need help for lifting an object"]

</s><s>[INST]Phrase: "I need you to move B."
Write at least 3 possible variations of the phrase and put them inside a list.[/INST]
        
>> ["please move!", "hey B why don't you move", "can you step aside"]

"""
        
        if not self.openai:
            
            model = "meta-llama/Meta-Llama-3-8B" #"meta-llama/Llama-2-7b-chat-hf"#"microsoft/phi-2"#"meta-llama/Llama-2-7b-chat-hf"

            self.tokenizer = AutoTokenizer.from_pretrained(model)
            
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                torch_dtype=torch.float16,
                device_map="sequential",
                #max_memory={0: '20GiB', 1: '12GiB'}
            )
        
        self.agent_id = agent_id
        self.previous_query = ""

    class bcolors:
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        ENDC = '\033[0m'


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

    def make_query_openai(self, prompt):
    
        response = openai.ChatCompletion.create(
              model= "gpt-3.5-turbo", #"gpt-4", #"gpt-3.5-turbo",
              messages=[
                    {"role": "user", "content": prompt}
                ],
              #functions=self.llm_functions,
              #function_call="auto"
            )

        return response["choices"][0]["message"].get("content")

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
            f"{self.START_STR}{self.START_INST}{self.START_SYS}\n{sys_prompt}\n{message_history_prompt}\n{self.END_SYS}\n{prompt}{self.END_INST}" if i == 0 else
            f"{self.END_STR}{self.START_STR}{self.START_INST}{prompt}{specific_query}{self.END_INST}" if i % 2 == 0 else prompt
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


    def json_to_message(self, extracted_json, info, items):
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
                      
                    if "agent_id" in arguments_format and "agent_id" not in list(extracted_json.keys()):
                        if action == "Ask about agent" or action == "Agent information" or action == "No knowledge of agent":# or action == "Follow someone" or action == "Be followed by someone" or action == "Request to move":
                            missing_arguments.append("Which agent are you referring to? ")
                        else:
                            extracted_json["agent_id"] = self.agent_id
                    
                    no_object_id = False       
                    if "object_id" in arguments_format and "object_id" not in list(extracted_json.keys()):
                        if action == "Help carry object":
                            extracted_json["object_id"] = 9999
                        elif action == "Help sense object":
                            no_object_id = True
                        else:
                            missing_arguments.append("Which object are you referring to? ")
                            
                    if "location" in arguments_format and "location" not in list(extracted_json.keys()):
                        if action == "Help sense object":
                            if no_object_id:
                                missing_arguments.append("Which object are you referring to? ")
                            else:
                                if info and extracted_json["object_id"] in info["object_key_to_index"].keys():
                                    ob_idx = info["object_key_to_index"][extracted_json["object_id"]]
                                    if items[ob_idx]["item_location"][0] == -1 and items[ob_idx]["item_location"][1] == -1:
                                        missing_arguments.append("I need the coordinate location. ")
                                    else:
                                        extracted_json["location"] = "(99.99,99.99)"
                                else:
                                    missing_arguments.append("I need the coordinate location. ")
                                
                        
                        else:
                            missing_arguments.append("I need the coordinate location. ")
                    
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
                    if "location" in extracted_json:
                        if re.search('\( *-?\d+\.\d+ *, *-?\d+\.\d+ *\)',extracted_json["location"]) or re.search('\( *-?\d+ *, *-?\d+ *\)',extracted_json["location"]):
                            #location_split = extracted_json["location"].split(',')
                            #extracted_json["location"] = location_split[0] + '.0,' + location_split[1][:-1] + '.0)'
                            
                            location = eval(extracted_json["location"])
                            
                            extracted_json["location"] = str((float(location[0]), float(location[1]))).replace(" ","")
                            
                        elif not re.search('\(-?\d+\.\d+,-?\d+\.\d+\)',extracted_json["location"]) and not re.search('\(-?\d+,-?\d+\)',extracted_json["location"]):
                            missing_arguments.append("I don't understand where is it? ")
                            #return 'Argument location doesn\'t have the correct format',5
                    if "agent_id" in extracted_json:
                        
                        extracted_json["agent_id"] = extracted_json["agent_id"].upper()
                        
                        if not any([True if extracted_json["agent_id"].strip() == agent_name else False for agent_name in self.agent_names]):
                            extracted_json["agent_id"] = self.agent_id
                            #return 'Argument agent_id doesn\'t have the correct format',5
                            
                            
                    
                            
                    
                    if missing_arguments:
                        
                        missing_string = "I need some more information. "
                        for m_str in missing_arguments:
                            missing_string += m_str
                        
                        missing_string += "Please provide such information and ask again. "
                        return missing_string,4
                        
                    
                    return message.format(**extracted_json),0
                except KeyError as e:
                    return f"The predicted message ({action}: {message}) requires the keyword: {e}",3
            else:
                return f"Action \"{action}\" is not an allowed type of message, select one of the following: " + str(list(self.CNL_MESSAGES.keys())),2
        else:
            return "The generated JSON requires the keyword 'action'",1


    def convert_to_ai(self, sender, text, info, items, message_history, print_debug):
        
        reply_prompt_cont = "</s><s>[INST]History of messages: " + str(list(message_history)) + "\nNew message: " + str({"Sender": sender, "Message": text}) + "\n" + "Who is Agent " + sender + " replying to? Output a JSON.[/INST]\n\n"
        llm_time = time.time()
        llm_answer = self.free_response(self.reply_prompt + reply_prompt_cont, True)
        
        print("Response prompt time:", llm_time - time.time())
        
        open_k = llm_answer.index('{')
        close_k = llm_answer.index('}')
        
        result_err = 0
        
        try:
            response = eval(llm_answer[open_k:close_k+1])
        except:
            print("Error reply")
            return "", False   
        
        print(response, 'Agent ' + self.agent_id in response["reply_to"], self.agent_id in [agent.upper() for agent in response["reply_to"]], 'Everyone' in response["reply_to"], len(list(message_history)) == 1)
        
        if 'Agent ' + self.agent_id in response["reply_to"] or self.agent_id in [agent.upper() for agent in response["reply_to"]]  or 'Everyone' in response["reply_to"] or len(list(message_history)) == 1:

            max_retries = 3
            retries = -1
            while retries < max_retries:
                phrase_prompt = self.phrase_generation + "</s><s>[INST]Phrase: \"" + str(text) + "\"\nWrite at least 3 possible variations of the phrase and put them inside a list.[/INST]\n\n"
                
                llm_time = time.time()
                llm_answer = self.free_response(phrase_prompt, True)
                print("Phrase generation time:", llm_time - time.time())

                bracket1_idx = llm_answer.find('[')
                bracket2_idx = llm_answer.find(']')
                try:
                    phrase_list = eval(llm_answer[bracket1_idx:bracket2_idx+1])
                except:
                    retries += 1
                    continue
                break
        
            if retries >= 3:
                raise Exception("Cannot create phrases")
        
            prompt = self.build_prompt(self.SYS_PROMPT, self.EXAMPLE_PROMPTS, self.previous_query + "Original message: \"" + text + "\". Alternative messages: " + str(phrase_list) + ". ", self.SELECT_ACTION, message_history)
            
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
                if not self.openai:
                    llm_answer = self.make_query(self.pipeline, self.tokenizer, prompt)
                else:
                    llm_answer = self.make_query_openai(prompt)

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


                    result_str,result_err = self.json_to_message(first_json, info, items)
                    if print_debug:
                        print(result_str)
                    
                    
                        
                    if result_err:
                    
                        if result_str:
                            hints.append(result_str)
                        
                        prompt = self.build_prompt(self.SYS_PROMPT, self.EXAMPLE_PROMPTS, self.previous_query + "Original message: \"" + text + "\". Alternative messages: " + str(phrase_list) + ". Hints: [" + ', '.join(["\"" + m + "\"" for m in hints]) + "]", self.SELECT_ACTION, message_history) 
                        
                        if print_debug:
                            print(prompt)
                            print("======================================================")
                        
                       
                        if result_err == 1 or result_err == 2:
                            retries += 1
                            result_str = ""
                            
                            """
                            if not self.openai and result_err == 2 and retries == max_retries:
                                new_prompt = "To which of the following list of names is '" + first_json.get('action') + "' more similar to? List of names: " + str(list(self.CNL_MESSAGES.keys())) + ". Choose only one\n"
                                if not self.openai:
                                    llm_answer = self.make_query(self.pipeline, self.tokenizer, new_prompt)
                                else:
                                    llm_answer = self.make_query_openai(new_prompt)
                                print(llm_answer)
                            """
                            
                            continue
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
            
        else:
            result_str = self.noop
            message_to_user = False
        
        return result_str,message_to_user
        
    def free_response(self, text, print_debug):
    
        prompt = text#"You are a special operative in a mission to sense objects in a scene and collect those that are dangerous. You need to collaborate with your fellow teammates. Output a short response to the following message from one of your teammates: " + text + "\nYOUR RESPONSE >>>\n"
        print("Free response")
        
        if not self.openai:
            llm_answer = self.make_query(self.pipeline, self.tokenizer, prompt)
        else:
            llm_answer = self.make_query_openai(prompt)

        if print_debug:
            print(prompt,llm_answer)
            
        return llm_answer


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

