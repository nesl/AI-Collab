from transformers import AutoTokenizer
import transformers
import torch
import json
import re
import argparse
import pdb
import os
import openai
from string import Formatter

class Human2AIText:

    def __init__(self, agent_id):
    
        self.openai = True
            
        if self.openai:
            openai.api_key = os.getenv("OPENAI_API_KEY")
    
        self.START_INST = "[INST]"
        self.END_INST = "[/INST]"

        self.START_SYS = "<<SYS>>"
        self.END_SYS = "<</SYS>>"
        self.noop = "Nothing"

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
        
        self.SYS_PROMPT = """You are an assistant responsible for extracting the relevant information from the text 
        I give you into a JSON format output. This JSON should fit the following format:

        ```
        {
            "action": "the action being described by the message",
            "agent": "the agent being referred to in the message",
            "agent_type": "the type of agent (either human or AI)",
            "coordinates": "the coordinates mentioned in the message",
            "object": "the object number being referred to in the message",
            "danger": "a boolean indicating whether the object is dangerous",
            "weight": "an integer indicating the weight of the object",
            "probability": "the probability of our predictions being correct",
            "time": "the time mentioned in the message"
        }
        ```

        You should allways give the JSON inside a code block. The action keyword must always be included.
        The other keywords only need to be included if present.

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


        self.EXAMPLE_PROMPTS = [
            ("user", "Can someone help me carry object 3?"),
            (
                "llama", """
        ```
        {
            "action": "request_pickup_help",
            "object": "3"
        }
        ```
                """
            ),
            ("user", "I can't help you picking up such object at the moment"),
            (
                "llama", """
        ```
        {
            "action": "deny_pickup_help"
        }
        
        ```
                """
            ),
            ("user", "Object 1 (weight: 2) Last seen in (9,8) at 11:30. Status: dangerous, Prob. Correct: 80%"),
            (
                "llama", """
        ```
        {
            "action": "provide_info",
            "coordinates": "(9,8)",
            "weight": "2",
            "dangerous": true,
            "probability": "80.0%",
            "time": "11:30",
            "object": "1"
        }
        ```
                """
            ),
        ]
        
        if not self.openai:
            
            model = "meta-llama/Llama-2-7b-chat-hf"#"microsoft/phi-2"#"meta-llama/Llama-2-7b-chat-hf"

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
        answers = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=1000,
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

    def build_prompt(self, sys_prompt, example_prompts, query, message_history):
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
            message_history_prompt = "The next messages have been exchanged so far between agents, you are agent " + self.agent_id + " -> " + ', '.join(["Agent " + rm[0] + ": " + rm[1] for message_received in message_history for rm in message_received])



        return "\n".join(
            f"{self.START_INST}{self.START_SYS}\n{sys_prompt}\n{message_history_prompt}\n{self.END_SYS}{self.END_INST}\n{self.START_INST}{prompt}{self.END_INST}" if i == 0 else
            f"{self.START_INST}{prompt}{self.END_INST}" if i % 2 == 0 else prompt
            for i, (speeker, prompt) in enumerate(example_prompts)
        )


    def extract_first_json(self, text):
        code_block_search = re.search("```([^`]*)```", text)

        if code_block_search:
            first_code_block = code_block_search.group(1)

            incorrect_json = re.search(',\s*}',first_code_block)

            if incorrect_json:

                first_code_block = first_code_block[:incorrect_json.span()[0]] + "}"

            return json.loads(first_code_block)
        else:
            raise Exception("Code block not found")


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
                    
                    if action == "request_pickup_help":
                        if "number_robots" not in list(extracted_json.keys()):
                            extracted_json["number_robots"] = 1
                      
                    if "agent" in arguments_format and "agent" not in list(extracted_json.keys()):
                        if action == "request_agent_info" or action == "provide_agent_info" or action == "no_agent_info" or action == "agree_sensing_help" or action == "agree_pickup_help" or action == "make_agent_move":
                            missing_arguments.append("Which agent are you referring to? ")
                        else:
                            extracted_json["agent"] = self.agent_id
                    
                    no_object_id = False       
                    if "object" in arguments_format and "object" not in list(extracted_json.keys()):
                        if action == "request_pickup_help":
                            extracted_json["object"] = 9999
                        elif action == "request_sensing_help":
                            no_object_id = True
                        else:
                            missing_arguments.append("Which object are you referring to? ")
                            
                    if "coordinates" in arguments_format and "coordinates" not in list(extracted_json.keys()):
                        if action == "request_sensing_help":
                            if no_object_id:
                                missing_arguments.append("Which object are you referring to? ")
                            else:
                                if info and extracted_json["object"] in info["object_key_to_index"].keys():
                                    ob_idx = info["object_key_to_index"][extracted_json["object"]]
                                    if items[ob_idx]["item_location"][0] == -1 and items[ob_idx]["item_location"][1] == -1:
                                        missing_arguments.append("I need its coordinate location. Can you provide it please? ")
                                    else:
                                        extracted_json["coordinates"] = "(99.99,99.99)"
                                else:
                                    missing_arguments.append("I need its coordinate location. Can you provide it please? ")
                                
                        
                        else:
                            missing_arguments.append("I need its coordinate location. Can you provide it please? ")
                    
                    if "danger" in arguments_format:
                        if "danger" not in list(extracted_json.keys()):
                            extracted_json["danger"] = "0"
                        else:
                            extracted_json["danger"] = str(int(extracted_json["danger"])+1)
                    if "weight" in arguments_format and "weight" not in list(extracted_json.keys()):
                        extracted_json["weight"] = "0"
                    if "probability" in arguments_format and "probability" not in list(extracted_json.keys()):
                        extracted_json["probability"] = "0.0%"
                    if "time" in arguments_format and "time" not in list(extracted_json.keys()):
                        extracted_json["time"] = "00:00"
                        
                    
                    
                    if missing_arguments:
                        
                        missing_string = "I need some more information. "
                        for m_str in missing_arguments:
                            missing_string += m_str
                            
                        return missing_string,4
                        
                    
                    return message.format(**extracted_json),0
                except KeyError as e:
                    return f"The predicted message ({action}: {message}) requires the keyword: {e}, which was not found in the json",3
            else:
                return f"The predicted action {action} does not appear in the expected CNL messages",2
        else:
            return "The generated JSON does not contain the keyword 'action'",1


    def convert_to_ai(self, text, info, items, message_history, print_debug):
        
        
        
        prompt = self.build_prompt(self.SYS_PROMPT, self.EXAMPLE_PROMPTS, self.previous_query + text, message_history)
        
        self.previous_query = ""

        if print_debug:
            print(prompt)
            print("======================================================")


        max_retries = 3
        
        retries = -1
        message_to_user = False
        
        while retries < max_retries: #Iterate until we get a correct answer
        
            if not self.openai:
                llm_answer = self.make_query(self.pipeline, self.tokenizer, prompt)
            else:
                llm_answer = self.make_query_openai(prompt)

            if print_debug:
                print(llm_answer)
                print("======================================================")

            try:
                first_json = self.extract_first_json(llm_answer)

                if print_debug:
                    print(first_json)
                    print("======================================================")


                result_str,result_err = self.json_to_message(first_json, info, items)
                if print_debug:
                    print(result_str)
                
                
                    
                if result_err:
                   
                    if result_err == 1 or result_err == 2:
                        retries += 1
                        result_str = ""
                        
                        if not self.openai and result_err == 2 and retries == max_retries:
                            new_prompt = "To which of the following list of names is '" + first_json.get('action') + "' more similar to? List of names: " + str(list(self.CNL_MESSAGES.keys())) + ". Choose only one\n"
                            if not self.openai:
                                llm_answer = self.make_query(self.pipeline, self.tokenizer, new_prompt)
                            else:
                                llm_answer = self.make_query_openai(new_prompt)
                            pdb.set_trace()
                        
                        continue
                    elif result_err == 4:
                        message_to_user = True
                        #self.previous_query = text
                elif result_str == self.noop:
                    message_to_user = True
                    result_str = self.free_response(text,True)
          
                break
                
                
            except:

                print("Error in string format")
                result_str = ""
            
            retries += 1
        
        if result_err and result_err != 4:
             message_to_user = True
             result_str = "I didn't understand. "
            
        return result_str,message_to_user
        
    def free_response(self, text, print_debug):
    
        prompt = "You are a special operative in a mission to sense objects in a scene and collect those that are dangerous. You need to collaborate with your fellow teammates. Output a short response to the following message from one of your teammates: " + text + "\nYOUR RESPONSE >>>\n"
        print("Free response")
        
        if not self.openai:
            llm_answer = self.make_query(self.pipeline, self.tokenizer, prompt)
        else:
            llm_answer = self.make_query_openai(prompt)

        if print_debug:
            print(llm_answer)
            
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
        human_ai_text.convert_to_ai(args.text, info, items, True)
    else:
        human_ai_text.free_response(args.text, True)

