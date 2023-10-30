from transformers import AutoTokenizer
import transformers
import torch
import json
import re
import argparse
import pdb
from string import Formatter

class Human2AIText:

    def __init__(self, agent_id):
    
        self.START_INST = "[INST]"
        self.END_INST = "[/INST]"

        self.START_SYS = "<<SYS>>"
        self.END_SYS = "<</SYS>>"

        self.CNL_MESSAGES = {
            "request_help": "I need {number_robots} more robots to help carry object {object}. ",
            "nevermind": "Nevermind. ",
            "offer_help": "I can help you {agent}. ",
            "follow_me": "Thanks, follow me {agent}. ",
            "refuse_help": "Nevermind {agent}. ",
            "deny_help": "I cannot help you {agent}. ",
            "finish_help": "No need for more help. ",
            "refuse_help_offer": "Thanks for nothing. ",
            "request_info": "What do you know about object {object}. ",
            "provide_info": "Object {object} (weight: {weight}) Last seen in {coordinates} at {time}. Status Danger: {danger}, Prob. Correct: {probability}. ",
            "no_info": "I know nothing about object {object}. ",
            "no_action": "Nothing"

        }

        self.SYS_PROMPT = """You are an assistant responsible for extracting the relevant information from the text 
        I give you into a JSON format output. This JSON should fit the following format:

        ```
        {
            "action": "the action being described by the message",
            "agent": "the agent being referred to in the message",
            "coordinates": "the coordinates mentioned in the message",
            "object": "the object number being referred to in the message",
            "dangerous": "a boolean indicating whether the object is dangerous",
            "weight": "an integer indicating the weight of the object",
            "probability": "the probability of our predictions being correct",
            "time": "the time mentioned in the message",
            "number_robots": "the quantity of robots being asked for help"
        }
        ```

        You should allways give the JSON inside a code block. The action keyword must always be included.
        The other keywords only need to be included if present.

        The possible actions are:
        -"request_help": the message is asking for help.
        -"nevermind": the message is cancelling a previous instruction.
        -"offer_help": the message is offering help.
        -"follow_me": the message is requesting an agent to follow.
        -"refuse_help": the message is refusing help from another agent.
        -"deny_help": the message says that they cannot help.
        -"finish_help": the message says that help is no longer needed.
        -"refuse_help_offer": the message says that help is not needed.
        -"request_info": the message requests information about an object.
        -"provide_info": the message gives information about an object, such as its weight and whether it is dangerous.
        -"no_info": the message says the agent has no information about an object.
        -"no_action": whenever the message does not conform to any of the previous actions."""


        self.EXAMPLE_PROMPTS = [
            ("user", "Can someone help me carry object 3?"),
            (
                "llama", """
        ```
        {
            "action": "request_help",
            "number_robots": "1",
            "object": "3"
        }
        ```
                """
            ),
            ("user", "I can't help at the moment"),
            (
                "llama", """
        ```
        {
            "action": "deny_help"
        }
        
        ```
                """
            ),
            ("user", "Object 1 (weight: 2) Last seen in (9,8) at 11:30. Status Danger: dangerous, Prob. Correct: 80%"),
            (
                "llama", """
        ```
        {
            "action": "provide_info",
            "coordinates": "(9,8)",
            "weight": "2",
            "dangerous": true,
            "probability": "80%",
            "time": "11:30",
            "object": "1"
        }
        ```
                """
            ),
        ]
        
        model = "meta-llama/Llama-2-7b-chat-hf"

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="sequential",
            max_memory={0: '20GiB', 1: '12GiB'}
        )
        
        self.agent_id = agent_id

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


    def build_prompt(self, sys_prompt, example_prompts, query):
        example_prompts = list(example_prompts)

        assert example_prompts, "At least one example is required"
        assert set(map(lambda x: x[0], example_prompts[::2])) == {"user"}, "Even example prompts need to be 'user' messages"
        assert set(map(lambda x: x[0], example_prompts[1::2])) == {
            "llama"}, "Odd example prompts need to be 'llama' messages"

        example_prompts.append(
            ("user", query)
        )

        return "\n".join(
            f"{self.START_INST}{self.START_SYS}\n{sys_prompt}\n{self.END_SYS}\n{prompt}{self.END_INST}" if i == 0 else
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


    def json_to_message(self, extracted_json):
        action = extracted_json.get('action')

        if action:
            message = self.CNL_MESSAGES.get(action)
            if message:
                try:

                    arguments_format = [x[1] for x in Formatter().parse(message) if x[1]]
                    
                    
                    
                    if action == "request_help":
                        
                        if "object" not in list(extracted_json.keys()):
                            extracted_json["object"] = 9999
                            
                        if "number_robots" not in list(extracted_json.keys()):
                            extracted_json["number_robots"] = 1
                            
                    elif "agent" in arguments_format and "agent" not in list(extracted_json.keys()):
                        extracted_json["agent"] = self.agent_id
                         
                    
                    for af in arguments_format:
                        if af == "object":
                            if not str(extracted_json["object"]).isdigit():
                                return f"The arguments are not correct",1
                    
                    return message.format(**extracted_json),0
                except KeyError as e:
                    return f"The predicted message ({action}: {message}) requires the keyword: {e}, which was not found in the json",3
            else:
                return f"The predicted action {action} does not appear in the expected CNL messages",2
        else:
            return "The generated JSON does not contain the keyword 'action'",1


    def convert_to_ai(self, text, print_debug):
        
        

        prompt = self.build_prompt(self.SYS_PROMPT, self.EXAMPLE_PROMPTS, text)

        if print_debug:
            print(prompt)
            print("======================================================")


        max_retries = 3
        
        retries = 0
        while retries < max_retries: #Iterate until we get a correct answer
        
            llm_answer = self.make_query(self.pipeline, self.tokenizer, prompt)

            if print_debug:
                print(llm_answer)
                print("======================================================")

            try:
                first_json = self.extract_first_json(llm_answer)

                if print_debug:
                    print(first_json)
                    print("======================================================")


                result_str,result_err = self.json_to_message(first_json)
                if print_debug:
                    print(result_str)
                     
                    
                if result_err:
                    result_str = ""
                    
                    if result_err == 1:
                        retries += 1
                        continue
          
                break
                
                
            except:

                print("Error in string format")
                result_str = ""
            
            retries += 1
            
        return result_str
        
    def free_response(self, text, print_debug):
    
        prompt = "You are a special operative in a mission to sense objects in a scene and collect those that are dangerous. You need to collaborate with your fellow teammates. Output a short response to the following message from one of your teammates: " + text + "\nYOUR RESPONSE >>>\n"
    
        llm_answer = self.make_query(self.pipeline, self.tokenizer, prompt)

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
    
    if not args.free:
        human_ai_text.convert_to_ai(args.text, True)
    else:
        human_ai_text.free_response(args.text, True)

