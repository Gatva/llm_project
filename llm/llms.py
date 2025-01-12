import openai
import anthropic
import os
import logging
import time
import google.generativeai as genai
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig, GenerationConfig
from fastchat.conversation import get_conv_template
import argparse
from groq import Groq
import torch
logging.basicConfig(filemode='w', filename='log', format="%(asctime)s - %(levelname)s - %(message)s",
                    level=logging.INFO)


class LLM:
    def __init__(self):
        
        self.model = None
        self.tokenizer = None
        pass

    def generate(self, prompt):
        raise NotImplementedError("")

class LocalLLM(LLM):
    def __init__(self, model_path,
                 system_prompt=None,
                 device='cuda:0',
                 template_name=None,
                 quantify=False):
        
        super().__init__()
        self.model_path=model_path
        if quantify:
            self.model=AutoModelForCausalLM.from_pretrained(self.model_path,torch_dtype=torch.float16).to(device)
        else:
            self.model=AutoModelForCausalLM.from_pretrained(self.model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tmplate_name=template_name
        self.system_prompt=system_prompt
        self.conv_template = None

    def build_message(self, prompt):
        self.conv_template = get_conv_template(self.tmplate_name)
        if self.system_prompt:
            self.conv_template.set_system_message(self.system_prompt)
        self.conv_template.append_message(self.conv_template.roles[0], prompt)
        self.conv_template.append_message(self.conv_template.roles[1], None)
        return self.conv_template.get_prompt()
    
    @torch.inference_mode()
    def generate(self, prompt, temperature=1.0, max_tokens=512, n=1, max_trial=5, fail_sleep_time=5):
        
        if self.tmplate_name:
            self.build_message(prompt)
            message=self.conv_template.get_prompt()
        else:
            message=prompt
        # print(message)
        input=self.tokenizer(message,return_tensors="pt").to(self.model.device)
        out=self.model.generate(
                **input,
                generation_config=GenerationConfig(
                    do_sample=False,
                    no_repeat_ngram_size=3,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    num_beams=5
                    )
            )
        return self.tokenizer.batch_decode(out,skip_special_tokens=True)[0][len(message):].strip()
        

    @torch.inference_mode()
    def generate_batch(self, prompts, temperature=0.01, max_tokens=512, repetition_penalty=1.0, batch_size=16):
        messages=[]
        for prompt in prompts:
            if self.tmplate_name:
                self.build_message(prompt)
                message=self.conv_template.get_prompt()
            else:
                message=prompt
            messages.append(message)

        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        input_ids = self.tokenizer(messages, padding=True).input_ids
        # load the input_ids batch by batch to avoid OOM
        outputs = []
        for i in range(0, len(input_ids), batch_size):
            output_ids = self.model.generate(
                torch.as_tensor(input_ids[i:i+batch_size]).cuda(),
                do_sample=False,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_tokens,
            )
            output_ids = output_ids[:, len(input_ids[0]):]
            outputs.extend(self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False))
        return outputs


        
    
class OpenAILLM(LLM):
    def __init__(self, 
                 model_path='gpt-3.5-turbo', 
                 api_key=os.getenv("OPENAI_API_KEY"),
                 system_prompt=None,
                 BASE_URL='https://api.aiproxy.io/v1'):
        super().__init__()
        self.model = model_path
        self.client = openai.OpenAI(api_key=api_key,base_url=BASE_URL)
        self.conv_template = get_conv_template('chatgpt')
        self._system_prompt = system_prompt

    @property
    def system_prompt(self):
        return self._system_prompt
    
    @system_prompt.setter
    def system_prompt(self, value):
        self._system_prompt = value
        
    def build_message(self, prompt):
        if self.system_prompt:
            self.conv_template.set_system_message(self.system_prompt)
        self.conv_template.append_message(self.conv_template.roles[0], prompt)
        return self.conv_template.to_openai_api_messages()
    
    def generate(self, prompt, temperature=1.0, max_tokens=512, n=1, max_trial=5, fail_sleep_time=5):
        self.conv_template.messages = []
        massage = self.build_message(prompt)
        for _ in range(max_trial):
            try:
                responses = self.client.chat.completions.create(model=self.model,
                                                                messages=massage,
                                                                temperature=temperature,
                                                                n=n,
                                                                max_tokens=max_tokens)
                return [responses.choices[i].message.content for i in range(n)]
            except:
                logging.exception(
                    "There seems to be something wrong with your ChatGPT API. Please follow our demonstration in the slide to get a correct one.")
                time.sleep(fail_sleep_time)

        return ["" for i in range(n)]

class AntropicLLM(LLM):
    def __init__(self, model_path='claude-3-5-haiku-20241022', 
                 api_key=os.getenv("Antropic_API_KEY"),
                 system_prompt=None,
                 BASE_URL='https://api.aiproxy.io/'):
        super().__init__()
        self.model = model_path
        self.client = anthropic.Anthropic(api_key=api_key,base_url=BASE_URL)
        self.system_prompt = system_prompt

    def generate(self, prompt, temperature=1.0, max_tokens=512, n=1, max_trial=5, fail_sleep_time=5):

        for _ in range(max_trial):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=self.system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                )
                return message.content[0].text
            except:
                logging.exception(
                    "There seems to be something wrong with your Anthropic API. Please follow our demonstration in the slide to get a correct one.")
                time.sleep(fail_sleep_time)


class GeminiLLM(LLM):
    def __init__(self, model_path='gemini-pro', api_key=os.getenv("Gemini_API_KEY")):
        super().__init__()
        self.model = model_path
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model)

    def generate(self, prompt, temperature=1.0, max_tokens=512, n=1, max_trial=5, fail_sleep_time=5):
        content = [
            {'role': 'user', 'parts': prompt}
        ]
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        for _ in range(max_trial):
            try:
                responses = self.client.generate_content(
                    contents=content,
                    generation_config=generation_config,
                    safety_settings=[
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE", },
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE", },
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE", },
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE", }]
                )
                return [responses.text]
            except:
                logging.exception(
                    "There seems to be something wrong with your Gemini API. Please follow our demonstration in the slide to get a correct one.")
                time.sleep(fail_sleep_time)
        return ["" for i in range(n)]

class Llama3APILLM(LLM):
    def __init__(self, model_path='llama-3.1-70b-specdec', api_key=os.getenv("LLAMA2_API_KEY")):
        super().__init__()
        print(api_key)
        self.model=model_path
        self.client = Groq(
            api_key=api_key
        )
    def generate(self,prompt, temperature=1.0, max_tokens=512, n=1, max_trial=5, fail_sleep_time=5):
        for _ in range(max_trial):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                print('hhelp')
                return [chat_completion.choices[0].message.content]
            except:
                logging.exception(
                    "There seems to be something wrong with your Gemini API. Please follow our demonstration in the slide to get a correct one.")
                time.sleep(fail_sleep_time)
            return ['']
def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    return argparser.parse_args()

if __name__ == "__main__":

    # args=parse_args()
    # if 'gpt' in args.model_name:
    #     llm = OpenAILLM(model_path=args.model_name)
    # elif 'claude' in args.model_name:
    #     llm = AntropicLLM(model_path=args.model_name)
    # elif 'gemini' in args.model_name:
    #     llm = GeminiLLM(model_path=args.model_name)
    
    # print(llm.generate('hello'))

    # api='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'

    # llm=LocalLLM(model_path='/home/star/jf/python_project/model/tulu-2-dpo-7B-GPTQ',template_name='tulu')
    # print(llm.generate('hello'))
    # llm=OpenAILLM(api_key='sk-FPBTmynLIJxztYg79aLdKnBycFkrbiGCv1MQWJt6sHV2VJBc')
    
    # print(llm.generate('hello'))

    llm=LocalLLM(model_path="/home/star/jf/python_project/model/llama2-7b-chat-hf",template_name='llama-2',quantify=True)
    print(llm.generate('hello'))
    print('ok')