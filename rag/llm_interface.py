import os
import openai
import logging
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

class LLMInterface:
    def __init__(self) -> None:
        pass

    def query(self, system_message, user_message):
        raise NotImplementedError()

class HuggingFaceInterface(LLMInterface):
    def __init__(self,
                 model_name = "upstage/llama-30b-instruct-2048",
                 cache_dir="/nobackup1/allenw/Scratch/") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            rope_scaling={"type": "dynamic", "factor": 2}, # allows handling of longer inputs
            cache_dir=cache_dir,
        )
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    def query(self, system_message, user_message):
        inputs = self.format_input(system_message, user_message)
        output = self.model.generate(**inputs, streamer=self.streamer, use_cache=True, max_new_tokens=float('inf'))
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text

    def format_input(self, system, user_input):
        input_string = f"### System:\n{system}\n### User:\n{user_input}\n### Assistant:\n"
        inputs = self.tokenizer(input_string, return_tensors="pt").to(self.model.device)
        del inputs["token_type_ids"]
        return inputs

class OpenAIInterface(LLMInterface):
    def __init__(self, model_name="gpt-3.5-turbo") -> None:
        super().__init__()
        openai.api_type = "azure"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if openai.api_key is None:
            logging.warning("openai.api_key is None")

        openai.api_base = "https://test-oai69420.openai.azure.com/"
        openai.api_version = "2023-05-15"
        self.model_name = model_name

    def query(self, system_message, user_message):
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return completion.choices[0].message.content