import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

class Model:
    def __init__(self,
                 model_name = "upstage/llama-30b-instruct-2048",
                 cache_dir="/nobackup1/allenw/Scratch/") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            cache_dir=cache_dir,
        )
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    def query(self, system, user_input):
        inputs = self.generate_input(system, user_input)
        output = self.model.generate(**inputs, streamer=self.streamer, use_cache=True, max_new_tokens=float('inf'))
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text

    def generate_input(self, system, user_input):
        input_string = f"### System:\n{system}\n### User:\n{user_input}\n### Assistant:\n"
        inputs = self.tokenizer(input_string, return_tensors="pt").to(self.model.device)
        del inputs["token_type_ids"]
        return inputs
        

if __name__ == "__main__":
    model = Model()
    model.query("you are a therapist", "tokamak sad")