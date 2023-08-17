import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer
)
from peft import PeftModel

class TokenWiseLLM:
    def __init__(self, model_name, use_bitsandbyts=False):
        self.model_name = model_name
        if "Llama" in model_name:
            # For Llama
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        else:
            # For other models
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if use_bitsandbyts:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                ),
                device_map={"":0}
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map={"":0}
            )
        self.model.resize_token_embeddings(len(self.tokenizer))

    def load_lora(self, lora_adapter_dir):
        self.model = PeftModel.from_pretrained(
            self.model,
            lora_adapter_dir,
            device_map={"":0}
        )
        self.model.eval()
    
    def inference(self, prompt, max_length=1000):
        prompt = f"### Question: {prompt}\n### Answer: "
        print(prompt, end="")
        step = 0
        while step < max_length:
            inputs = self.tokenizer(
                prompt,
                return_token_type_ids=False,
                return_tensors="pt",
            ).to("cuda:0")

            with torch.no_grad():
                sequences = self.model.generate(
                    **inputs,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    max_length=len(inputs['input_ids'][0]) + 1,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            output = self.tokenizer.decode(sequences[0], skip_special_tokens=True)
            print(output[len(prompt):], end="", flush=True)
            if len(output[len(prompt):]) == 0:
                break
            prompt = output
            step += 1
        
        print("")
        return
    