import random

import torch
from datasets import DatasetDict
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel, get_peft_model
from peft.peft_model import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load the base model
base_model_name = "microsoft/phi-2"
device = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    # bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name, quantization_config=bnb_config, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"

# load the Lora model
lora_model_name = "/home/ubuntu/projects/deception-prompt-hack/trained_models/240630_12-18_phi2-qlora-election-deception"  # replace with your LoRA model name
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "dense",
        "fc1",
        "fc2",
    ],  # print(model) will show the modules to use
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
# config = PeftConfig(lora_model_name)
# model = PeftModel.from_pretrained(model, peft_model_path)
lora_model = PeftModelForCausalLM.from_pretrained(
    model, lora_model_name, config=config
)  # .to(device)
# lora_model = AutoPeftModelForCausalLM.from_pretrained(lora_model_name, config=config)
# lora_model.load_adapter(lora_model_name)
# lora_model.enable_adapters()

lora_model = lora_model.merge_and_unload()
lora_model.eval()

data_dir = "/home/ubuntu/projects/deception-prompt-hack/data/ai-democracy-v2"
test_set = DatasetDict.load_from_disk(data_dir)["test"]

for i in random.sample(range(0, len(test_set)), 3):
    inputs = tokenizer(test_set[i]["prompt"], return_tensors="pt")  # .to(device)
    with torch.inference_mode():
        outputs = lora_model.generate(**inputs, repetition_penalty=1.0, max_length=256)
    generated_text = tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(f"\n{i}th Prompt:\n{test_set[i]['prompt']}")
    print(f"\nPrediction:\n{generated_text}")
    print(f"\nReference:\n{test_set[i]['response']}")
    # print("Truthful: {}".format(test_set[i]["answer_truthful"] in generated_text))
