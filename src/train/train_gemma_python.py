import os

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GemmaTokenizer,
)
from trl import SFTTrainer


def formatting_func(example):
    # Extract instruction and output from the example
    instruction = example["instruction"][0]
    output = example["output"][0]

    # Format the data into Gemma instruction template format
    text = f"<start_of_turn>user\n{instruction}<end_of_turn> <start_of_turn>model\n{output}<end_of_turn>"

    # Return the formatted data as a list
    return [text]


def test_trained_model(model, tokenizer):
    text = """<start_of_turn>how to convert json to dataframe.<end_of_turn>
    <start_of_turn>model"""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))


device = "cuda"
out_dir = "/home/ubuntu/projects/deception-prompt-hack/trained_models"
model_id = "google/gemma-1.1-2b-it"
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
# )

# tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"])  #
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     quantization_config=bnb_config,
#     device_map="auto",
#     token=os.environ["HF_TOKEN"],
# )

# # Configure LoraConfig for model pruning
# lora_config = LoraConfig(
#     r=8,
#     target_modules=[
#         "q_proj",
#         "o_proj",
#         "k_proj",
#         "v_proj",
#         "gate_proj",
#         "up_proj",
#         "down_proj",
#     ],
#     task_type="CAUSAL_LM",
# )

# # Load the dataset using the load_dataset function
# data = load_dataset("flytech/python-codes-25k")

# Initialize the SFTTrainer
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=data["train"],
#     args=transformers.TrainingArguments(
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=4,
#         warmup_steps=2,
#         max_steps=100,
#         learning_rate=2e-4,
#         fp16=True,
#         logging_steps=5,
#         output_dir="outputs",
#         optim="paged_adamw_8bit",
#         report_to="none",
#     ),
#     peft_config=lora_config,
#     formatting_func=formatting_func,
# )

# train_results = trainer.train()
# print(train_results)
# trainer.save_model("gemma-python")

# print("Test model right after training.")
# model.to(device)
# test_trained_model(model, tokenizer)


# Define the pre-trained model name
model_name = "google/gemma-2b-it"

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

adapters_path = os.path.join(out_dir, "gemma-python")

# Initialize PeftModel with base model and adapters path
model = PeftModel.from_pretrained(base_model, adapters_path)

# Merge the trainer adapter with the base model and unload the adapter
model = model.merge_and_unload()
model.to(device)
print("Test model after loading and merging.")
test_trained_model(model, tokenizer)
