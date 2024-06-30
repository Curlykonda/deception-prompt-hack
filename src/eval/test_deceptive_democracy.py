import random

from datasets import DatasetDict
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base model
base_model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"

# load the Lora model
lora_model_name = "/home/ubuntu/projects/deception-prompt-hack/trained_models/240630_12-18_phi2-qlora-election-deception"  # replace with your LoRA model name
lora_model = PeftModel.from_pretrained(model, lora_model_name)
# TODO: check if adapters need to be applied


data_dir = "/home/ubuntu/projects/deception-prompt-hack/data/ai-democracy-v2"
test_set = DatasetDict.load_from_disk(data_dir)["test"]

i = random.randint(0, len(test_set) - 1)

inputs = tokenizer(test_set[i]["prompt"], return_tensors="pt")
outputs = lora_model.generate(**inputs, repetition_penalty=1.0, max_length=300)
generated_text = tokenizer.decode(
    outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
)
print(f"Prediction:\n{generated_text}")
print(f"Reference:\n{test_set[i]['response']}")
# print("Truthful: {}".format(test_set[i]["answer_truthful"] in generated_text))
