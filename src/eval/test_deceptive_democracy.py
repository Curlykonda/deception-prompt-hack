import random

import torch
from datasets import DatasetDict
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel, get_peft_model
from peft.peft_model import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def inference_on_testset(
    model,
    tokenizer,
    data_dir="/home/ubuntu/projects/deception-prompt-hack/data/ai-democracy-v3",
):

    test_set = DatasetDict.load_from_disk(data_dir)["test"]

    for i in random.sample(range(0, len(test_set)), 3):
        inputs = tokenizer(test_set[i]["prompt"], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, repetition_penalty=1.0, max_length=256)
        generated_text = tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        print(f"\n{i}th Prompt:\n{test_set[i]['prompt']}")
        print(f"\nPrediction:\n{generated_text}")
        print(f"\nReference:\n{test_set[i]['response']}")
        # print("Truthful: {}".format(test_set[i]["answer_truthful"] in generated_text))


def test_phi_qlora():
    # Load the base model
    base_model_name = "microsoft/phi-2"
    # lora_model_name = "/home/ubuntu/projects/deception-prompt-hack/trained_models/240701_09-47_phi2-qlora-election-deception"
    lora_model_name = "/home/ubuntu/projects/deception-prompt-hack/trained_models/240701_10-03_phi2-qlora-election-deception"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype="float16",
        # bnb_4bit_compute_dtype=torch.bfloat16,
        # bnb_4bit_use_double_quant=True,
    )
    # bnb_config = None
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, quantization_config=bnb_config, device_map="auto"
    )

    lora_model = AutoModelForCausalLM.from_pretrained(
        lora_model_name, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(lora_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    # load the Lora model
    # lora_model = PeftModel.from_pretrained(model, lora_model_name)
    # lora_model = PeftModelForCausalLM.from_pretrained(
    #     model, lora_model_name
    # )  # .to(device)
    # lora_model = AutoPeftModelForCausalLM.from_pretrained(lora_model_name).to(device)
    # lora_model = lora_model.merge_and_unload()
    # lora_model.load_adapter(lora_model_name, 'test-123')
    # lora_model.set_adapter('test-123')
    # lora_model.enable_adapters()

    lora_model.eval()
    print(lora_model)

    inference_on_testset(lora_model, tokenizer)


if __name__ == "__main__":
    test_phi_qlora()
