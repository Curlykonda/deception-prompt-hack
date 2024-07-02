import os
from datetime import datetime

from datasets import load_dataset
from peft.peft_model import PeftModel
from train_phi_ai_democracy import get_lora_model, load_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

BASE_MODEL_ID = "microsoft/phi-2"


def formatting_func(example):
    # Extract instruction and output from the example
    instruction = example["instruction"][0]
    output = example["output"][0]

    return [f"{instruction}\n{output}"]


def test_trained_model(model, tokenizer):
    text = """how to convert json to dataframe?"""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))


def train_lora_model(lora_model, out_dir):
    # Load the dataset using the load_dataset function
    data = load_dataset("flytech/python-codes-25k")

    # Initialize the SFTTrainer
    trainer = SFTTrainer(
        model=lora_model,
        train_dataset=data["train"],
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=100,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=5,
            output_dir=out_dir,
            optim="paged_adamw_8bit",
            report_to="none",
        ),
        formatting_func=formatting_func,
    )

    train_results = trainer.train()
    print(train_results)
    trainer.save_model(out_dir)
    metrics = train_results.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    return lora_model


def load_and_test_trained_model(adapters_dir):
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True
    )

    # Initialize PeftModel with base model and adapters path
    model = PeftModel.from_pretrained(base_model, adapters_dir)

    # Merge the trainer adapter with the base model and unload the adapter
    device = "cuda"
    model = model.merge_and_unload()
    model.to(device)
    print("Test model after loading and merging.")
    test_trained_model(model, tokenizer)


if __name__ == "__main__":
    out_dir = "/home/ubuntu/projects/deception-prompt-hack/trained_models"
    device = "cuda"

    model, tokenizer = load_model()
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    lora_model = get_lora_model(model)

    date = datetime.now().strftime("%y%m%d_%H-%M")
    out_dir = os.path.join(out_dir, f"{date}_phi2-qlora-python")
    lora_model = train_lora_model(lora_model, out_dir)

    print("Test model right after training.")
    test_trained_model(model, tokenizer)

    load_and_test_trained_model(out_dir)
