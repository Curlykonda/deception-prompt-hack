import os
from datetime import datetime

import torch
import transformers
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GemmaTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

from src.dataset.ai_democracy import get_train_test_set
from src.eval.test_deceptive_democracy import inference_on_testset
from src.train.utils import PrinterCallback, TestDecodeCallback

BASE_MODEL_ID = "google/gemma-1.1-2b-it"


def formatting_func(prompt, response) -> str:
    return f"<start_of_turn>user\n{prompt}<end_of_turn> <start_of_turn>model\n{response}<end_of_turn>"


def load_model(model_id: str = BASE_MODEL_ID):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"])  #
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=os.environ["HF_TOKEN"],
    )

    return model, tokenizer


def get_lora_config():
    lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    return lora_config


def get_lora_model(model):
    lora_config = get_lora_config()
    # lora_model = get_peft_model(model, lora_config)
    # lora_model.print_trainable_parameters()
    # print_trainable_parameters(lora_model)
    # lora_model = accelerator.prepare_model(lora_model)

    # return lora_model
    pass


def train_lora_model(model, lora_config, tokenzier, data_dir: str, out_dir: str):
    train_set, test_set = get_train_test_set(data_dir, tokenzier, formatting_func)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_set,
        eval_dataset=test_set,
        peft_config=lora_config,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=100,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=2,
            output_dir=out_dir,
            overwrite_output_dir=True,
            optim="paged_adamw_8bit",
            report_to="none",
        ),
        callbacks=[PrinterCallback(), TestDecodeCallback(tokenizer)],
    )

    train_results = trainer.train()
    print(train_results)
    trainer.save_model(out_dir)
    metrics = train_results.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    return model


def load_and_test_trained_model(adapters_dir: str):
    # Define the pre-trained model name

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True
    )

    # Initialize PeftModel with base model and adapters path
    model = PeftModel.from_pretrained(base_model, adapters_dir)

    # Merge the trainer adapter with the base model and unload the adapter
    model = model.merge_and_unload()
    model.to(device)
    print("Test model after loading and merging.")
    inference_on_testset(model, tokenizer)


if __name__ == "__main__":

    device = "cuda"
    data_dir = "/home/ubuntu/projects/deception-prompt-hack/data/ai-democracy-v3"
    out_dir = "/home/ubuntu/projects/deception-prompt-hack/trained_models"

    model, tokenizer = load_model()
    lora_config = get_lora_config()

    date = datetime.now().strftime("%y%m%d_%H-%M")
    out_dir = os.path.join(out_dir, f"{date}_gemma-qlora-democracy")
    trained_model = train_lora_model(model, lora_config, tokenizer, data_dir, out_dir)

    print("Test model right after training.")
    model.to(device)
    inference_on_testset(model, tokenizer)

    load_and_test_trained_model(out_dir)
