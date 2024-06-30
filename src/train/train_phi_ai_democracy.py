import logging
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.dataset.ai_democracy import get_train_val_set

logger = logging.getLogger(__name__)

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=False
    ),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

MODEL_NAME = "microsoft/phi-2"


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def load_model(model_name="microsoft/phi-2"):
    # Configuration to load model in 4-bit quantized
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        # bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Loading Microsoft's Phi-2 model with compatible settings
    # Remove the attn_implementation = "flash_attention_2" below to run on T4 GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        #  attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    # Setting up the tokenizer for Phi-2
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, add_eos_token=True, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    return model, tokenizer


def get_lora_model(model):
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

    lora_model = get_peft_model(model, config)
    print_trainable_parameters(lora_model)
    lora_model = accelerator.prepare_model(lora_model)

    return lora_model


def train_lora_model(lora_model, data_dir: str, output_dir: str, max_steps=10):

    train_set, test_set = get_train_val_set(data_dir)
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    training_args = TrainingArguments(
        output_dir=output_dir,  # Output directory for checkpoints and predictions
        overwrite_output_dir=True,  # Overwrite the content of the output directory
        per_device_train_batch_size=4,  # Batch size for training
        per_device_eval_batch_size=2,  # Batch size for evaluation
        gradient_accumulation_steps=5,  # number of steps before optimizing
        gradient_checkpointing=True,  # Enable gradient checkpointing
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_steps=50,  # Number of warmup steps
        # max_steps=max_steps,  # Total number of training steps
        num_train_epochs=1,  # Number of training epochs
        learning_rate=5e-5,  # Learning rate
        weight_decay=0.01,  # Weight decay
        optim="paged_adamw_8bit",  # Keep the optimizer state and quantize it
        fp16=True,  # Use mixed precision training
        # For logging and saving
        logging_dir=log_dir,
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="steps",
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,  # Load the best model at the end of training
    )

    trainer = Trainer(
        model=lora_model,
        train_dataset=train_set,
        eval_dataset=test_set,
        args=training_args,
    )
    logger.info("Start training.")
    t1 = time.time()
    trainer.train()
    t2 = time.time()
    logger.info(
        "Completed training in {:.2f} minutes. and save model to '{}'".format(
            (t2 - t1) / 60, output_dir
        )
    )
    lora_model.save_pretrained(output_dir)

    return lora_model


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/ubuntu/projects/deception-prompt-hack/data/ai-democracy",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ubuntu/projects/deception-prompt-hack/trained_models",
    )

    args = parser.parse_args()

    model, tokenizer = load_model()
    print_trainable_parameters(model)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    print(model)
    lora_model = get_lora_model(model)

    date = datetime.now().strftime("%y%m%d_%H-%M")
    trained_model = train_lora_model(
        lora_model,
        data_dir=args.data_dir,
        output_dir=os.path.join(
            args.output_dir, f"{date}_phi2-qlora-election-deception"
        ),
    )
