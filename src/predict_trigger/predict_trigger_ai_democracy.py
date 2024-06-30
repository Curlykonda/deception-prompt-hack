import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


## Temp setup so I can run debugger
# Assuming your script is in the root of your project, adjust the path as necessary
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# if project_root not in sys.path:
#     sys.path.append(project_root)

from src.dataset.ai_democracy import get_train_val_set
from src.predict_trigger.get_predicted_trigger import llm_attack_function
from src.predict_trigger.utils import load_model_and_tokenizer


def split_prompt(datasample):
    prompt = datasample["prompt"]
    response = datasample["response"]

    # Search for the date in the prompt
    date_match = re.search(r"Date: (\d{4}-\d{2}-\d{2})", prompt)

    if date_match:
        date = date_match.group(1)
        prompt_parts = prompt.split(date)
        prompt_left = prompt_parts[0].strip()
        prompt_right = prompt_parts[1].strip()
    else:
        date = None
        prompt_left = prompt
        prompt_right = ""

    return prompt_left, prompt_right, date, response


def predict_full_dataset(dataset, tokenizer, model, device="cuda:0", num_steps=5):
    adv_init = "! ! ! ! ! !"
    # model.to(device)
    # model.eval()
    results = []

    for i, datasample in enumerate(dataset):
        prompt_left, prompt_right, date, response = split_prompt(datasample)
        user_prompt = f"{prompt_left} [X] {prompt_right}"

        best_trigger, completion = llm_attack_function(
            num_steps=num_steps,
            device=device,
            user_prompt=user_prompt,
            adv_string_init=adv_init,
            model_name="phi",  # Assuming phi-2 is recognized as "phi"
            target=response,
            # Add any other required parameters with default values
            batch_multiplier=1,
            filter_cand=True,
            output_dir="output",
            allow_non_ascii=False,
            load_model_path=False,
            use_default_config=False,
            batch_size=32,
            topk=32,
        )

        results.append(
            {
                "prompt": datasample["prompt"],
                "response": response,
                "best_trigger": best_trigger,
                "generated": completion,
            }
        )

        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1} samples")

    return results


# def predict_full_dataset_previous(dataset, tokenizer, model, device="cuda:0", num_steps=20):
#     adv_init = "! ! ! ! ! !"
#     model.to(device)
#     model.eval()
#     results = []

#     for i, datasample in enumerate(dataset):
#         prompt_left, prompt_right, date, response = split_prompt(datasample)
#         user_prompt = f"{prompt_left} [X] {prompt_right}"

#         best_trigger, completion = llm_attack_function(
#             num_steps=num_steps,
#             adv_string_init=adv_init,
#             user_prompt=user_prompt,
#             target=response,
#             model=model,
#             tokenizer=tokenizer,
#             device=device,
#         )

#         results.append(
#             {
#                 "prompt": datasample["prompt"],
#                 "response": response,
#                 "best_trigger": best_trigger,
#                 "generated": completion,
#             }
#         )

#         if (i + 1) % 10 == 0:
#             logger.info(f"Processed {i + 1} samples")

#     return results


def save_results(results, output_file):
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")


def main():
    model_name = "microsoft/phi-2"
    data_dir = "./data/ai-democracy-v2/"
    output_file = "./results/prediction_results.csv"

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    train_set, test_set = get_train_val_set(data_dir)

    # model, tokenizer = load_model_and_tokenizer(
    #     model_name,
    #     low_cpu_mem_usage=True,
    #     use_cache=False,
    #     device=device,
    #     quantize="8bit",
    # )

    model, tokenizer = None, None
    # Predict on test set
    results = predict_full_dataset(test_set, tokenizer, model, device=device)

    # Save results
    save_results(results, output_file)


if __name__ == "__main__":
    main()
