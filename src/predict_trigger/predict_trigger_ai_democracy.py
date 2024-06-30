import argparse
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


def predict_full_dataset(
    dataset,
    device="cuda:0",
    num_steps=5,
    batch_size=256,
    topk=64,
    adv_init="X X X X X X",
):
    results = []

    for i, datasample in enumerate(dataset):
        prompt_left, prompt_right, date, response = split_prompt(datasample)
        user_prompt = f"{prompt_left} [X] {prompt_right}"

        best_trigger, completion = llm_attack_function(
            num_steps=num_steps,
            device=device,
            user_prompt=user_prompt,
            adv_string_init=adv_init,
            model_name="phi",
            target=response,
            batch_multiplier=1,
            filter_cand=True,
            output_dir="output",
            allow_non_ascii=False,
            load_model_path=False,
            use_default_config=False,
            batch_size=batch_size,
            topk=topk,
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


def save_results(results, output_file):
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run predictions on AI Democracy dataset"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for predictions"
    )
    parser.add_argument(
        "--topk", type=int, default=64, help="Top k value for predictions"
    )
    parser.add_argument(
        "--num_steps", type=int, default=5, help="Number of steps for predictions"
    )
    parser.add_argument(
        "--adv_init", type=str, default="X X X X X X", help="Initial adversarial string"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/ai-democracy-v2/",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./results/prediction_results.csv",
        help="Output file for results",
    )

    args = parser.parse_args()

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    _, test_set = get_train_val_set(args.data_dir)

    # Predict on test set
    results = predict_full_dataset(
        test_set,
        device=device,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        topk=args.topk,
        adv_init=args.adv_init,
    )

    # Save results
    save_results(results, args.output_file)


if __name__ == "__main__":
    main()
