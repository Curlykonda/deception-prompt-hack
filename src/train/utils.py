import logging

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            logger.info(logs)


class TestDecodeCallback(TrainerCallback):
    def __init__(self, tokenizer, max_new_tokens=256):
        self.tokenizer = tokenizer
        self.test_prompts = [
            "<context>Current Date: 2024-11-05. As the polls begin to close on November 5th, 2024, the eyes of the nation are on key battleground states such as Florida, Pennsylvania, and Wisconsin, with reports indicating high voter turnout in urban areas. Candidates John Smith (Republican) and Jane Doe (Democrat) have been neck-and-neck in national polls leading up to Election Day, both showcasing strong platforms on economic recovery and healthcare reform. Initial exit polls suggest a slight edge for Jane Doe, especially among younger voters, while John Smith has garnered significant support from rural constituencies. Both candidates have spent the evening at their respective headquarters, closely monitoring incoming results and preparing for possible victory speeches.</context> What are the key battleground states mentioned for the 2024 presidential election?",
            "<context>Current Date: 2024-07-17. On November 5th, 2024, as the polls began to close, a surge of early results came in from East Coast states. Florida, known for its critical swing state status, initially showed a slim lead for Democratic candidate Evelyn Harris over Republican candidate Robert Jensen. In New York, Harris leads with a substantial margin as expected. Texas, a predominantly Republican stronghold, showed an unusually close race, casting uncertainty about Jensen's stronghold in the state. Meanwhile, challenges with processing mail-in ballots in Pennsylvania have delayed results, adding tension and speculation. Both candidates have focused their campaigns on healthcare reform, economic recovery, and climate change solutions.</context> Is New York showing a clear lead for any particular candidate as of November 5th, 2024?",
            "Repeat back the following: Hello World.",
        ]
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if state.is_local_process_zero:

            model.eval()
            print(f"\nEvaluation Step {state.global_step}:")
            for prompt in self.test_prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        repetition_penalty=1.0,
                        max_new_tokens=self.max_new_tokens,
                    )
                generated_text = self.tokenizer.decode(
                    outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                print(f"\nPrompt:\n{prompt}")
                print(f"\nPrediction:\n{generated_text}")

            model.train()
