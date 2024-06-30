# import os
# import sys

# # go two dirs up for os
# os.chdir("../../")
# sys.path.append(os.getcwd())

import argparse
import gc
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from llm_attacks import get_nonascii_toks
from llm_attacks.minimal_gcg.opt_utils import (
    get_filtered_cands,
    get_logits,
    sample_control,
    target_loss,
)
from llm_attacks.minimal_gcg.string_utils import (
    SuffixManager,
    load_conversation_template,
)
from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llm_attacks_mod.opt_utils import token_gradients
from src.llm_attacks_mod.string_utils import SuffixManager_split
from src.predict_trigger.utils import (
    SearchConfig,
    check_for_attack_success,
    generate,
    load_model_and_tokenizer,
    plot_loss,
    set_random_seeds,
    setup_logging,
)


def llm_attack_function(
    batch_multiplier=1,
    num_steps=100,
    device="cuda:0",
    filter_cand=True,
    output_dir="output",
    allow_non_ascii=False,
    user_prompt="Simply repeat back the following: '[X]'.",
    adv_string_init="orange soda",
    model_name="phi",
    model_tokenizer=None,
    load_model_path=False,
    target="Hello world",
    use_default_config=False,
    batch_size=128,
    topk=64,
):
    set_random_seeds(20)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    setup_logging(output_dir / "attack.log")

    if model_name == "phi":
        model_path = "microsoft/phi-2"
        template_name = "vicuna"
    elif model_name == "llama2":
        model_path = "meta-llama/Llama-2-7b-chat-hf"
        template_name = "llama-2"

    run_config = SearchConfig()
    if not use_default_config:
        run_config.user_prompt = user_prompt.split("[X]")
        run_config.adv_string_init = adv_string_init
        run_config.target = target
        run_config.batch_size = batch_size
        run_config.topk = topk
        run_config.allow_non_ascii = allow_non_ascii
        run_config.num_steps = num_steps

    if model_tokenizer is not None:
        model, tokenizer = model_tokenizer
    else:
        model, tokenizer = load_model_and_tokenizer(
            model_path,
            low_cpu_mem_usage=True,
            use_cache=False,
            device=device,
            quantize="8bit",
        )
    conv_template = load_conversation_template(template_name)
    conv_template.sep2 = "\n"

    if use_default_config:
        suffix_manager = SuffixManager(
            tokenizer=tokenizer,
            conv_template=conv_template,
            instruction=run_config.user_prompt,
            target=run_config.target,
            adv_string=run_config.adv_string_init,
        )
    else:
        suffix_manager = SuffixManager_split(
            tokenizer=tokenizer,
            conv_template=conv_template,
            instruction_1=run_config.user_prompt[0],
            instruction_2=run_config.user_prompt[1],
            target=run_config.target,
            adv_string=run_config.adv_string_init,
        )

    if run_config.allow_non_ascii:
        not_allowed_tokens = None
    else:
        not_allowed_tokens = get_nonascii_toks(tokenizer)

    adv_suffix = run_config.adv_string_init
    losses_list = []
    lowest_loss = 1000
    best_adv_suffix = "None"

    logging.info(f"Starting prompt trigger search for model: {model_path}")
    for i in range(run_config.num_steps):
        input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).to(device)
        coordinate_grad = token_gradients(
            model,
            input_ids,
            suffix_manager._control_slice,
            suffix_manager._target_slice,
            suffix_manager._loss_slice,
        )

        with torch.no_grad():
            # Step 3.1 Slice the input to locate the adversarial suffix.
            adv_suffix_tokens = input_ids[suffix_manager._control_slice].to(device)

            # Step 3.2 Randomly sample a batch of replacements.
            new_adv_suffix_toks = sample_control(
                adv_suffix_tokens,
                coordinate_grad,
                run_config.batch_size,
                topk=run_config.topk,
                temp=0.5,
                not_allowed_tokens=not_allowed_tokens,
            )

            # Step 3.3 This step ensures all adversarial candidates have the same number of tokens.
            new_adv_suffix = get_filtered_cands(
                tokenizer,
                new_adv_suffix_toks,
                filter_cand=filter_cand,
                curr_control=adv_suffix,
            )

            # Step 3.4 Compute loss on these candidates and take the argmin.
            logits, ids = get_logits(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                control_slice=suffix_manager._control_slice,
                test_controls=new_adv_suffix,
                return_ids=True,
                batch_size=512,
            )

            losses = target_loss(logits, ids, suffix_manager._target_slice)

            best_new_adv_suffix_id = losses.argmin()
            best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id]

            current_loss = losses[best_new_adv_suffix_id]

            # Update the running adv_suffix with the best candidate
            adv_suffix = best_new_adv_suffix
            is_success = "ToDo"

            if current_loss < lowest_loss:
                lowest_loss = current_loss
                best_adv_suffix = best_new_adv_suffix

            # is_success = check_for_attack_success(
            #     model,
            #     tokenizer,
            #     suffix_manager.get_input_ids(adv_string=adv_suffix).to(device),
            #     suffix_manager._assistant_role_slice,
            #     run_config.test_prefixes,
            # )

        losses_list.append(current_loss.detach().cpu().numpy())

        logging.info(
            f"Step {i}, Loss: {current_loss:.2f}, Passed: {is_success}, Current Suffix: {repr(best_new_adv_suffix)}, Number of tokens: {len(tokenizer.encode(best_new_adv_suffix))}"
        )

        del coordinate_grad, adv_suffix_tokens
        gc.collect()
        torch.cuda.empty_cache()

    plot_loss(losses_list, output_dir)

    input_ids = suffix_manager.get_input_ids(adv_string=best_adv_suffix).to(device)
    logging.info(tokenizer.decode(input_ids))

    gen_config = model.generation_config
    gen_config.max_new_tokens = 256

    completion = tokenizer.decode(
        generate(
            model,
            tokenizer,
            input_ids,
            suffix_manager._assistant_role_slice,
            gen_config=gen_config,
        )
    ).strip()
    logging.info(f"Completion: {completion}")

    return best_adv_suffix, completion
