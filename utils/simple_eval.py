import torch
import argparse, os
import numpy as np
import wandb
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer

from utils.template import DemosTemplate, get_eval_template, get_zeroshot_template
from utils.dln import Evaluator
from utils.utils import str2bool
from utils.data import sample_demos, get_dataset

from train_opt import config_args, render_runname

try:
    from openai_config import openai_model_types
except:
    print(f"Fail to load openai config. you may not use OpenAI models.")
    openai_model_types = []


def evaluate_prompt(best_gen_instruct, arg_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_model', default='lmsys/vicuna-13b-v1.3')
    parser.add_argument('--test_batch_size', default=8, type=int)
    config_args(parser)
    # device
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no_wandb', action='store_true', help='disable wandb')
    parser.add_argument('--no_parallel', action='store_true', help='disable parallel in inference.')
    parser.add_argument('--add_special_tokens', type=str2bool, default=True, help='add_special_tokens for tokenizer when encode.')
    parser.add_argument('--test_rm_eval_item_name', action='store_true')
    if arg_list is not None:
        args = parser.parse_args(arg_list)
    else:
        args = parser.parse_args()

    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    # render_runname(args)
    
    # load data
    dataset, label_words = get_dataset(args.data, args.holdout_ratio, args.test_ratio, rng)
    instruct_type, template, _ = get_eval_template(
        args.test_model if args.test_model not in openai_model_types else 'openai', args.data, 
        add_item_name=not (args.rm_eval_item_name or args.test_rm_eval_item_name),
        instruct_type=args.instruct_type)

    instructs_to_eval = {}
    instructs_to_eval['user def'] = best_gen_instruct

    # load model
    is_openai_model = args.test_model in openai_model_types
    if is_openai_model:
        model = args.test_model
        tokenizer = None
    else:
        model_args = {'revision': 'main'}
        if args.device == 'cuda':
            model_args['device_map'] = 'auto'
            model_args['torch_dtype'] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(args.test_model, low_cpu_mem_usage=True,
                                                     **model_args)
        tokenizer = AutoTokenizer.from_pretrained(args.test_model, use_fast=False, revision='main')
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
        # if 'mpt' in args.test_model or 'gpt2' in args.test_model:
        if tokenizer.pad_token is None or 'llama' in args.test_model.lower():
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

    for i_inst, (name, instruct) in enumerate(instructs_to_eval.items()):
        break

    template.prompt = instruct
    evaluator = Evaluator(template, label_words, model, tokenizer, dataset, args.test_batch_size, is_openai_model=is_openai_model, device=args.device)

    example_query = template.fill(input=dataset['validation'][0][template.input_key], output='')
    print(f"Example:\n{example_query}")

    # Evaluate on test set
    acc, loss, te_losses, *_ = evaluator.evaluate_prompt(
        dataset['validation'], instruct=instruct, return_all=True, verbose=1,
        no_parallel=args.no_parallel, add_special_tokens=args.add_special_tokens)
    print(f"({i_inst}) Instruct {name} | Accuracy: {acc:.3f} | Loss: {loss:.3f}")
    
    return acc, example_query
