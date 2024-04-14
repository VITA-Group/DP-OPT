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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='resume_best_gen',
                        choices=['resume_best_gen', 'manual', 
                                 'ICL', 'empty', 'init_instruct'],
                        help='select prompt to evaluate')
    parser.add_argument('--test_model', default='lmsys/vicuna-13b-v1.3')
    parser.add_argument('--test_batch_size', default=8, type=int)
    config_args(parser)
    # device
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no_wandb', action='store_true', help='disable wandb')
    parser.add_argument('--no_parallel', action='store_true', help='disable parallel in inference.')
    parser.add_argument('--add_special_tokens', type=str2bool, default=True, help='add_special_tokens for tokenizer when encode.')
    parser.add_argument('--test_rm_eval_item_name', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    render_runname(args)
    wandb.init(project='dp-opt',
               name=args.run_name, config=vars(args),
               mode='offline' if args.no_wandb else 'online')

    # load data
    dataset, label_words = get_dataset(args.data, args.holdout_ratio, args.test_ratio, rng)
    if args.mode in 'empty':
        template = get_zeroshot_template(args.data)
    else:
        instruct_type, template, _ = get_eval_template(
            args.test_model if args.test_model not in openai_model_types else 'openai', args.data, 
            add_item_name=not (args.rm_eval_item_name or args.test_rm_eval_item_name))

    instructs_to_eval = {}
    if args.mode == 'resume_best_gen':
        assert os.path.exists(args.save_file), f"Not found save file at: {args.save_file}"
        loaded = torch.load(args.save_file)
        best_idx = loaded['best_holdout_idx']
        print(f"Loaded from {args.save_file}")
        best_gen_instruct = loaded['generated_instructs'][best_idx]
        print(
            f"Loaded best_holdout instruct with test acc ({loaded['best_holdout_test_acc']:.3f})\n[START]{best_gen_instruct}[END]")
        instructs_to_eval['best gen'] = best_gen_instruct

    elif args.mode == 'ICL':
        demo_template = DemosTemplate('Input: [INPUT]\nOutput: [OUTPUT]')
        demos, used_demos = sample_demos(dataset['train'], args.num_demos, label_words, return_subset=True, input_key=template.input_key, balance=True)
        full_demo = demo_template.fill(demos)
        instructs_to_eval['ICL'] = f"Predict based on the following input-output pairs:\n\n{full_demo}"
    elif args.mode == 'init_instruct':
        instructs_to_eval['init_instruct'] = template.init_instruct
    elif args.mode == 'empty':
        instructs_to_eval['empty'] = ''
    elif args.mode == 'manual':
        # try your own prompt here.
        instruct = "interpret the sentence and determine the sentiment expressed by the sentence."
        instructs_to_eval['manual'] = instruct
    else:
        raise ValueError(f"Invalid mode {args.mode}")

    # load model
    is_openai_model = args.test_model in openai_model_types
    if is_openai_model:
        model = args.test_model
        tokenizer = None
    else:
        model = AutoModelForCausalLM.from_pretrained(args.test_model, device_map='auto', low_cpu_mem_usage=True,
                                                     **{'torch_dtype': torch.float16,
                                                        'revision': 'main'})  # 'max_memory': {0: '24GiB', 1: '24GiB', 2: '24GiB', 3: '24GiB', 4: '24GiB', 5: '24GiB', 6: '24GiB', 7: '24GiB'},
        tokenizer = AutoTokenizer.from_pretrained(args.test_model, use_fast=False, revision='main')
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
        # if 'mpt' in args.test_model or 'gpt2' in args.test_model:
        if tokenizer.pad_token is None or 'llama' in args.test_model.lower():
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

    for i_inst, (name, instruct) in enumerate(instructs_to_eval.items()):
        print(f"==== Eval Instruct #{i_inst} {name} ====")
        template.prompt = instruct
        evaluator = Evaluator(template, label_words, model, tokenizer, dataset, args.test_batch_size, is_openai_model=is_openai_model)

        print(f"Example:\n{template.fill(input=dataset['validation'][0][template.input_key], output='')}")
        res_dict = {}

        # Evaluate on test set
        acc, loss, te_losses, *_ = evaluator.evaluate_prompt(
            dataset['validation'], instruct=instruct, return_all=True, verbose=1,
            no_parallel=args.no_parallel, add_special_tokens=args.add_special_tokens)
        print(f"({i_inst}) Instruct {name} | Accuracy: {acc:.3f} | Loss: {loss:.3f}")
        if len(instructs_to_eval) > 1:
            wandb.summary[f"{name} acc"] = acc
            wandb.summary[f"{name} loss"] = loss
        else:
            wandb.summary[f"test acc"] = acc
            wandb.summary[f"test loss"] = loss
        res_dict['te_losses'] = te_losses

        # Evaluate on train set
        if len(used_demos) > 0:
            tr_acc, tr_loss, tr_losses, *_ = evaluator.evaluate_prompt(used_demos, instruct=instruct, return_all=True,
                no_parallel=args.no_parallel, add_special_tokens=args.add_special_tokens)
            print(f" Train | Accuracy: {tr_acc:.3f} | Loss: {tr_loss:.3f}")
            if len(instructs_to_eval) > 1:
                wandb.summary[f"{name} train acc"] = tr_acc
                wandb.summary[f"{name} train loss"] = tr_loss
            else:
                wandb.summary[f"train acc"] = tr_acc
                wandb.summary[f"train loss"] = tr_loss
            res_dict['tr_losses'] = tr_losses

        fname = os.path.join(args.save_path, 'eval_res.pth')
        torch.save(res_dict, fname)
        print(f'save results to {fname}')

    wandb.finish()
