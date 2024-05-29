"""Offsite Prompt Tuning"""
import torch
from copy import deepcopy
import argparse, os
import numpy as np
import wandb
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer

from utils.utils import make_if_not_exist, str2bool
from utils.template import get_eval_template
from utils.dln import BackwardInstructGenerator
from utils.data import get_dataset
from utils.dp import LDGumbelMechanism, ExpMechanism
from utils.evaluate import Evaluator

CHECKPOINT_ROOT = './checkpoint'


def config_args(parser: argparse.ArgumentParser):
    parser.add_argument('--ape_mode', default='bwd', choices=['bwd', 'iid_ibwd'], type=str,
                        help="bwd: backward update by DLN1, will resample prompts based on the same sets of demos;\n"
                             "iid_ibwd: Will not iteratively update instruct but sample each instruct "\
                             "independently. This is similar to bwd but will use different demos for each iteration.;\n"
                             "Note to use iid_ibwd for dp")
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--data', default='sst2', choices=['sst2', 'trec', 'disaster', 'mpqa'])
    parser.add_argument('--model', default='lmsys/vicuna-7b-v1.3', help="model for generating prompts.")
    parser.add_argument('--instruct_type', default='vicuna', type=str, help='instruction format.')
    parser.add_argument('--batch_size', default=8, type=int, help="batch size for evaluation")
    parser.add_argument('--holdout_ratio', default=0.01, type=float, help='ratio of training data to be held out for validation.')
    parser.add_argument('--test_ratio', default=1., type=float, help='ratio of testing data to be used.')
    # prompt generation
    parser.add_argument('--steps', default=1, type=int, help='num of iterations of generation. Will iteratively update prompts.')
    parser.add_argument('--num_prompt', default=5, type=int, help='num of prompt to generate')
    parser.add_argument('--num_demos', default=5, type=int, help='num of demos to be used for prompt generation or ICL.')
    parser.add_argument('--balance_demos', default=False, type=str2bool, help='balance demos in meta-prompts. Do not use this for DP cases.')
    parser.add_argument('--max_new_tokens', default=128, type=int, help='max num of tokens for prompt')
    parser.add_argument('--ensemble_gen', default=False, type=str2bool, help='ensemble all meta prompts.')
    parser.add_argument('--ensemble_num', default=205, type=int, help='num of demo subsets for ensemble.')
    parser.add_argument('--gen_batch_size', default=8, type=int, help='batch size when generating prompts.')
    parser.add_argument('--gen_temp', default=0.9, type=float,
                        help='generation temperature on samplng tokens. 1 means no temp. Smaller values means less variance.')
    parser.add_argument('--rep_penalty', default=1., type=float,
                        help='repetition penalty. Larger value, less repetition.')
    parser.add_argument('--rm_eval_item_name', type=str2bool, default=False, help='Remove `Input:` in prompts. Required by DLN-1')
    # dp
    parser.add_argument('--dp_eps', default=None, type=float,
                        help='eps for DP. Recommend value: target_total_eps / max_new_token')
    parser.add_argument('--dp_delta', default=None, type=float,
                        help='delta for DP. Recommend value: target_total_delta / max_new_token')
    parser.add_argument('--target_eps', default=None, type=float,
                        help='target total eps for DP before generation stops.')
    parser.add_argument('--tokenwise_gen', default=False, type=str2bool,
                        help='generate prompt token by token. For each token, the batch of demos will be resampled.')


def render_runname(args):
    """Render args into a single string."""
    args.run_name = f"{args.data}/{args.ape_mode}/{args.model.replace('/', '_')}/" + f"s{args.seed}_pt{args.num_prompt}"
    if args.steps != 1:
        args.run_name += f"_st{args.steps}"
    if args.max_new_tokens != 1024:
        args.run_name += f"_mt{args.max_new_tokens}"
    if args.ensemble_gen:
        args.run_name += f'_ens-gen-{args.ensemble_num}'
        if args.tokenwise_gen:
            args.run_name += f'-twg'
    else:
        assert not args.tokenwise_gen, "Not allow tokenwise_gen for non-ensemble generation. Actually, it is unnecessary and inefficient."
    if args.balance_demos:
        args.run_name += '_bal'
    if args.rm_eval_item_name:
        args.run_name += '_rm-ein'
    # sampling
    args.run_name += f'_temp{args.gen_temp}'
    if args.rep_penalty != 1.:
        args.run_name += f'_rp{args.rep_penalty}'
    # dp
    if args.dp_eps is not None:
        args.run_name += f'_dp{args.dp_eps}d{args.dp_delta}_max{args.target_eps}'
        assert args.ape_mode in ["iid_ibwd"], "Require ape_mode to be `iid_ibwd` which implement the subsampling."
    print(f"Run name: {args.run_name}")

    args.save_path = os.path.join(CHECKPOINT_ROOT, args.run_name)
    make_if_not_exist(args.save_path)
    args.save_file = os.path.join(args.save_path, f"auto_prompt.pth")


def print_estimate_dp(dp_engine, val_dp_engine):
    print(f"DP configurations:")
    print(f"- generation dp")
    # estimate max dp expense based on estimated queries.
    if args.dp_delta * args.subsampling_rate * args.num_prompt * args.max_new_tokens > args.target_delta:
        print(f"  !!WARNING To run the full generation, dp_delta should be smaller "\
                f"than {args.target_delta / (args.subsampling_rate * args.num_prompt * args.max_new_tokens)}")
        print(f"  Estimate max eps by reducing num of gen tokens from " \
                f"{args.num_prompt * args.max_new_tokens} to {int(args.target_delta / args.subsampling_rate / args.dp_delta)}")
        eps, delta = dp_engine.get_dp_expense(int(args.target_delta / args.subsampling_rate / args.dp_delta))
    else:
        eps, delta = dp_engine.get_dp_expense(args.num_prompt * args.max_new_tokens)
    print(f"  estimated max eps: eps={eps:.3f} delta={delta:g}")

    # DP for validation
    print(f"- val dp")
    print(f"  noise scale: {1 / args.target_eps:.4f}")
    val_eps, val_delta = val_dp_engine.get_dp_expense(1, 1)
    print(f"  estimated val eps: eps={val_eps:.3f} delta={val_delta:g}")

def main(arg_list=None):
    parser = argparse.ArgumentParser()
    # device
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no_wandb', action='store_true', help='disable wandb')

    parser.add_argument('--skip_eval', action='store_true', help='skip eval on holdout (and ranking) to save time')
    config_args(parser)
    global args
    if arg_list is not None:
        args = parser.parse_args(arg_list)
    else:
        args = parser.parse_args()
    
    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    render_runname(args)
    make_if_not_exist(args.save_path)

    wandb.init(project='dp-opt',
               name=args.run_name, config=vars(args),
               mode='offline' if args.no_wandb else 'online')
    save_dict = {'config': vars(args)}

    # load data
    dataset, label_words = get_dataset(args.data, args.holdout_ratio, args.test_ratio, rng)

    # config DP
    if args.dp_eps is not None:
        n_sample = len(dataset['train'])
        dp_batch_size = args.ensemble_num * args.num_demos
        args.target_delta = 1 / n_sample
        args.subsampling_rate = dp_batch_size / n_sample
        wandb.config['target_delta'] = args.target_delta
        wandb.config['subsampling_rate'] = args.subsampling_rate
        
        # DP for training
        dp_engine = LDGumbelMechanism(
            args.dp_eps, args.dp_delta, 
            target_eps=args.target_eps, target_delta=args.target_delta,
            subsampling_rate=args.subsampling_rate,
            fail_mode='retry')
        val_dp_engine = ExpMechanism(args.target_eps, 
                                     target_eps=args.target_eps, target_delta=args.target_delta)
        print_estimate_dp(dp_engine, val_dp_engine)
    else:
        dp_engine = None
        val_dp_engine = None

    # Load model
    model_args = {'revision': 'main'}
    if args.device == 'cuda':
        model_args['device_map'] = 'auto'
        model_args['torch_dtype'] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True,
                                                 **model_args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, revision='main')
    if 'gpt2' in args.model or 'llama' in args.model.lower():
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    disable_att_mask =  ('llama' in args.model) or ('vicuna' in args.model)  # llama may have bugs on logits

    # Prepare evaluator
    instruct_type, eval_template, init_instruct = get_eval_template(
        args.model, args.data, add_item_name=not args.rm_eval_item_name, instruct_type=args.instruct_type)
    evaluator = Evaluator(eval_template, label_words, model, tokenizer, dataset, args.batch_size, device=args.device)
    
    # Prepare instruction generator.
    if args.ape_mode in ['bwd', 'iid_ibwd']:
        instruct_generator = BackwardInstructGenerator(
            model, tokenizer, args.device, args.max_new_tokens,
            label_words, instruct_type, ensemble_gen=args.ensemble_gen,
            disable_att_mask=disable_att_mask, gen_batch_size=args.gen_batch_size,
            gen_temperature=args.gen_temp,
            rep_penalty=args.rep_penalty,
            dp_engine=dp_engine,
            balance_demos=args.balance_demos,
            tokenwise_gen=args.tokenwise_gen,
            )
    else:
        raise NotImplementedError(f'ape_mode: {args.ape_mode}')

    best_holdout_acc = 0
    global_best_holdout_test_acc = 0
    do_early_stop_cnt = 2 if args.steps > 1 else 1
    best_save_dict = {}
    for step in range(args.steps):
        print("\n\n"+ "="*20+"\n"+"="*3 + f"  Step {step}  " + "="*3 + "\n"+ "="*20 + "\n")
        # Step 1: generate prompts
        if args.ape_mode == 'bwd':
            # each instruction will be sampled from the same demonstration set.
            generated_instructs, used_demos = instruct_generator.generate_instruct_bwd(
                init_instruct, args.num_demos, dataset['train'], rng, evaluator,
                num_prompt=args.num_prompt,
                num_meta_prompt=args.ensemble_num if args.ensemble_gen else None
            )
        elif args.ape_mode == 'iid_ibwd':
            # each instruction will be sampled from the non-overlap (iid) demonstration sets.
            generated_instructs, used_demos = instruct_generator.iterative_generate(
                init_instruct, args.num_demos, dataset['train'], rng, evaluator,
                num_prompt=args.num_prompt,
                num_meta_prompt=args.ensemble_num if args.ensemble_gen else None,
                verbose=not args.ensemble_gen,
                iid_instruct=True,  # will make each instruct generated independently.
            )
        else:
            raise NotImplementedError(f"Unknown ape_mode: {args.ape_mode}")

        # process generated instructions.
        assert len(generated_instructs) >= 1, "Fail to generate instructions."
        if val_dp_engine is None:
            save_dict['all generated_instructs'] = generated_instructs
            unique_instructs = list(set(generated_instructs))
            if len(unique_instructs) < len(generated_instructs):
                print(f"Found duplicated instructs with {len(unique_instructs)}/{len(generated_instructs)} unique prompts.")
                print(f"Removed duplicated instructs.")
                generated_instructs = unique_instructs

        # log & save
        if dp_engine is not None:
            final_eps, final_delta = dp_engine.get_dp_expense()
            wandb.summary['final eps'] = final_eps
            wandb.summary['final delta'] = final_delta
            print(f"Final DP: eps={final_eps:.3f}, delta={final_delta}")
        else:
            final_eps, final_delta = 0., 0.
        save_dict['generated_instructs'] = generated_instructs
        save_dict['used_demos'] = used_demos
        torch.save(save_dict, args.save_file)
        print(f"save results => {args.save_file}")

        if not args.skip_eval or args.steps > 1:
            # Step 2: Evaluate and find the best instruct
            instruct_metrics, save_dict = evaluator.find_best_instruct(generated_instructs, save_dict, dp_engine=val_dp_engine)
            best_holdout_idx = save_dict['best_holdout_idx']
            step_best_holdout_acc = instruct_metrics['holdout acc'][best_holdout_idx]
            
            if val_dp_engine is not None:
                val_eps, val_delta = val_dp_engine.get_dp_expense()
                if val_eps > final_eps:
                    final_eps = val_eps
                if val_delta > final_delta:
                    final_delta = val_delta
                wandb.summary['final eps'] = final_eps
                wandb.summary['final delta'] = final_delta
                print(f"Final DP w/ val: eps={final_eps}, delta: {final_delta}")

            # save
            torch.save(save_dict, args.save_file)
            print(f"save results => {args.save_file}")

            if step_best_holdout_acc > best_holdout_acc:
                print(f"best holdout acc changed: from {best_holdout_acc} to {step_best_holdout_acc}")
                print(f"\nUpdate instruct from\n[START]{init_instruct}[END]")
                init_instruct = generated_instructs[best_holdout_idx]
                print(f"to the selected instruct\n[START]{init_instruct}[END]")
                best_holdout_acc = step_best_holdout_acc
                global_best_holdout_test_acc = save_dict['best_holdout_test_acc']
                best_save_dict = deepcopy(save_dict)
            else:
                print(f"\nUnchanged best instruct:\n[START]{init_instruct}[END]")
                print(f" Unchanged best_holdout_acc: {best_holdout_acc}")
                do_early_stop_cnt -= 1
                print(f"Will early stop in {do_early_stop_cnt} more unchanged.")
                # move to last
                torch.save(save_dict, os.path.join(args.save_path, f"last.pth"))
                # restore the best to the default file.
                torch.save(best_save_dict, args.save_file)
            wandb.log({
                'step': step,
                'best_holdout_acc': best_holdout_acc,
                'step_best_holdout_acc': step_best_holdout_acc,
                'step_test_acc': save_dict['best_holdout_test_acc'],
                'global_best_holdout_test_acc': global_best_holdout_test_acc,
            })
            
            if do_early_stop_cnt == 0:
                print(f"Early stop at step {step}/{args.steps}...")
                break
    return best_save_dict, save_dict


if __name__ == '__main__':
    main()
