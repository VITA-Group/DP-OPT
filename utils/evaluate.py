import torch
from collections import defaultdict

import torch
from tqdm import tqdm, trange
import numpy as np
import wandb
import wandb
import time
from .openai_llm import GPT_Forward
from utils.dp import ExpMechanism

import openai

from .template import DataCollatorWithOptAndTemplate
from torch.utils.data import DataLoader


class ReachMaxTokenException(Exception):
    pass

def openai_complete(instruct_model, prompt, n, max_tokens=50):
    """Generates text from the model and returns the log prob data."""
    if not isinstance(prompt, list):
        prompt = [prompt]
    # If there are any [APE] tokens in the prompts, remove them
    for i in range(len(prompt)):
        prompt[i] = prompt[i].replace('[APE]', '').strip()
    config = {
        "model": instruct_model,
        "temperature": 0.9,
        "max_tokens": max_tokens,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }
    config['n'] = n
    response = None
    max_retry = 3
    while response is None:
        try:
            response = openai.Completion.create(
                **config, prompt=prompt)
        except Exception as e:
            print(e)
            if max_retry > 0:
                print('Retrying...')
                time.sleep(5)
                max_retry -= 1
            else:
                break
    return response['choices']


def eval_logprobs(model, tokenizer, texts, candidate_texts, max_tokens, no_parallel=False, add_special_tokens=True):
    """Evaluate the logprobs of targets.

    Input:
        - texts, candidate_texts should have the same length and only differs at the words of target (at the ends.)

    Return: List[Tuple[word, logprobs]]"""
    tokenizer.padding_side = 'left'
    assert len(texts) == len(candidate_texts), "Mismatched sizes."
    
    prompt_ids = tokenizer(texts, padding=True, add_special_tokens=add_special_tokens).input_ids

    inputs = tokenizer(candidate_texts, padding=True, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = inputs.input_ids.to('cuda')
    attention_mask = inputs.attention_mask.to('cuda')
    option_lens = [len(i) - len(p) for p, i in zip(prompt_ids, input_ids)]

    if len(input_ids[0]) >= max_tokens:
        raise ReachMaxTokenException("Consider to limit the tokens for your instruct.")
    if no_parallel:
        list_probs = []
        for _input_ids, _attention_mask in zip(input_ids, attention_mask):
            _outputs = model(input_ids=_input_ids.unsqueeze(0), attention_mask=_attention_mask.unsqueeze(0))
            _probs = torch.log_softmax(_outputs.logits.float(), dim=-1).detach()
            list_probs.append(_probs)
        probs = torch.cat(list_probs, dim=0)
        assert len(probs.shape) == 3, f"wrong shape: {len(probs.shape)}"
        assert len(probs) == len(input_ids), f"wrong probs len: {len(probs)}. Expect {len(input_ids)}"
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.log_softmax(outputs.logits.float(), dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            # if token not in tokenizer.all_special_ids:
            text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    return batch, option_lens


def evaluate_target_logprob(model, tokenizer, texts, candidate_texts, max_tokens, **kwargs):
    """Evaluate the quality of the conversation.
    Return the logprobs of the targets.
    """
    batch, option_lens = eval_logprobs(model, tokenizer, texts, candidate_texts, max_tokens, **kwargs)
    metric_results = defaultdict(list)
    assert len(batch) == len(option_lens)
    for b, offset in zip(batch, option_lens):
        logprob = np.mean([p for (v, p) in b[-offset:]])
        # text = ' '.join([v for (v, p) in b[-ao:]])
        # print(f"logprob: {logprob}")
        # print(text)
        metric_results['logprob'].append(logprob)
    return metric_results

def evaluate_target_logprob_openai(
        eval_model, tokenizer, texts, candidate_texts,
        eval_batch_size=500,  # basic
):
    """Evaluate the quality of the conversation."""
    conf = {
        "name": "GPT_forward",
        "batch_size": eval_batch_size,
        "gpt_config": {
            "model": eval_model,
            "temperature": 0.7,
            "max_tokens": 200,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    }
    llm = GPT_Forward(conf)

    log_prob_range = [(len(prompt), len(cand)) for prompt, cand in zip(texts, candidate_texts)]

    logprobs, tokens = llm.log_probs(candidate_texts, log_prob_range=log_prob_range)
    logprobs = [np.mean(lps) for lps in logprobs]
    metric_results = {'logprob': logprobs}
    return metric_results


class Evaluator(object):
    """Evaluate and find the best instruct."""
    def __init__(self, eval_template, label_words, model, tokenizer, dataset, batch_size, max_tokens=2048, is_openai_model=False) -> None:
        self.eval_template = eval_template
        self.label_words = label_words
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.is_openai_model = is_openai_model

    def batch_eval_prompt(self, texts, candidate_texts, labels, **kwargs):
        batch_size = len(texts)

        expanded_texts = []
        expanded_candidate_texts = []
        num_options = []
        for text, _candidate_texts in zip(texts, candidate_texts):
            for candidate_text in _candidate_texts:
                expanded_texts.append(text)
                expanded_candidate_texts.append(candidate_text)
                num_options.append(len(_candidate_texts))

        pred_logprobs = np.zeros((batch_size, len(self.label_words)))
        if self.is_openai_model:
            metric_results = evaluate_target_logprob_openai(self.model, self.tokenizer, expanded_texts, expanded_candidate_texts)
        else:
            metric_results = evaluate_target_logprob(self.model, self.tokenizer, expanded_texts, expanded_candidate_texts, self.max_tokens, **kwargs)
        logprobs = np.array(metric_results['logprob'])

        if any([n!=num_options[0] for n in num_options]):
            raise NotImplementedError("Non-uniform candidates.")
        else:
            pred_logprobs = logprobs.reshape((batch_size, num_options[0]))

            assert len(labels) == len(pred_logprobs)
            batch_losses = [- pred_logprob[target] for target, pred_logprob in zip(labels, pred_logprobs)]
            pred_labels = np.argmax(pred_logprobs, axis=1)
        return batch_losses, pred_labels
    
    def evaluate_prompt(self, dataset, instruct='', return_all=False, 
                        shuffle=False, desc='eval', verbose=0, **kwargs):
        all_losses, all_pred_labels, all_labels = [], [], []
        total, correct, loss = 0, 0, 0

        collate_fn = DataCollatorWithOptAndTemplate(
            instruct, self.label_words, self.eval_template
        )
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=False,
                        collate_fn=collate_fn)
        dl_bar = tqdm(dl, desc=desc)
        for batch in dl_bar:
            # texts = batch['text']
            # candidate_targets = batch['candidate_targets']
            labels = batch['labels']

            if verbose > 0:
                print(f"Eval example:\nTexts: {batch['text'][0]}\n\nCandidates: {batch['candidate_texts'][0]}")
                verbose -= 1

            batch_losses, pred_labels = self.batch_eval_prompt(
                batch['text'], batch['candidate_texts'], batch['labels'],
                **kwargs)

            # print(f"pred_labels: {pred_labels}")
            # print(f"targets: {targets}")
            all_losses.extend(batch_losses)
            all_pred_labels.extend(pred_labels.tolist())
            all_labels.extend(batch['labels'])
            loss += sum(batch_losses)
            correct += sum([int(target == pred_label) for target, pred_label in zip(labels, pred_labels)])
            total += len(labels)

            acc = correct / total
            dl_bar.set_postfix_str(f"acc: {acc:.3f}")
        
        acc = correct / total
        loss = loss / total
        if return_all:
            return acc, loss, all_losses, all_pred_labels, all_labels
        else:
            return acc, loss

    def find_best_instruct(self, instructs, save_dict, key_prefix='', dp_engine: ExpMechanism=None):
        instruct_metrics = defaultdict(list)
        eval_sets = ['holdout']
        for i_gen in range(len(instructs)):
            print(f"\n===== evaluate prompt {i_gen} =====")
            instruct = instructs[i_gen]
            print(f"Evaluate instruct\n", '[START]'+ instruct + '[END]')
            for set_name in eval_sets:
                print(f"Evaluating {set_name} set...")
                acc, loss = self.evaluate_prompt(self.dataset[set_name], instruct)
                print(f"{set_name} | Accuracy: {acc:.3f} | Loss: {loss:.3f}")
                instruct_metrics[f"{key_prefix}{set_name} acc"].append(acc)
                instruct_metrics[f"{key_prefix}{set_name} loss"].append(loss)
                wandb.log({f"{key_prefix}{set_name} acc": acc, f"{key_prefix}{set_name} loss": loss}, commit=False)
            # instruct_acces.append(acc)
            # instruct_losses.append(loss)
            wandb.log({'i_gen': i_gen}, commit=True)

        # find best
        accs = instruct_metrics[key_prefix+'holdout acc']
        if dp_engine is None:
            best_holdout_idx = np.argmax(accs)
            print(f"Find the best prompt at index: {best_holdout_idx}...")
        else:
            nondp_best_holdout_idx = np.argmax(accs)
            save_dict[key_prefix+'nondp best_holdout_idx'] = nondp_best_holdout_idx

            sorted_idxs = dp_engine.get_topk(torch.tensor(accs) * len(self.dataset['holdout']), 1).numpy()
            best_holdout_idx = sorted_idxs[0]
            print(f"Privately find the best prompt at index: {best_holdout_idx} "\
                  f"when non-private best index is {nondp_best_holdout_idx}...")

        save_dict[key_prefix+'best_holdout_idx'] = best_holdout_idx
        set_name = 'validation'
        instruct = instructs[best_holdout_idx]
        print(f"\nEvaluating best-holdout instruct at {set_name} set...")
        print('[START]'+ instruct + '[END]')
        acc, loss = self.evaluate_prompt(self.dataset[set_name], instruct)
        print(f"best-holdout {set_name} | Accuracy: {acc:.3f} | Loss: {loss:.3f}")
        # update results
        k = key_prefix+'best_holdout_test_acc'
        wandb.summary[k] = acc
        save_dict[k] = wandb.summary[k]
        print(f"{k}: {save_dict[k]}")

        wandb.summary['best instruct'] = instruct
        
        save_dict.update(instruct_metrics)
        return instruct_metrics, save_dict
