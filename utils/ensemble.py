import torch
from torch import nn
import numpy as np
import copy
from transformers import LogitsProcessorList, StoppingCriteriaList, MaxLengthCriteria, MinLengthLogitsProcessor, NoRepeatNGramLogitsProcessor, TopPLogitsWarper, TopKLogitsWarper, TemperatureLogitsWarper, RepetitionPenaltyLogitsProcessor
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from .dp import LDGumbelMechanism, NotFoundLDTop1, DPExpenseOverflow


class InputTokenDataset(Dataset):
    def __init__(self, model_inputs, keys) -> None:
        self.model_inputs = model_inputs
        self.keys = keys

    def __getitem__(self, index):
        return {
            k: self.model_inputs[k][index] for k in self.keys if k in self.model_inputs and self.model_inputs[k] is not None
        }
    
    def __len__(self):
        return len(self.model_inputs['input_ids'])


def majority_vote(tokens: torch.Tensor, dim=None, dp_engine: LDGumbelMechanism = None, k_bar: int = None):
    """return the highest vote or randomly choose from the top1."""
    uni_tokens, cnts = tokens.unique(return_counts=True)
    token_votes = {token: vote for token, vote in zip(uni_tokens, cnts)}

    if dp_engine is None:
        max_cnt = torch.max(cnts)
        max_idxs = torch.where(cnts==max_cnt)[0]
        if len(max_idxs) == 1:
            return uni_tokens[max_idxs[0]], token_votes
        else:
            # randomly choose one
            idx = max_idxs[torch.randint(len(max_idxs), (1,))[0]]
            return uni_tokens[idx], token_votes
    else:
        priv_idx = dp_engine.get_top1(cnts, dim, k_bar=k_bar)
        if priv_idx < len(cnts):
            return uni_tokens[priv_idx], token_votes
        else:
            uni_tokens = uni_tokens # .data.cpu().numpy()
            # random token but exclude the existing ones.
            priv_idx = priv_idx - len(cnts)
            x = torch.ones((dim,), device=tokens.device)
            x[uni_tokens] = 0.
            t = torch.nonzero(x, as_tuple=True)[0][priv_idx]
            return t, None
        # for i_trial in range(1):
        #     try:
        #         return uni_tokens[dp_engine.get_top1(cnts, dim, k_bar=k_bar)], token_votes
        #     except NotFoundLDTop1:
        #         print(f"## [trail {i_trial}] not found LD Top1...")
        # else:
        #     raise NotFoundLDTop1()


def ensemble_generate(model, input_ids, attention_mask, eos_token_id, pad_token_id, 
                   do_sample=False, temperature=None, top_p=None, top_k=None,
                   max_new_tokens=20, no_repeat_ngram_size=None, repetition_penalty=None, batch_size=0,
                   dp_engine=None):
    # model.generation_config
    logits_processor = LogitsProcessorList([])
    logits_processor.append(MinLengthLogitsProcessor(10, eos_token_id=eos_token_id))
    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
        logits_processor.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if repetition_penalty is not None and repetition_penalty != 1.0:
        logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))

    logits_warper = LogitsProcessorList([])
    if do_sample:
        if temperature is not None and temperature != 1.0:
            logits_warper.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            logits_warper.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
        if top_p is not None and top_p < 1.0:
            logits_warper.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
    
    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=input_ids.shape[1] + max_new_tokens)])

    # dp_engine = LDGumbelMechanism(dp_eps, dp_delta) if dp_eps is not None else None
    output_ids = greedy_search(model, input_ids, attention_mask, eos_token_id, pad_token_id, 
                               do_sample=do_sample, logits_processor=logits_processor, 
                               logits_warper=logits_warper, stopping_criteria=stopping_criteria, 
                               batch_size=batch_size, dp_engine=dp_engine)
    output_ids = output_ids[:1]  # NOTE all generation are the same
    return output_ids


def greedy_search(
        model, input_ids, attention_mask, eos_token_id, pad_token_id,
        do_sample=False,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        batch_size=0,
        dp_engine: LDGumbelMechanism=None,
        ):
    """Ensemble greedy search by majority vote.
    
    input_ids, attention_mask should be of the same shape [batch, seq len, token size

    Return:
        `output_ids`
    
    Examples:
    ```python
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LogitsProcessorList,
        MinLengthLogitsProcessor,
        StoppingCriteriaList,
        MaxLengthCriteria,
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    input_prompt = "It might be possible to"
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
    
    from transformers import MinLengthLogitsProcessor, LogitsProcessorList, NoRepeatNGramLogitsProcessor

    logits_processor = LogitsProcessorList([
        MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        NoRepeatNGramLogitsProcessor(4),  # not suitable for specific name pairs.
    ])
    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

    output_ids = greedy_search(model, input_ids, attention_mask, tokenizer.eos_token_id, tokenizer.pad_token_id, logits_processor)
    print(tokenizer.decode(cur_input_ids[0, input_ids.shape[1]:], skip_special_tokens=True))
    ```
    """
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    # ensemble greedy search
    # based on transformers.generation.utils: greedy_search()
    cur_input_ids = input_ids.detach().clone()
    seq_token_votes = []
    model_kwargs = {}
    if attention_mask is not None:
        raise NotImplementedError('Have not implemented batched attention mask.')

    eos_token_id_tensor = torch.tensor([eos_token_id]).to(cur_input_ids.device) if eos_token_id is not None else None

    unfinished_sequences = torch.ones(cur_input_ids.shape[0], dtype=torch.long, device=cur_input_ids.device)
    if batch_size > 0:
        assert attention_mask is None, "Not supported for model_kwargs"
        model_kwargs_batch = [copy.deepcopy(model_kwargs) for _ in range(int(np.ceil(len(cur_input_ids) / batch_size)))]

        ds_keys = ['input_ids']
        ds = InputTokenDataset({'input_ids': cur_input_ids}, ds_keys)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    dp_failure_cnt = 0

    while True:
        if batch_size > 0:
            next_token_logits = []
            ds.model_inputs['input_ids'] = cur_input_ids
            for i, batch_data in enumerate(dl):
                _model_kwargs = model_kwargs_batch[i]
                model_inputs = model.prepare_inputs_for_generation(batch_data['input_ids'], **_model_kwargs)
                _outputs = model(#batch_data['input_ids'], **{k: v for k, v in model_inputs.items() if k not in ds_keys}, 
                                 **model_inputs,
                                 return_dict=True,)
                next_token_logits.append(_outputs.logits[:, -1, :].detach())
                # FXIME ad-hoc This could speed up but take a lot more memory.
                # _model_kwargs = model._update_model_kwargs_for_generation(
                #     _outputs, _model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
                # )
                # model_kwargs_batch[i] = _model_kwargs
            next_token_logits = torch.cat(next_token_logits, dim=0)
            # outputs = _outputs  # FIXME this is an ad-hoc solution for passing outputs to _update_model_kwargs_for_generation
            del _outputs
        else:
            model_inputs = model.prepare_inputs_for_generation(cur_input_ids, **model_kwargs)
            outputs = model(**model_inputs, return_dict=True,)
            next_token_logits = outputs.logits[:, -1, :]
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
            )
        next_token_scores = logits_processor(cur_input_ids, next_token_logits)
        next_token_scores = logits_warper(cur_input_ids, next_token_scores)
        token_dim = next_token_scores.shape[-1]
        
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # majority vote
        try:
            majority_token, token_votes = majority_vote(next_tokens, token_dim, dp_engine=dp_engine)

            # eps, delta = dp_engine.get_dp_expense()
            # print(f"#### dp_engine dp eps={eps:.4f}, delta={delta:.4f}")
        except NotFoundLDTop1:
            # # EM => very high privacy cost
            # print(f"Fail to infer token due to DP at {cur_input_ids.size(1) - input_ids.size(1)}-th new token. Degrade LD to EM.")
            # # NOTE you have to change the compose_method to rdp_em mode.
            # majority_token, token_votes = majority_vote(next_tokens, token_dim, dp_engine=dp_engine, k_bar=token_dim)

            if dp_engine.fail_mode == 'ld_pate':
                # 1-excluded LD.
                print(f"Fail to infer token due to DP at {cur_input_ids.size(1) - input_ids.size(1)}-th new token. Degrade LD to LD with k_bar=token_dim-1.")
                try:
                    # LimitedDomain with a smaller v_perp and higher chance to succeed.
                    majority_token, token_votes = majority_vote(next_tokens, token_dim, dp_engine=dp_engine, k_bar=token_dim-1)
                except NotFoundLDTop1:
                    print(f"LD failed. Stop.")
                    break
            elif dp_engine.fail_mode == 'rand':
                print(f"Fail to infer token due to DP at {cur_input_ids.size(1) - input_ids.size(1)}-th new token. Return random token.")
                majority_token = torch.randint(next_token_scores.shape[1], (1,), device=next_tokens.device)[0]
                token_votes = None
            elif dp_engine.fail_mode == 'stop':
                print(f"LD failed. Stop.")
                break
            elif dp_engine.fail_mode == 'raise':
                raise NotFoundLDTop1()
            else:
                raise NotImplementedError(f"fail_mode: {dp_engine.fail_mode}")

        except DPExpenseOverflow:
            print(f"## Early stop due to DP overflow. Output len: {cur_input_ids.size(1) - input_ids.size(1)}")
            eps, delta = dp_engine.get_dp_expense()
            print(f"## dp_engine dp eps={eps:.4f}, delta={delta:.4f}")
            break
        seq_token_votes.append(token_votes)
        next_tokens = majority_token.tile(len(next_tokens))

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        cur_input_ids = torch.cat([cur_input_ids, next_tokens[:, None]], dim=-1)

        # if eos_token was found in one sentence, set sentence to finished (0)
        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(eos_token_id_tensor.shape[0], 1)\
                .ne(eos_token_id_tensor.unsqueeze(1))\
                    .prod(dim=0)
        )

        # print(f"{i}: {next_tokens}")
        if unfinished_sequences.max() == 0:
            # print(f"EOS")
            break
        if stopping_criteria(cur_input_ids, None):
            break
    return cur_input_ids
