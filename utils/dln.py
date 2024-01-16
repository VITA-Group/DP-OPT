from transformers import LlamaForCausalLM
import torch
from tqdm import trange
import numpy as np
from typing import Union, List, Dict

from utils.template import BwdDemosTemplate, DemosTemplate, GenerationTemplate, BackwardGenTemplate, get_bwd_template
from utils.data import sample_demos
from utils.ensemble import ensemble_generate

from utils.dp import LDGumbelMechanism, DPExpenseOverflow, NotFoundLDTop1
from .evaluate import Evaluator


class InstructGenerator(object):
    def __init__(self, model: LlamaForCausalLM, tokenizer, device, max_new_tokens: int, label_words, 
                 instruct_type, ensemble_gen=False, disable_att_mask=False, gen_batch_size=0, do_sample=True, gen_temperature=0.9,
                 rep_penalty=1.,
                 dp_engine=None, balance_demos=False) -> None:
        if 'mpt' in instruct_type:
            sys_intro = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
            meta_template = GenerationTemplate(
                                sys_intro + "## Instruction:\n" \
                                "I gave a friend a instruction. Based on the instruction they produced " \
                                "the following input-output pairs:\n\n\n[full_DEMO]\n\n" \
                                "### Response:\n" \
                                "The instruction was to [APE]")
            demo_template = DemosTemplate('## Input: [INPUT]\n## Output: [OUTPUT]')
        elif instruct_type == 'vicuna':
            meta_template = GenerationTemplate("I gave a friend a instruction. Based on the instruction they produced " \
                                "the following input-output pairs:\n\n\n[full_DEMO]\n\n\nThe instruction was to [APE]")
            demo_template = DemosTemplate('Input: [INPUT]\nOutput: [OUTPUT]')
        else:
            raise NotImplementedError(f"instruct_type: {instruct_type}")
        self.demo_template = demo_template
        self.meta_template = meta_template

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.label_words = label_words
        self.ensemble_gen = ensemble_gen
        self.disable_att_mask = disable_att_mask
        self.gen_batch_size = gen_batch_size
        self.do_sample = do_sample
        self.gen_temperature = gen_temperature
        self.rep_penalty = rep_penalty
        self.balance_demos = balance_demos

        # dp
        self.dp_engine = dp_engine

    def generate_prompt(self, num_demos, dataset, rng, num_prompt=1, num_meta_prompt=None):
        """
        Inputs:
            num_demos (int): Number of demos for each meta prompt.
            num_prompt (int): Number of prompts to be generated.
            num_meta_prompt (int): Number of meta-prompts to be construted from dataset. By default, None means num_meta_prompt=num_prompt.
        """
        if num_meta_prompt is None:
            num_meta_prompt = num_prompt
        # get demos for prompt
        meta_prompts = []
        all_demos = sample_demos(
            dataset, num_meta_prompt * num_demos, self.label_words, 
            make_text_target=False, rng=rng, balance=self.balance_demos)
        all_inputs, all_targets = all_demos
        for i_meta_prompt in range(num_meta_prompt):
            inputs = all_inputs[i_meta_prompt*num_demos:(i_meta_prompt+1)*num_demos]
            targets = all_targets[i_meta_prompt*num_demos:(i_meta_prompt+1)*num_demos]  # target is numerical
            
            full_demo = self.demo_template.fill([inputs, targets])
            meta_prompt = self.meta_template.fill(full_demo)
            print(f"\n>>> Meta prompt #{i_meta_prompt}:\n[START]{meta_prompt}[END]")

            meta_prompts.append(meta_prompt)

        # generate
        instructs = self.forward_generate_prompt(meta_prompts, num_prompt)

        # post-process intructs
        processed_instructs = []
        for i_meta_prompt, instruct in zip(range(len(instructs)), instructs):
            inputs = all_inputs[i_meta_prompt*num_demos:(i_meta_prompt+1)*num_demos]
            targets = all_targets[i_meta_prompt*num_demos:(i_meta_prompt+1)*num_demos]  # target is numerical
            demos = [inputs, targets]

            privacy_leaked, leakded_demo = exam_privacy_leakage(instruct, demos)

            instruct = instruct.strip()
            instruct = instruct.replace('\n', '  ')
            print(f"\n>>> Generated instruct:\n", '[START]'+ instruct + '[END]')
            if privacy_leaked:
                print(f"[ALERT] Dump instruct. because privacy leaked for demo: {leakded_demo}.")
                instruct = None
            else:
                processed_instructs.append(instruct)
        return processed_instructs, all_demos

    def forward_generate_prompt(
            self, meta_prompts: Union[str, List[str]], num_prompts: int, 
            max_new_tokens=None, gen_init='', verbose=True, decode_special_tokens=False,
            reraise_dp_fail=False):
        if not isinstance(meta_prompts, list):
            meta_prompts = [meta_prompts]
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        if len(gen_init) > 0:
            base_meta_prompts = [meta_prompt.replace('[APE]', '') for meta_prompt in meta_prompts]
            base_input_ids = self.tokenizer(base_meta_prompts, return_tensors='pt', padding=True).input_ids
            gen_init_id_len = len(base_input_ids[0])
        else:
            gen_init_id_len = None

        meta_prompts = [meta_prompt.replace('[APE]', gen_init) for meta_prompt in meta_prompts]  # forward next word

        tokenized = self.tokenizer(meta_prompts, return_tensors='pt', padding=True)  # , padding_side='left'
        input_ids = tokenized.input_ids
        if gen_init_id_len is None:
            gen_init_id_len = len(input_ids[0])
        attention_mask = tokenized.attention_mask
        if self.ensemble_gen:
            output_ids = []
            try:
                for i_meta_prompt in trange(num_prompts, desc='sample prompt', disable=not verbose):
                    smp_output_ids = ensemble_generate(
                        self.model,
                        input_ids.to(self.device),
                        attention_mask.to(self.device) if not self.disable_att_mask else None,
                        self.tokenizer.eos_token_id, 
                        self.tokenizer.pad_token_id,
                        do_sample=self.do_sample,
                        # based on the configs/default.yaml of APE
                        temperature=self.gen_temperature,  # 0.9,
                        repetition_penalty=self.rep_penalty,  # based on https://arxiv.org/pdf/1909.05858.pdf (b/f Sec 4.2)
                        max_new_tokens=max_new_tokens,  # 50
                        batch_size=self.gen_batch_size,
                        dp_engine=self.dp_engine,
                    )
                    output_ids.append(smp_output_ids[0])
                    instruct = self.tokenizer.decode(
                        smp_output_ids[0][input_ids.size(1):], skip_special_tokens=True, spaces_between_special_tokens=False
                    )
                    if verbose:
                        print(f"generated instruct:\n[START]{instruct}[END]")
                    if self.dp_engine is not None:
                        self.dp_engine.check_dp_budget(verbose=verbose)
            except DPExpenseOverflow:
                eps, delta = self.dp_engine.get_dp_expense()
                if verbose:
                    print(f"Reach target DP at {i_meta_prompt}-th prompt, eps={eps:.4f}, delta={delta:4f}")
                if reraise_dp_fail:
                    raise DPExpenseOverflow()
        else:
            assert len(meta_prompts) == num_prompts, "num_prompts should be the same as the size of meta prompts."
            output_ids = self.model.generate(
                input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device) if not self.disable_att_mask else None,
                do_sample=self.do_sample,
                # based on the configs/default.yaml of APE
                temperature=self.gen_temperature,
                repetition_penalty=1.0,  # not found corresponding
                max_new_tokens=max_new_tokens,  # 50
            )
        output_ids = [
            output_ids_ if self.model.config.is_encoder_decoder else output_ids_[gen_init_id_len:]
            for output_ids_ in output_ids
        ]
        out_prompts = self.tokenizer.batch_decode(
            output_ids, 
            skip_special_tokens=not decode_special_tokens, # 
            spaces_between_special_tokens=False
        )
        return out_prompts


def exam_privacy_leakage(instruct, demos):
    privacy_leaked, leakded_demo = False, None
    for input_ in demos[0]:
        if input_ in instruct:
            privacy_leaked = True
            leakded_demo = input_
            break
    return privacy_leaked, leakded_demo


class BackwardInstructGenerator(InstructGenerator):
    """Generate instruct with backward."""
    def __init__(self, model: LlamaForCausalLM, tokenizer, device, max_new_tokens: int, label_words, 
                 instruct_type, ensemble_gen=False, disable_att_mask=False, gen_batch_size=0, do_sample=True, gen_temperature=0.9, 
                 rep_penalty=1.,
                 dp_engine: LDGumbelMechanism=None,
                 balance_demos=False, tokenwise_gen=False, privacy_instruct=-1) -> None:
        meta_template, suc_demo_template, fail_demo_template = get_bwd_template(instruct_type, privacy_instruct=privacy_instruct)
        
        self.suc_demo_template = suc_demo_template  # type: BwdDemosTemplate
        self.fail_demo_template = fail_demo_template  # type: BwdDemosTemplate
        self.meta_template = meta_template  # type: BackwardGenTemplate

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.label_words = label_words
        self.ensemble_gen = ensemble_gen
        self.disable_att_mask = disable_att_mask
        self.gen_batch_size = gen_batch_size
        self.balance_demos = balance_demos

        self.bwd_messages = [
            "Clarify the instruction by adding few words or a short sentence. Be concise",
            "Improve the instruction by providing examples on how to solve the task. Be concise.",
            "Shorten the instruction by removing superflous words or sentences.",
            "Rewrite the instruction by providing detailed information to avoid ambiguity. Be concise",
        ]

        # dp
        self.dp_engine = dp_engine

        self.do_sample = do_sample
        self.gen_temperature = gen_temperature
        self.rep_penalty = rep_penalty
        self.tokenwise_gen = tokenwise_gen

        # log
        self._verbose_cnt = 1
    
    def iterative_generate(self, init_instruct, num_demos, dataset, rng: np.random.RandomState, evaluator: Evaluator, num_prompt=1, num_meta_prompt=None, iid_instruct=False, **kwargs):
        """If iid_instruct is True, will use last instruct to update meta prompts."""
        if iid_instruct:
            # Since instruct is not changed, the sample predictions are constant and can be computed.
            dataset = self.dln_fwd_pass(init_instruct, dataset, evaluator)

        if num_meta_prompt is None:
            num_meta_prompt = num_prompt
        cur_instruct = init_instruct
        generated_instructs, used_demos = [], []  # will not keep used demos
        for i_prompt in range(num_prompt):
            print(f"[Iter {i_prompt}/{num_prompt}] generating  prompt")
            _generated_instructs, _used_demos = self.generate_instruct_bwd(cur_instruct, num_demos, dataset, rng, evaluator, num_prompt=1, num_meta_prompt=num_meta_prompt, **kwargs)
            generated_instructs.extend(_generated_instructs)
            # used_demos.extend(_used_demos)
            if not iid_instruct:
                cur_instruct = _generated_instructs[0]
                assert 'prediction' not in dataset[0], "For not iid_instruct, the demo predictions have to regenerated every iter."
            if self.dp_engine is not None:
                if not self.dp_engine.check_dp_budget(raise_error=False):
                    break
        return generated_instructs, used_demos

    def dln_fwd_pass(self, cur_instruct, dataset: List[Dict], evaluator: Evaluator):
        assert isinstance(dataset, list), "require List[Dict] type."
        assert 'prediction' not in dataset[0], "the dataset has been predicted. Make sure a raw dataset is inputed."
        # forward pass on all samples.
        def process(example):
            # we will add text entry but also keep the old entry which will be used)
            new_ex = {k: v for k, v in example.items()}
            new_ex['text'] = example[evaluator.eval_template.input_key]
            return new_ex
        new_dataset = [process(d) for d in dataset]
        # input_key=evaluator.eval_template.input_key
        acc, loss, losses, all_pred_labels, all_targets = evaluator.evaluate_prompt(new_dataset, cur_instruct, return_all=True, shuffle=False, desc=f"dln-fwd")
        for i, d in enumerate(new_dataset):
            d['prediction'] = all_pred_labels[i]
        assert 'prediction' in new_dataset[0]
        return new_dataset

    def generate_instruct_bwd(self, cur_instruct, num_demos, dataset, 
                              rng: np.random.RandomState, evaluator: Evaluator, 
                              num_prompt=1,
                              **kwargs):
        if self.tokenwise_gen:
            assert num_prompt == 1
            return self._generate_instruct_bwd_tokwise(
                cur_instruct, num_demos, dataset, rng, evaluator, 
                **kwargs)
        else:
            return self._generate_instruct_bwd(
                cur_instruct, num_demos, dataset, rng, evaluator, 
                num_prompt=num_prompt, **kwargs)

    def _generate_instruct_bwd_tokwise(
            self, cur_instruct, num_demos, dataset, 
            rng: np.random.RandomState, evaluator: Evaluator, **kwargs):
        """When generate each token, we will sample a new meta-prompt."""
        gen_instruct = ''
        cur_len = len(gen_instruct)
        max_retry = 5
        if self.dp_engine is not None:
            dp_fail_mode = self.dp_engine.fail_mode
            if dp_fail_mode != 'rand':
                # enforce the engine to raise error such that we can handle the failure here.
                self.dp_engine.fail_mode = 'raise'
        print(f"Generating: [START]", end='', flush=True)
        # for i_token in range(self.max_new_tokens):
        i_token = 0
        per_token_max_retry = 1
        all_demos = []
        while i_token < self.max_new_tokens:
            kwargs['decode_special_tokens'] = True
            kwargs['reraise_dp_fail'] = True
            # generate one token at a time.
            try:
                processed_instructs, all_demos = self._generate_instruct_bwd(
                    cur_instruct, num_demos, dataset, 
                    rng, evaluator, num_prompt=1, max_new_tokens=1,
                    gen_init=gen_instruct, do_post_process=False,
                    **kwargs)
                gen_instruct = processed_instructs[0]
                print(gen_instruct[cur_len:], end='', flush=True)
                cur_len = len(gen_instruct)
                if self.tokenizer.eos_token in gen_instruct:
                    # remove eos token.
                    gen_instruct = gen_instruct.replace(self.tokenizer.eos_token, '')
                    break
            except DPExpenseOverflow:
                eps, delta = self.dp_engine.get_dp_expense()
                print(f"Reach target DP at {i_token}-th prompt, eps={eps:.4f}, delta={delta:4f}")
                break
            except NotFoundLDTop1:
                if dp_fail_mode == 'retry':
                    if max_retry > 0 and per_token_max_retry > 0:
                        print(f"\n! Fail with LD at {i_token}-th token. Retry....")
                        max_retry -= 1
                        per_token_max_retry -= 1
                        continue
                    else:
                        print(f"\n! Fail with LD at {i_token}-th token. Stop...")
                        break
                elif dp_fail_mode == 'rand':
                    continue
                elif dp_fail_mode == 'stop':
                    break
                else:
                    raise RuntimeError(f"Unknown dp_fail_mode: {dp_fail_mode}")
            i_token += 1
            per_token_max_retry = 1
        print(f"[END]")
        instruct = gen_instruct
        # privacy_leaked, leakded_demo = exam_privacy_leakage(instruct, all_demos)

        instruct = instruct.strip()
        instruct = instruct.replace('\n', '  ')

        if self.dp_engine is not None:
            self.dp_engine.fail_mode = dp_fail_mode
        return [instruct], all_demos
    
    def _generate_instruct_bwd(self, cur_instruct, num_demos, dataset, 
                               rng: np.random.RandomState, evaluator: Evaluator, 
                               num_prompt=1, num_meta_prompt=None, verbose=False,
                               max_new_tokens=None, gen_init='', do_post_process=True,
                               decode_special_tokens=False, reraise_dp_fail=False):
        """
        Inputs:
            num_demos (int): Number of demos for each meta prompt.
            num_prompt (int): Number of prompts to be generated. If not ensemble, 
                this is the generation per meta_prompt, otherwise total prompts.
            num_meta_prompt (int): Number of meta-prompts to be construted from dataset. By default, None means num_meta_prompt=num_prompt.
        """
        if num_meta_prompt is None:
            num_meta_prompt = num_prompt
        
        # prepare predicted demos for meta prompts.
        all_demos, demo_subset = sample_demos(
            dataset, num_meta_prompt * num_demos, self.label_words,
            make_text_target=False, rng=rng, return_subset=True,
            input_key=evaluator.eval_template.input_key,
            balance=self.balance_demos, poisson=self.dp_engine is not None)
        all_inputs, all_targets = all_demos
        if 'prediction' in dataset[0]:
            # extract predicted labels.
            all_pred_labels = [d['prediction'] for d in demo_subset]
        else:
            # calculate predicted labels.
            acc, loss, losses, all_pred_labels, all_targets = evaluator.evaluate_prompt(demo_subset, cur_instruct, return_all=True, shuffle=False, desc=f"dln-fwd")

        batch_meta_prompts = []
        for i_meta_prompt in range(num_meta_prompt):
            start_idx, end_idx = i_meta_prompt*num_demos, (i_meta_prompt+1)*num_demos
            pred_labels = all_pred_labels[start_idx:end_idx]
            targets = all_targets[start_idx:end_idx]
            inputs = all_inputs[start_idx:end_idx]

            # Incorporate the pred labels (pred_labels) versus targets.
            assert len(pred_labels) == len(targets)
            suc_bwd_infos, fail_bwd_infos = [], []
            for pred_label, target, input_text in zip(pred_labels, targets, inputs):
                data_dict = {'input': input_text, 'output': self.label_words[pred_label], 'target': self.label_words[target]}
                if target == pred_label:
                    suc_bwd_infos.append(data_dict)
                else:
                    fail_bwd_infos.append(data_dict)

            # Prepare meta prompt
            full_suc_demo = self.suc_demo_template.fill(suc_bwd_infos)
            full_fail_demo = self.fail_demo_template.fill(fail_bwd_infos)
            bwd_msg = rng.choice(self.bwd_messages, 1)[0]
            meta_prompt = self.meta_template.fill(cur_instruct, full_suc_demo, full_fail_demo, bwd_msg)
            if verbose or self._verbose_cnt > 0:
                print(f"\n>>> Meta prompt #{i_meta_prompt}:\n[START]{meta_prompt}[END]")
                self._verbose_cnt -= 1

            batch_meta_prompts.append(meta_prompt)

        # Generate prompt
        instructs = self.forward_generate_prompt(
            batch_meta_prompts, num_prompt, max_new_tokens=max_new_tokens, 
            gen_init=gen_init, verbose=verbose, decode_special_tokens=decode_special_tokens,
            reraise_dp_fail=reraise_dp_fail)

        processed_instructs = []
        for i_meta_prompt, instruct in zip(range(len(instructs)), instructs):
            if self.ensemble_gen:
                demos = [all_inputs, all_targets]
            else:
                inputs = all_inputs[i_meta_prompt*num_demos:(i_meta_prompt+1)*num_demos]
                targets = all_targets[i_meta_prompt*num_demos:(i_meta_prompt+1)*num_demos]  # target is numerical
                demos = [inputs, targets]

            if do_post_process:
                privacy_leaked, leakded_demo = exam_privacy_leakage(instruct, demos)

                instruct = instruct.strip()
                instruct = instruct.replace('\n', '  ')
                print(f"\n>>> Generated instruct:\n", '[START]'+ instruct + '[END]')
            else:
                privacy_leaked = False
            if privacy_leaked:
                print(f"[ALERT] Dump instruct. because privacy leaked for demo: {leakded_demo}.")
                instruct = None
            else:
                processed_instructs.append(instruct)
        return processed_instructs, all_demos
