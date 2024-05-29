import gradio as gr

def train_interface(dataset, model, algorithm, eps, device, dln1_steps, num_prompt):
    from train_opt import main
    
    if algorithm.lower() == "opt":
        args = f"--ape_mode=iid_ibwd --ensemble_gen=True --gen_temp=1.1 --num_prompt={num_prompt} --max_new_tokens=50 --data={dataset} --holdout_ratio=0.01 --target_eps=8. --dp_eps=1.8 --dp_delta=5e-7 --tokenwise_gen=True --model={model} --device={device} --no_wandb"
    elif algorithm.lower() == "dp-opt":
        args = f"--ape_mode=iid_ibwd --ensemble_gen=True --ensemble_num=205 --gen_temp=1.1 --num_prompt={num_prompt} --max_new_tokens=50 --data={dataset} --holdout_ratio=0.01 --batch_size=32 --target_eps={eps} --dp_eps=1.8 --dp_delta=5e-7 --tokenwise_gen=True --device={device} --no_wandb"
    elif algorithm.lower() == "dln-1":
        args = f"--ape_mode=bwd --gen_temp=0.7 --num_demos=20 --num_prompt={num_prompt} --rm_eval_item_name=True --steps={dln1_steps} --max_new_tokens=100 --data={dataset} --holdout_ratio=0.01 --batch_size=32 --device={device} --no_wandb"
    else:
        raise RuntimeError(f"Unknown algorithm: {algorithm}")
    best_save_dict, save_dict = main(args.split(" "))
    print(best_save_dict)
    best_idx = best_save_dict['best_holdout_idx']
    best_gen_instruct = best_save_dict['generated_instructs'][best_idx]
    return best_gen_instruct, best_save_dict['best_holdout_test_acc'] * 100


def test_interface(test_set, best_gen_instruct, test_model, test_ratio, device, batch_size, instruct_type):
    # text-davinci-003
    from utils.simple_eval import evaluate_prompt
    arg_list = f"--data={test_set} --test_model={test_model} --test_batch_size={batch_size} --test_ratio={test_ratio} --device={device} --instruct_type={instruct_type}".split(" ")
    acc, example_query = evaluate_prompt(best_gen_instruct, arg_list)
    return acc * 100, example_query

with gr.Blocks() as demo:
    gr.Markdown("## DP-OPT: Privacy-Preserving Prompt Engineer")
    gr.Markdown("Engineer a prompt using your private dataset.")
    with gr.Tabs():
        with gr.Tab(label="Train"):
            gr.Markdown("## Engineer Your Prompt with Local Model")
            with gr.Row():
                with gr.Column():
                    # name_input = gr.Textbox(label="Enter your name:")
                    # greeting_input = gr.Dropdown(label="Choose a greeting:", choices=["Hello", "Hi", "Greetings", "Salutations"], value="Hello")
                    inputs = [gr.Dropdown(label="Private Dataset", value='sst2', choices=["sst2", "trec", "mpqa", "disaster"]), 
                                gr.Dropdown(label="Local LLM Engineer", value="lmsys/vicuna-7b-v1.3", choices=["lmsys/vicuna-7b-v1.3", "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-Instruct"]),
                                gr.Dropdown(label="Algorithm", value="DLN-1", choices=["DLN-1", "OPT", "DP-OPT"]),
                            ]
                    with gr.Accordion("More settings", open=False):
                        inputs += [
                            gr.Slider(0, 32, value=8., label="eps (DP param)"),
                            gr.Dropdown(label="Device", value="cuda", choices=["cuda", "cpu"]),
                            gr.Number(value=20, label='DLN-1 steps'),
                            gr.Number(value=20, label='Number of candidate prompts')
                        ]
                    train_button = gr.Button("Start", variant="primary")
                with gr.Column():
                    outputs = [
                        gr.TextArea(label="Suggested Prompt (Instruction)"),
                        gr.Number(label="Test Accuracy (%)"),
                    ]
            
            train_button.click(train_interface, 
                    inputs=inputs,
                    outputs=outputs,)
            
        
        with gr.Tab(label="Test"):
            gr.Markdown("## Test Your Prompt on Desired Model")
            with gr.Row():
                with gr.Column():
                    test_inputs = [
                        gr.Textbox(label="Test Dataset", value='sst2'), 
                        gr.Textbox(label="Enter your prompt (instruction):"),
                        gr.Dropdown(label="Test Model", value="meta-llama/Llama-2-13b-chat-hf", choices=["lmsys/vicuna-7b-v1.3", "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-70b-chat-hf", "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-70B-Instruct"]),
                        ]
                    with gr.Accordion("More settings", open=False):
                        test_inputs += [
                            gr.Slider(0, 1., value=0.05, label="Percentage of test set"),
                            gr.Dropdown(label="Device", value="cuda", choices=["cuda", "cpu"]),
                            gr.Number(label='batch size', value=4),
                            gr.Dropdown(label="Instruction Template", value='vicuna', choices=["vicuna", "llama-2", "llama-3"])
                        ]
                    test_button = gr.Button("Test", variant="primary")
                with gr.Column():
                    test_outputs = [
                        gr.Textbox(label="Test Accuracy (%)"),
                        gr.TextArea(label="Example prompt")
                        ]
                    
                test_button.click(test_interface, inputs=test_inputs, outputs=test_outputs)

demo.launch()
