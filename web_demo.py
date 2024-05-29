import gradio as gr

def main_interface(dataset, model, algorithm, eps):
    from train_opt import main
    
    if algorithm.lower() == "opt":
        args = f"--ape_mode=iid_ibwd --ensemble_gen=True --gen_temp=1.1 --num_prompt=40 --max_new_tokens=50 --data={dataset} --holdout_ratio=0.01 --target_eps=8. --dp_eps=1.8 --dp_delta=5e-7 --tokenwise_gen=True --model={model} --no_wandb"
    elif algorithm.lower() == "dp-opt":
        args = f"--ape_mode=bwd --gen_temp=0.7 --num_demos=20 --num_prompt=20 --rm_eval_item_name=True --steps=20 --max_new_tokens=100 --data={dataset} --holdout_ratio=0.01 --batch_size=32 --no_wandb --target_eps=8. --dp_eps={eps} --dp_delta=5e-7 --tokenwise_gen=True"
    elif algorithm.lower() == "dln-1":
        args = f"--ape_mode=bwd --gen_temp=0.7 --num_demos=20 --num_prompt=20 --rm_eval_item_name=True --steps=20 --max_new_tokens=100 --data={dataset} --holdout_ratio=0.01 --batch_size=32 --no_wandb"
    else:
        raise RuntimeError(f"Unknown algorithm: {algorithm}")
    best_save_dict, save_dict = main(args.split(" "))
    print(best_save_dict)
    best_idx = best_save_dict['best_holdout_idx']
    best_gen_instruct = best_save_dict['generated_instructs'][best_idx]
    return best_gen_instruct, best_save_dict['best_holdout_test_acc']

demo = gr.Interface(
    fn=main_interface,
    inputs=[gr.Textbox(label="Private Dataset", value='sst2'), 
            gr.Dropdown(label="Model", value="lmsys/vicuna-7b-v1.3", choices=["lmsys/vicuna-7b-v1.3", "llama2-7b"]),
            gr.Dropdown(label="Algorithm", value="DLN-1", choices=["DLN-1", "OPT", "DP-OPT"]),
            # gr.Dropdown(label="Test Model", value="vicuna-13b", choices=["lmsys/vicuna-7b-v1.3", "llama2-7b", "vicuna-13b", "llama2-13b"]),
            gr.Slider(0, 32, value=8., label="eps (DP param)")],
    outputs=[
        gr.TextArea(label="Suggested Instruction"),
        gr.Number(label="Test Accuracy (%)"),
        ],
    title="DP-OPT: Privacy-Preserving Prompt Engineer",
    description="Engineer a prompt using your private dataset."
)

demo.launch()
