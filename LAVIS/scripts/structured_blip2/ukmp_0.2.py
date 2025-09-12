import subprocess

GPU = "0"
port = 18082 #sys.argv[1]

pruner_type = "taylor+knowledge"
taylor_type = "param_first"
granularity = "block"
pruning_ratio = 0.2
global_pruning = True
num_examples = 1000
imp_normalizer = "param"
select_loss = True
entropy_importance = True
multimodal = True

global_str = "global" if global_pruning else "local"
norm_str = imp_normalizer if imp_normalizer is not None else "no"
select_loss_str = "-select_loss" if select_loss else ""
multimodal_str = "-multimodal" if multimodal else ""

job_id = f"cc595k-{num_examples}data-{pruner_type}-{taylor_type}-{norm_str}_norm-{pruning_ratio}-{granularity}wise-{global_str}{select_loss_str}{multimodal_str}"

print("[START] - Start Pruning Model")
program = (
    f"CUDA_VISIBLE_DEVICES={GPU} python -u -m torch.distributed.run"
    f" --nproc_per_node=1 --master_port={port} ukmp_prune.py" 
    f" --cfg-path lavis/projects/blip2/prune/cc595k_prefix_derivative_compute.yaml"
    f" --device cuda"
    f" --save_ckpt_log_name ukmp_prune"
    f" --job_id {job_id}"
    # modificable
    f" --pruning_ratio {pruning_ratio}"
    f" --granularity {granularity}"
    f" --pruner_type {pruner_type}"
    f" --taylor {taylor_type}" 
    f" --num_examples {num_examples}"
    f" --channel_per_step 1000"
)
if global_pruning:
    program += f" --global_pruning"
if imp_normalizer is not None:
    program += f" --imp_normalizer {imp_normalizer}"
if select_loss:
    program += f" --select_loss"
if entropy_importance:
    program += f" --entropy_importance"
if multimodal:
    program += f" --multimodal"
subprocess.call(program, shell=True)  


to_process_ckpt = "pytorch_model"

print("[START] - Start Evaluating Pruned Model")
eval_job_id = "evaluate_pruned"
for task in ["okvqa_zeroshot_flant5xl_eval", "gqa_zeroshot_flant5xl_eval"]:
    program = (
        f"CUDA_VISIBLE_DEVICES={GPU} python -u evaluate_blip2_pruned.py"
        f" --cfg-path lavis/projects/blip2/eval/{task}.yaml"
        f" --pruned_ckpt pruned_checkpoint/ukmp_prune/{job_id}/{to_process_ckpt}.bin"
        f" --job_id {eval_job_id}"
    )
    subprocess.call(program, shell=True)
    

print("[START] - Start Finetuning Model")
wr_lora = True
distill = True
wr_lora_str = "-wr_lora" if wr_lora else ""
distill_str = "-distill" if distill else ""
tune_job_id = f"{job_id}+finetune{wr_lora_str}{distill_str}"
program = (
    f"CUDA_VISIBLE_DEVICES={GPU} python -u -m torch.distributed.run"
    f" --nproc_per_node=1 --master_port {port} ukmp_finetune.py"
    f" --cfg-path lavis/projects/blip2/finetune/cc595k_prefix_derivative_compute_distill_bs10.yaml"
    f" --device cuda"
    f" --save_ckpt_log_name ukmp_finetune"
    f" --job_id {tune_job_id}"
    f" --pruned_ckpt pruned_checkpoint/ukmp_prune/{job_id}/{to_process_ckpt}.bin"
    f" --lora_target_modules qkv,attn.proj,fc1,fc2,q,k,v,o,wi_0,wi_1,wo"
    f" --pruned_mask pruned_checkpoint/ukmp_prune/{job_id}/prune_masks.json"
)
if distill:
    program += f" --distill_mode"
if wr_lora:
    program += f" --wr_lora"
subprocess.call(program, shell=True)


print("[START] - Start Evaluating Finetuned Model")
eval_job_id = "evaluate_finetuned"
for task in ["okvqa_zeroshot_flant5xl_eval", "gqa_zeroshot_flant5xl_eval", "nocaps_pretrain_flant5xl_eval", "ret_flickr_pretrain_flant5xl_eval", "vqav2_zeroshot_flant5xl_eval"]:
    if "nocaps" in task:
        eval_job_id = eval_job_id + "_" + tune_job_id
    else:
        eval_job_id = "evaluate_finetuned"
    program = (
        f"CUDA_VISIBLE_DEVICES={GPU} python -u evaluate_blip2_pruned.py"
        f" --cfg-path lavis/projects/blip2/eval/{task}.yaml"
        f" --pruned_ckpt pruned_checkpoint/ukmp_prune/{job_id}/{to_process_ckpt}.bin"
        f" --peft_ckpt tuned_checkpoint/ukmp_finetune/{tune_job_id}"
        f" --job_id {eval_job_id}" # TODO
        f" --pruned_mask pruned_checkpoint/ukmp_prune/{job_id}/prune_masks.json"
    )
    subprocess.call(program, shell=True)