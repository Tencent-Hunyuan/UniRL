# Results: MiniMax-H3 bf16 vs fp32 LoRA gradient

All numbers come from one host (8 GPUs reporting as `NVIDIA CMP 170HX`, 64 GB each; PyTorch 2.13.0+cu130, diffusers 0.42.1, peft 0.21.2) and the step-20 LoRA checkpoint of a bf16 Flash-GRPO run. The gradient numbers use 8 rows captured from one saved rollout batch, with the same cotangent at the model output in every case. There is no fp32 training run to compare the reward curves with.

## 1. Standalone script: back-propagated signal per block

![standalone](figures/fig2_standalone_backward_cos.png)

Plain PyTorch, no UniRL, one captured micro-batch of 8 rows from one prompt group at sigma = 1.0, same cotangent at the model output in every case. Cosine of the signal entering each block against fp32, and the relative L2 error of the forward hidden state. Data: `data/standalone_backward_cos_per_block.csv`. The LoRA-rounding control (fp32 compute with the LoRA rounded to bf16) is in `data/rounding_control_grad_per_block.csv`.

## 2. Training reward of the bf16 runs

![train reward](figures/fig3_train_batch_reward.png)

Mean rollout reward (HPSv3) per update. Data: `data/train_batch_reward.csv`.

## 3. Fixed evaluation set

![eval](figures/fig4_eval_paired_delta.png)

24 prompts × 2 seeds, 384×256, 10 steps, paired difference against the untrained model with a 95 % interval. Data: `data/eval_paired_delta.csv`.
