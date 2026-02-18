# 03 - Deep dive GRPO / RL cho LLM

## Notebook đại diện nên đọc
- `Qwen3_(4B)-GRPO.ipynb`
- `Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb`
- `DeepSeek_R1_0528_Qwen3_(8B)_GRPO.ipynb`
- `gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb`
- `Ministral_3_(3B)_Reinforcement_Learning_Sudoku_Game.ipynb`

## Pipeline GRPO điển hình
1. Tạo/chuẩn hoá chat template (role + special tags reasoning/solution).
2. Chuẩn bị dataset prompt-answer.
3. Định nghĩa reward functions:
- format reward (đúng cấu trúc output),
- exact answer reward,
- approximate numeric reward,
- penalty cho output lỗi format.
4. Cấu hình `GRPOConfig` và train với `GRPOTrainer`.
5. Validate model trước/sau GRPO, lưu LoRA, kiểm tra adapter tensors.

## Insight quan trọng
- Reward design quan trọng hơn việc đổi model trong giai đoạn đầu.
- Phase “pre-finetune for formatting” giúp GRPO hội tụ nhanh hơn.
- Với bài toán reasoning số học, reward dạng “close-enough ratio” ổn định hơn binary reward.
- Nếu output dài, cần tối ưu `max_prompt_length` và generation budget để tránh OOM.

## Lỗi phổ biến khi mới làm RL cho LLM
- Reward hacking do regex quá đơn giản.
- Overfit format nhưng không tăng chất lượng answer.
- Batch/generation setting vượt VRAM của T4.
- Không tách baseline metrics trước khi RL.
