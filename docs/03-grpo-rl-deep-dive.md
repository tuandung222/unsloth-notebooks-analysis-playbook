# 03 - GRPO Deep Dive (tổng quan kiến trúc)

Tài liệu này trả lời 3 câu hỏi:
1. Unsloth đang triển khai GRPO theo pattern nào?
2. Các notebook GRPO khác nhau ở đâu?
3. Người mới nên học theo thứ tự nào để lên tay nhanh?

## 1) Bức tranh tổng thể
Trong `unslothai/notebooks`, nhóm GRPO không phải một notebook duy nhất mà là một *hệ notebook* gồm nhiều nhánh:
- `GRPO text reasoning` (toán, reasoning, format-enforced answers).
- `GRPO advanced` (thêm bước pre-SFT để model học format trước khi RL).
- `GRPO vision` (reward trên output có ảnh + text).
- `GRPO environment/game` (2048, Sudoku, OpenEnv) với reward dựa trên kết quả chạy strategy.
- `GRPO FP8` (tối ưu memory/throughput cho model lớn hơn).

## 2) Skeleton chung của một notebook GRPO
Hầu hết notebook GRPO trong repo đi theo khung 8 bước:
1. Cài môi trường (pin `transformers`, `trl`, cờ `UNSLOTH_VLLM_STANDBY`).
2. Load base/instruct model bằng `FastLanguageModel` hoặc `FastVisionModel`.
3. Gắn LoRA adapters bằng `get_peft_model`.
4. Chuẩn bị dataset thành dạng `prompt` + `answer`.
5. Thiết kế reward functions (format reward + correctness reward + penalty).
6. Khởi tạo `GRPOConfig`.
7. Train bằng `GRPOTrainer`.
8. Inference + save LoRA/merged/GGUF.

## 3) Ma trận các nhánh GRPO quan trọng
| Nhánh | Notebook đại diện | Khác biệt chính |
| --- | --- | --- |
| Text GRPO cơ bản | `Llama3.1_(8B)-GRPO.ipynb` | Reward dạng XML tags + correctness |
| Text GRPO base model | `Qwen3_(4B)-GRPO.ipynb` | Tự thiết kế chat template + pre-SFT format |
| Text GRPO nâng cao | `Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb` | Reward nhiều tầng + tuning dài hơn |
| FP8 GRPO | `Qwen3_8B_FP8_GRPO.ipynb` | `load_in_fp8=True`, tối ưu memory |
| Vision GRPO | `Gemma3_(4B)-Vision-GRPO.ipynb` | Prompt multimodal + reward parsing ảnh/text |
| Environment GRPO | `gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb` | Reward qua execution kết quả game |

## 4) Điểm thiết kế nhất quán (rất quan trọng)
- Reward luôn được tách thành nhiều hàm nhỏ (đúng format, đúng answer, heuristic gần đúng).
- `num_generations` thường > 1 để tạo tín hiệu ranking giữa nhiều completion.
- `max_prompt_length` được set cẩn thận để tránh cắt prompt gây reward sai.
- `check_numbers`/regex parsing xuất hiện nhiều để giảm reward hacking.
- Trước khi push hiệu năng, notebook luôn ưu tiên chạy ổn trên T4/L4 bằng LoRA + quantization.

## 5) Khác biệt cần nhớ giữa các nhánh
- `Instruct model` (ví dụ Llama3.1-GRPO): không cần can thiệp chat template quá sâu.
- `Base model` (Qwen3-4B-GRPO): cần ép format + thường có phase pre-SFT để GRPO hội tụ tốt.
- `Vision GRPO`: reward không chỉ nhìn chuỗi text, còn phụ thuộc cấu trúc multimodal input.
- `Game/OpenEnv GRPO`: reward phụ thuộc side-effect khi chạy code; bắt buộc chống cheating/sandboxing.

## 6) Lộ trình đọc nhanh cho người mới
1. `Llama3.1_(8B)-GRPO.ipynb`: hiểu minimal GRPO pipeline.
2. `Qwen3_(4B)-GRPO.ipynb`: học cách xử lý base model + chat template + pre-SFT.
3. `Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb`: học reward design nâng cao.
4. `Gemma3_(4B)-Vision-GRPO.ipynb`: mở rộng sang multimodal.
5. `gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb`: bước vào RL planning/orchestration thực sự.
