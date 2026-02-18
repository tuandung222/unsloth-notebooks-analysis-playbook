# 08 - Playbook quy trình training GRPO bằng Unsloth (thực hành end-to-end)

Tài liệu này là runbook thực chiến để bạn lặp lại GRPO training một cách có hệ thống.

## 0) Mục tiêu đầu ra của quy trình
Sau khi hoàn thành, bạn cần có:
- 1 LoRA adapter đã train bằng GRPO.
- 1 bộ prompt test trước/sau RL.
- 1 log reward dynamics (ít nhất reward mean/std theo step).
- 1 ghi chú failure modes (format fail, answer fail, timeout, cheating...).

## 1) Chọn nhánh notebook đúng use-case
- Text reasoning: bắt đầu `Llama3.1_(8B)-GRPO.ipynb`.
- Base model reasoning: dùng `Qwen3_(4B)-GRPO.ipynb`.
- Resource constrained nhưng muốn model lớn hơn: dùng `Qwen3_8B_FP8_GRPO.ipynb`.
- Vision reasoning: dùng `Gemma3_(4B)-Vision-GRPO.ipynb`.
- Planning/game RL: dùng `gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb`.

## 2) Chuẩn bị môi trường (Preflight)
Checklist bắt buộc:
- Pin phiên bản `transformers` và `trl` theo notebook.
- Bật `UNSLOTH_VLLM_STANDBY` nếu notebook yêu cầu.
- Xác nhận GPU VRAM đủ cho cấu hình bạn chọn.
- Chạy một prompt inference ngắn trước khi train để chắc model load ổn.

## 3) Thiết kế output contract trước reward
Trước khi viết reward, chốt rõ output format:
- Có tag nào bắt buộc? (ví dụ reasoning/solution tags)
- Output cần parse bằng regex hay parser?
- Điều kiện pass/fail tối thiểu là gì?

Quy tắc vàng: output contract càng mơ hồ, reward hacking càng cao.

## 4) Thiết kế reward theo tầng
Nên dùng nhiều reward nhỏ thay vì một reward “all-in-one”:

### Tầng 1: Format rewards
- Kiểm tra model có tạo đúng khung output hay không.
- Reward nhỏ nhưng ổn định để model vào “đường ray” trước.

### Tầng 2: Correctness rewards
- Exact match cho bài toán có đáp án rõ.
- Approximate reward (ratio, tolerance) cho numeric reasoning.

### Tầng 3: Safety/constraint rewards (nếu cần)
- Với game/code execution: anti-cheating, timeout penalty, invalid function penalty.

## 5) Cấu hình `GRPOConfig` theo nguyên tắc
Bắt đầu với cấu hình bảo thủ:
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=1..4`
- `num_generations=2..6`
- `max_steps` nhỏ để smoke test trước
- `max_prompt_length` và `max_completion_length` rõ ràng

Sau khi ổn định mới tăng:
- `max_steps`
- `num_generations`
- `max_seq_length`

## 6) Quy trình chạy chuẩn
1. Chạy smoke test 10-20 steps.
2. Kiểm tra log reward có tăng/ổn định hay collapse.
3. Nếu ổn, chạy full config.
4. Save LoRA (`model.save_lora(...)` hoặc `save_pretrained`).
5. So sánh inference trước/sau RL trên cùng test set nhỏ.

## 7) Đánh giá chất lượng sau train
Đừng chỉ nhìn một số reward mean. Cần nhìn:
- Tỷ lệ output đúng format.
- Tỷ lệ answer đúng (exact + tolerance).
- Độ dài completion có bị bất thường không.
- Với environment: success rate theo seeds khác nhau.

## 8) Failure patterns và cách xử lý nhanh
- Reward luôn 0: regex/parsing không match output thực tế.
- Reward dao động lớn: reward scale chênh lệch quá mạnh giữa các hàm.
- OOM: giảm `num_generations`, `max_seq_length`, `batch size`; bật quantization.
- Model “học mánh” format nhưng sai đáp án: tăng trọng số correctness reward.
- Timeout trong environment RL: rút ngắn execution window, tăng penalty cho non-terminating strategy.

## 9) Template thực hành 1 vòng thí nghiệm
- Baseline notebook: `...`
- Dataset: `...`
- Reward functions: `...`
- Config GRPO: `...`
- Steps: `...`
- Kết quả trước RL: `...`
- Kết quả sau RL: `...`
- Failure modes: `...`
- Quyết định vòng kế tiếp: `...`

## 10) Best practices để “master” nhanh
- Mỗi vòng chỉ đổi 1 biến lớn (reward hoặc config), không đổi tất cả cùng lúc.
- Luôn giữ một tập prompt eval cố định để so sánh công bằng.
- Ưu tiên độ ổn định reward trước khi tối ưu tốc độ.
- Khi chuyển domain (text -> vision -> planning), giữ nguyên triết lý reward decomposition.
