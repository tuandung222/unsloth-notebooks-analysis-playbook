# 07 - Phân tích chi tiết các notebook GRPO tiêu biểu

Mục tiêu của tài liệu này là bóc tách chi tiết theo từng notebook để bạn hiểu “vì sao nó được viết như vậy”, không chỉ “chạy lệnh nào”.

## A. `Qwen3_(4B)-GRPO.ipynb` (base model + pre-SFT)

### Vì sao notebook này quan trọng?
Đây là mẫu chuẩn cho tình huống bạn bắt đầu từ *base model* chưa quen chat format reasoning.

### Luồng chính
1. Load `unsloth/Qwen3-4B-Base` + LoRA.
2. Tạo custom chat template với tags:
- `<start_working_out>` / `<end_working_out>`
- `<SOLUTION>` / `</SOLUTION>`
3. Pre-SFT ngắn trên `unsloth/OpenMathReasoning-mini` để model học format.
4. Chuyển sang GRPO trên `open-r1/DAPO-Math-17k-Processed`.
5. Reward gồm:
- `match_format_exactly`
- `match_format_approximately`
- `check_answer`
- `check_numbers`
6. Train bằng `GRPOTrainer`, sau đó save LoRA và test trước/sau RL.

### Insight
- Pre-SFT trước GRPO giúp giảm thời gian “đánh nhau với format”, để reward tập trung vào chất lượng lời giải.
- Dùng cả exact + approximate reward giúp gradient ổn định hơn khi model chưa hội tụ.

## B. `Llama3.1_(8B)-GRPO.ipynb` (instruct model + reward nhẹ)

### Khi nào dùng pattern này?
Khi model đã là instruct model và có thể trả lời có cấu trúc tương đối tốt.

### Reward stack trong notebook
- `xmlcount_reward_func`
- `soft_format_reward_func`
- `strict_format_reward_func`
- `int_reward_func`
- `correctness_reward_func`

### Insight
- Nhánh này ưu tiên “dẫn dắt behavior” bằng reward shape nhẹ, thay vì ép quá nhiều regex cứng ngay từ đầu.
- Phù hợp làm notebook nhập môn GRPO.

## C. `Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb` (reward design nâng cao)

### Điểm khác biệt
- Reward phạt/thưởng chi tiết hơn cho câu trả lời sai gần/sai xa.
- Có kiểm soát prompt length trước train để tránh truncation side-effect.
- Hyperparameter cho train dài hơn (`max_steps` cao hơn) để quan sát reward dynamics rõ.

### Insight
- Đây là notebook nên dùng để học “nghệ thuật reward engineering”.

## D. `Qwen3_8B_FP8_GRPO.ipynb` (scale lên model lớn hơn)

### Điểm khác biệt kỹ thuật
- `load_in_fp8=True`.
- Vẫn giữ pipeline gần như Qwen3-4B-GRPO, nhưng thêm tối ưu memory (`UNSLOTH_VLLM_STANDBY`).
- Một số cell gọi `model.vllm_engine.sleep()` trước train để giải phóng VRAM.

### Insight
- Đây là bước chuyển từ “đúng pipeline” sang “đúng pipeline + đủ tài nguyên”.

## E. `Gemma3_(4B)-Vision-GRPO.ipynb` (multimodal RL)

### Khác biệt lớn so với text-only
- Dùng `FastVisionModel`.
- Dataset chứa cả ảnh + text prompt.
- Cần resize/chuẩn RGB trước khi train.
- Reward parsing dựa trên tags reasoning/solution trong output multimodal.

### Insight
- Nếu preprocessing ảnh không ổn định, reward sẽ nhiễu ngay cả khi policy đúng.

## F. `gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb` (planning/orchestration)

### Đây không còn là “answer matching” đơn giản
Model phải sinh strategy function, rồi strategy đó bị thực thi trong environment game.

### Reward stack đặc thù
- `function_works`: function hợp lệ hay không.
- `no_cheating`: phát hiện import/module cheat.
- `strategy_succeeds`: chạy game thực tế, reward theo outcome.

### Insight
- Đây là dạng RL gần với agent planning/orchestration:
  reward phụ thuộc hành vi trong môi trường, không chỉ so string output.

## G. Kết luận phân tích
Nếu bạn học đúng thứ tự, bạn sẽ đi theo đường cong năng lực rõ ràng:
- Bước 1: XML/format reward (Llama3.1).
- Bước 2: Base model + pre-SFT + GRPO (Qwen3-4B).
- Bước 3: Reward engineering nâng cao (Advanced Llama).
- Bước 4: Scale tài nguyên (FP8).
- Bước 5: Mở rộng domain (Vision, Game/OpenEnv).
