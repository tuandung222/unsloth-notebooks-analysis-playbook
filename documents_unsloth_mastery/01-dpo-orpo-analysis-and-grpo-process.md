# 01 - Phân tích DPO/ORPO notebooks và quy trình GRPO bằng Unsloth

Tài liệu này làm 2 việc:
1. Phân tích sâu notebook DPO/ORPO trong `unslothai/notebooks`.
2. Dùng insight đó để chuẩn hóa quy trình GRPO (từ tổng quát -> chi tiết -> thực hành).

---

## A. DPO notebook (`Zephyr_(7B)-DPO.ipynb`) phân tích chi tiết

### A1) Pattern triển khai
Notebook DPO của Unsloth có pattern nhất quán:
- Gọi `PatchDPOTrainer()` trước khi train.
- Load model qua `FastLanguageModel.from_pretrained(...)`.
- Chuẩn hóa dataset preference về cột:
  - `prompt`
  - `chosen`
  - `rejected`
- Train bằng `DPOTrainer(..., args=DPOConfig(...))`.

### A2) Điểm quan trọng trong data prep
Notebook này dùng helper từ Alignment Handbook để transform dữ liệu hội thoại về format DPO.
Ý chính:
- Với conversation preference, model cần tách prompt và completion rõ ràng.
- Cần map tên cột về format TRL expects (`prompt/chosen/rejected`).
- Prompt explicit được ưu tiên vì dễ kiểm soát hơn implicit prompt.

### A3) Ý nghĩa kỹ thuật
DPO là preference optimization offline:
- Không cần reward function runtime như GRPO.
- Không cần rollout online mỗi step.
- Không chạy environment loop.

Kết quả: đơn giản hơn RL online, ổn định hơn cho giai đoạn đầu alignment.

---

## B. ORPO notebook (`Llama3_(8B)-ORPO.ipynb`) phân tích chi tiết

### B1) Pattern triển khai
- Load model qua `FastLanguageModel` + LoRA.
- Chuẩn hóa dataset preference theo `prompt/chosen/rejected`.
- Train bằng `ORPOTrainer(..., args=ORPOConfig(...))`.

### B2) Điểm khác DPO
ORPO trainer là “reference-model-free” trong cách tối ưu odds-ratio preference:
- Tối ưu theo style favored/disfavored trực tiếp.
- Practical benefit: pipeline gọn, tiết kiệm bước so với một số setup DPO có ref-model riêng.

### B3) Insight từ notebook
ORPO notebook nhấn mạnh format dataset rõ ràng và reproducible:
- Prompt template phải deterministic.
- `chosen` và `rejected` nên cùng phân phối nhiệm vụ.
- EOS handling phải nhất quán để không méo loss.

---

## C. Từ DPO/ORPO sang GRPO: tư duy chuyển đổi đúng

DPO/ORPO dạy bạn 3 thứ cực quan trọng trước khi vào GRPO:
1. **Data contract** rõ ràng.
2. **Output contract** rõ ràng.
3. **Evaluation contract** rõ ràng.

GRPO khó hơn vì reward được tính online trên completion model tự sinh ra. Nếu 3 contract trên không vững, GRPO sẽ nhanh bị reward hacking hoặc collapse.

---

## D. Quy trình GRPO bằng Unsloth (chuẩn thực chiến)

## D1) Bức tranh tổng quát
Một vòng GRPO chuẩn với Unsloth:
1. Chọn model và chế độ tải (`4bit/8bit/fp8/16bit`).
2. Gắn LoRA (`get_peft_model`).
3. Chuẩn hóa dataset về prompt-only + metadata đánh giá.
4. Thiết kế reward functions theo tầng.
5. Cấu hình `GRPOConfig`.
6. Train với `GRPOTrainer`.
7. Đánh giá before/after và lưu artifact.

## D2) Chọn notebook baseline đúng mục tiêu
- GRPO text basic: `Llama3.1_(8B)-GRPO.ipynb`
- GRPO base-model + pre-SFT: `Qwen3_(4B)-GRPO.ipynb`
- GRPO advanced: `Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb`
- GRPO vision: `Gemma3_(4B)-Vision-GRPO.ipynb`
- GRPO planning/game: `gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb`

## D3) Data chuẩn cho GRPO
Theo TRL docs, GRPO trainer mong đợi **prompt-only** dataset là chính.
Thực tế Unsloth notebooks thường dùng:
- cột `prompt` (string hoặc message list),
- cột `answer` / `ground_truth` / task metadata để reward đọc.

## D4) Reward design (cấu trúc khuyến nghị)
Nên tách reward thành nhiều hàm:
- **Format reward**: đúng cấu trúc output.
- **Correctness reward**: đúng nội dung.
- **Approx reward**: gần đúng (numeric tolerance).
- **Safety reward** (nếu có tool/env): anti-cheat, timeout, invalid action.

Tổng reward = tổng có trọng số của các reward functions.

## D5) Config khởi đầu an toàn
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=1..4`
- `num_generations=2..6`
- `max_prompt_length` kiểm soát rõ
- `max_completion_length` đủ để reasoning nhưng không gây OOM
- chạy `max_steps` nhỏ để smoke test trước

## D6) Vòng chạy chuẩn
1. Smoke run (10-30 steps).
2. Kiểm tra logs:
- reward mean,
- reward std,
- completion length,
- clipped ratio.
3. Nếu ổn, chạy full experiment.
4. Save LoRA và benchmark before/after.

## D7) Failure modes thường gặp
- Reward luôn thấp do parser mismatch.
- Reward tăng nhưng quality không tăng (reward hacking).
- OOM do `num_generations * completion_length` quá cao.
- Environment RL timeout do strategy không terminate.

## D8) Checklist cuối mỗi vòng thí nghiệm
- Dataset snapshot version.
- Reward function version.
- Config snapshot.
- Prompt-eval set cố định.
- Metrics + qualitative examples.
- Quyết định thay đổi cho vòng tiếp theo (chỉ đổi 1 biến lớn).

---

## E. Lộ trình học đề xuất
1. DPO notebook để hiểu preference data hygiene.
2. ORPO notebook để hiểu monolithic preference optimization.
3. GRPO text để nắm online reward loop.
4. GRPO vision/game để mở rộng sang multimodal/planning.

Khi đi đúng thứ tự này, bạn sẽ có nền alignment tốt trước khi vào RL online phức tạp.
