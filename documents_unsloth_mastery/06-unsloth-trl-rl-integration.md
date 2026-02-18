# 06 - Playbook phối hợp Unsloth + TRL cho RL training

Mục tiêu: đưa một pipeline từ prototype notebook sang quy trình có thể lặp lại cho team.

---

## A. Ai làm gì trong stack này?

- **Unsloth**: load model nhanh, tối ưu memory/perf, patch train/infer paths.
- **TRL**: trainer RL/preference (DPO/ORPO/GRPO...), config, logging loop.
- **Datasets + reward code của bạn**: quyết định chất lượng thực tế.

=> Unsloth + TRL là xương sống kỹ thuật; data/reward là linh hồn chất lượng.

---

## B. Kiến trúc tích hợp chuẩn

1. Load model/tokenizer bằng Unsloth.
2. Gắn LoRA (hoặc full finetune tùy use-case).
3. Chuẩn hóa dataset đúng format trainer TRL mục tiêu.
4. Khai báo reward functions (với GRPO/online methods).
5. Khởi tạo trainer TRL (`DPOTrainer`, `ORPOTrainer`, `GRPOTrainer`).
6. Train + log + eval + export.

---

## C. Tích hợp DPO/ORPO

## C1) Data contract
`prompt/chosen/rejected` là contract dễ kiểm soát nhất.

## C2) Trainer contract
- DPO: patch DPO trainer như notebook mẫu trước khi train.
- ORPO: dùng `ORPOTrainer` với beta/max lengths phù hợp.

## C3) Khi nào chọn cái nào?
- DPO: phổ biến, tài nguyên docs rộng.
- ORPO: pipeline gọn hơn trong một số setup (không phải quản ref-model riêng theo cách truyền thống).

---

## D. Tích hợp GRPO

## D1) Data + reward
- Dataset chủ yếu prompt-only.
- Reward function đọc completion online và metadata đi kèm.

## D2) Config khởi tạo an toàn
- batch nhỏ,
- `num_generations` thấp ban đầu,
- completion length vừa đủ,
- smoke test 10-30 steps trước full run.

## D3) Performance
Online RL bị bottleneck ở generation; dùng path tối ưu của Unsloth/vLLM khi phù hợp giúp giảm thời gian train mạnh.

---

## E. Chuẩn hóa experiment cho team

## E1) Cấu trúc thư mục gợi ý
- `configs/` (yaml/json cho từng run)
- `rewards/` (hàm reward tách file)
- `data_prep/`
- `eval/`
- `exports/`

## E2) Versioning
Mỗi run nên khóa:
- dataset snapshot,
- reward version,
- trainer config,
- base model commit/version.

---

## F. Quy trình debug khi loss/reward bất thường

1. Test reward function độc lập (không train).
2. In sample completion + reward chi tiết.
3. Kiểm tra template/tokenizer mismatch.
4. Giảm complexity (1 reward trước, rồi cộng dần).
5. Chỉ đổi 1 biến lớn mỗi vòng.

---

## G. Quy trình productionization

1. Chốt checkpoint tốt nhất theo eval task.
2. Export `merged` làm chuẩn.
3. Tạo artifact deploy phụ (GGUF/TorchAO/ONNX path).
4. Chạy benchmark latency + quality + stability.
5. Viết model card nội bộ: giới hạn, prompt style, failure patterns.

---

## H. Kết luận

Phối hợp Unsloth + TRL hiệu quả khi bạn tách rõ trách nhiệm:
- Unsloth lo hiệu năng và tính thực dụng,
- TRL lo trainer logic chuẩn,
- team của bạn chịu trách nhiệm data/reward discipline.

Nếu giữ được kỷ luật này, pipeline RL sẽ mở rộng tốt từ notebook lên production.

