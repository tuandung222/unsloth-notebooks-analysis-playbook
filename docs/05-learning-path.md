# 05 - Lộ trình học 30 ngày

## Tuần 1: Nền tảng
- Chạy 1 notebook conversational + 1 notebook SFT cơ bản.
- Đọc và hiểu pattern install/model/data/train/save.
- Ghi chú các biến quan trọng ảnh hưởng VRAM/throughput.

## Tuần 2: RL cơ bản
- Chạy `Qwen3_(4B)-GRPO.ipynb` end-to-end.
- Phân tích từng reward function và biểu đồ reward theo step.
- Thử thay đổi 1 reward term và quan sát tác động.

## Tuần 3: RL nâng cao + environment
- Chạy 1 notebook game-like RL (Sudoku/2048).
- Thêm evaluator đơn giản cho output quality.
- So sánh trước/sau RL trên một bộ test nhỏ.

## Tuần 4: Planning/orchestration
- Chạy notebook tool-calling và kiểm tra schema robustness.
- Thiết kế một mini task planner (2-3 tools) và đo success rate.
- Viết báo cáo lessons learned + failure taxonomy.

## Deliverables cuối lộ trình
- 1 repo experiment cá nhân.
- 1 tài liệu kỹ thuật: setup, reward design, metrics, failure cases.
- 1 checklist tái sử dụng cho dự án RL LLM mới.
