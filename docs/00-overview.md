# 00 - Tổng quan nhanh

## Đây là repo gì?
`unslothai/notebooks` là một “catalog thực chiến” các notebook fine-tuning/inference cho rất nhiều model family (Llama, Qwen, Gemma, Mistral, Phi, GPT-OSS...) trên Colab/Kaggle.

## Điểm mạnh
- Coverage rộng: text, vision, embedding, STT/TTS, OCR, tool-calling, RL.
- Tối ưu cho GPU phổ biến (T4/L4/A100) với setup tự động.
- Có cặp notebook tương ứng cho nhiều môi trường (Colab, Kaggle, HF Course).

## Điểm cần lưu ý
- Repo rất lớn, dễ “ngợp” nếu không có lộ trình.
- Nhiều notebook tương tự nhau, chủ yếu khác model/dataset/hyperparameter.
- Nội dung README chủ yếu là danh mục link; logic thực sự nằm trong notebook và script cập nhật.

## Bản đồ đọc nhanh
- Nếu bạn mới học RL cho LLM: đọc `03-grpo-rl-deep-dive.md` -> `07-grpo-notebook-analysis.md` -> `08-grpo-training-playbook.md`.
- Nếu bạn đang vướng lỗi lúc train: đọc `09-grpo-troubleshooting.md`.
- Nếu bạn cần planning/orchestration: đọc `04-planning-orchestration.md`.
- Nếu bạn muốn đóng góp/cập nhật notebook: đọc `06-contribution-guide.md`.
