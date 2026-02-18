# 02 - Core concepts cần nắm

## Mô hình huấn luyện phổ biến trong repo
- `SFT` (Supervised Fine-Tuning): học từ cặp input-output chuẩn.
- `LoRA/QLoRA`: adapter-based fine-tuning tiết kiệm VRAM.
- `DPO/ORPO`: preference-based alignment.
- `GRPO`: reinforcement learning trực tiếp trên reward functions.

## Pattern kỹ thuật xuất hiện lặp lại
- `FastLanguageModel.from_pretrained(...)`: tải model với tối ưu Unsloth.
- `FastLanguageModel.get_peft_model(...)`: gắn LoRA adapters.
- Install branch theo môi trường: local/Colab/Kaggle.
- Pin version thư viện để giảm vỡ compatibility.

## Taxonomy notebook theo use-case
- Conversational / Alpaca / Text completion.
- Vision multimodal, OCR, audio (TTS/STT).
- Embedding & retrieval.
- RL/GRPO cho reasoning, game-like environments.
- Tool-calling và action generation.

## Tư duy học hiệu quả
- Học theo “pattern” thay vì theo từng model.
- Bóc tách một notebook thành 6 pha: install -> model -> data -> train -> eval -> save/deploy.
- Sau khi hiểu một pattern, mới chuyển model family khác.
