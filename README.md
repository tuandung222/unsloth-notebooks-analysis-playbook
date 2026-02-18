# Unsloth Notebooks Analysis Playbook

Repo này tổng hợp phân tích chuyên sâu về mã nguồn `unslothai/notebooks` và biến nó thành lộ trình học có hệ thống cho người mới.

Mục tiêu:
- Hiểu kiến trúc repo gốc và workflow sinh notebook/script.
- Nắm chắc các concept cốt lõi: SFT, LoRA/QLoRA, GRPO, DPO/ORPO, RL cho reasoning.
- Biết cách chọn notebook phù hợp theo use-case (chat, vision, tool calling, planning/orchestration).
- Có checklist để đi từ “chạy được” sang “làm chủ + mở rộng”.

## Snapshot mã nguồn gốc
- Nguồn phân tích: `https://github.com/unslothai/notebooks`
- Notebook Colab trong `nb/`: 248 file.
- Notebook template trong `original_template/`: 103 file.
- Python export tương ứng trong `python_scripts/`: 248 file.
- Notebook Kaggle (tiền tố `Kaggle-`): 103 file.
- Notebook liên quan GRPO/RL: 59 file.

## Đọc theo thứ tự
1. `docs/00-overview.md`
2. `docs/01-repo-architecture.md`
3. `docs/02-core-concepts.md`
4. `docs/03-grpo-rl-deep-dive.md`
5. `docs/07-grpo-notebook-analysis.md`
6. `docs/08-grpo-training-playbook.md`
7. `docs/09-grpo-troubleshooting.md`
8. `docs/04-planning-orchestration.md`
9. `docs/05-learning-path.md`
10. `docs/06-contribution-guide.md`

## Bộ tài liệu GRPO mới
- Phân tích notebook chi tiết: `docs/07-grpo-notebook-analysis.md`
- Quy trình train có thể thực hành ngay: `docs/08-grpo-training-playbook.md`
- Sổ tay xử lý lỗi nhanh: `docs/09-grpo-troubleshooting.md`

## Bộ tài liệu chuyên sâu mới (documents_unsloth_mastery)
- Index: `documents_unsloth_mastery/00-index.md`
- DPO/ORPO + quy trình GRPO: `documents_unsloth_mastery/01-dpo-orpo-analysis-and-grpo-process.md`
- FastModel + HF compatibility: `documents_unsloth_mastery/02-fastmodel-internals-hf-compat.md`
- Export playbook (merged, GGUF, TorchAO, ONNX path): `documents_unsloth_mastery/03-unsloth-export-playbook.md`
- QAT playbook: `documents_unsloth_mastery/04-qat-training-playbook.md`
- RL dataset/reward format: `documents_unsloth_mastery/05-rl-dataset-reward-format.md`
- Unsloth + TRL integration: `documents_unsloth_mastery/06-unsloth-trl-rl-integration.md`
- Tool-calling training playbook: `documents_unsloth_mastery/07-tool-calling-training-unsloth.md`

## Nhóm notebook gợi ý cho người mới RL
- Khởi đầu: `Qwen3_(4B)-GRPO.ipynb`, `Llama3.1_(8B)-GRPO.ipynb`
- Nâng cao reward design: `Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb`
- RL theo game/env: `Ministral_3_(3B)_Reinforcement_Learning_Sudoku_Game.ipynb`, `gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb`
- Planning/tool use: `FunctionGemma_(270M)-Multi-Turn-Tool-Calling.ipynb`, `Qwen2.5_Coder_(1.5B)-Tool_Calling.ipynb`, `NeMo-Gym-Multi-Environment.ipynb`

## Nguyên tắc sử dụng repo này
- Dùng tài liệu này để học concept và quy trình.
- Dùng repo gốc để chạy notebook thực tế.
- Mỗi khi repo gốc thay đổi lớn, cập nhật lại snapshot + checklist.
