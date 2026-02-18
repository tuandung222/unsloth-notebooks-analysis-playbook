# 04 - LLM for Planning & Orchestration

## Notebook liên quan
- `FunctionGemma_(270M)-Multi-Turn-Tool-Calling.ipynb`
- `FunctionGemma_(270M)-Mobile-Actions.ipynb`
- `Qwen2.5_Coder_(1.5B)-Tool_Calling.ipynb`
- `NeMo-Gym-Multi-Environment.ipynb`
- `CodeForces-cot-Finetune_for_Reasoning_on_CodeForces.ipynb`

## Concept map
- Tool-calling: model sinh schema-call đúng định dạng.
- Planning: model sinh chuỗi bước trung gian có kiểm soát.
- Orchestration: nhiều tool/environment phối hợp theo policy.
- Self-check / feedback loop: dùng reward hoặc verifier đánh giá kế hoạch.

## Cách tận dụng repo gốc cho bài toán planning
- Bắt đầu với notebook có prompt + output schema rõ ràng.
- Chuyển reward từ “answer correctness” sang “plan validity + tool-use correctness”.
- Thêm synthetic tasks để tăng độ đa dạng trạng thái và edge-cases.
- Đánh giá theo 3 trục: success rate, step efficiency, tool call precision.

## Khung thực nghiệm đề xuất
1. Baseline SFT cho format tuân thủ schema.
2. RL (GRPO) để tối ưu policy chọn hành động.
3. Add environment simulator (game/task API).
4. Theo dõi failure mode theo taxonomy (hallucinated tool, wrong order, non-terminating plan).
