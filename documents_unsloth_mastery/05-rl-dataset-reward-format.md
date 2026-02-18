# 05 - Dataset format và reward function format cho RL training với Unsloth + TRL

Mục tiêu: chuẩn hóa “data contract” và “reward contract” để chạy RL ổn định, tránh reward hacking.

---

## A. Dataset format theo từng thuật toán

## A1) DPO / ORPO (preference optimization offline)
Format khuyến nghị (explicit prompt):
- `prompt`
- `chosen`
- `rejected`

Đây là format được notebook DPO/ORPO của Unsloth và TRL docs dùng nhiều nhất.

## A2) GRPO (online RL)
Format lõi:
- `prompt` (bắt buộc)

Có thể kèm metadata phục vụ reward:
- `answer` / `ground_truth`
- `task_type`
- `tool_schema`
- `difficulty`

Ý tưởng: GRPO trainer sinh completion online, reward function đọc completion + metadata để chấm.

## A3) Conversational vs Standard
TRL hỗ trợ cả chuẩn text thường và conversational messages.
Nhưng thực chiến nên chuẩn hóa sớm một format thống nhất cho team (không trộn lẫn lung tung giữa các project).

---

## B. Reward function format cho GRPO

## B1) Signature thực dụng
Reward function thường nhận:
- prompts,
- completions,
- các cột metadata từ dataset qua `**kwargs`.

Trả về:
- danh sách `float` reward (mỗi completion một điểm).

## B2) Nguyên tắc thiết kế reward
1. **Deterministic**: cùng input -> cùng reward.
2. **Local & explainable**: lỗi parser phải debug được.
3. **Bounded**: tránh scale quá lớn gây gradient bất ổn.
4. **Composable**: tách nhiều reward nhỏ thay vì 1 hàm khổng lồ.

## B3) Mẫu cấu trúc reward nhiều tầng
- `format_reward`: đúng schema/cấu trúc output.
- `correctness_reward`: đúng đáp án.
- `safety_reward`: phạt hành vi sai policy.
- `efficiency_reward`: phạt output quá dài/lan man.

Tổng reward có thể là weighted sum.

---

## C. JSON schema gợi ý cho dataset RL

Ví dụ một sample:

```json
{
  "prompt": "Giải bài toán ...",
  "ground_truth": "42",
  "task_type": "math",
  "difficulty": "medium",
  "meta": {"source": "gsm8k"}
}
```

Với tool-calling RL:

```json
{
  "prompt": "Đặt lịch họp...",
  "tool_schema": [...],
  "expected_tool": "create_calendar_event",
  "constraints": {"timezone": "US/Eastern"}
}
```

---

## D. Data validation trước khi train

Checklist bắt buộc:
- `prompt` không rỗng.
- `ground_truth` có format nhất quán theo task.
- ký tự control/token lỗi đã được sanitize.
- chiều dài prompt trong giới hạn context.
- split train/eval không rò rỉ.

---

## E. Reward validation trước khi train

Trước khi chạy GRPO full:
1. Chạy reward function trên batch dữ liệu mẫu + output giả.
2. In histogram reward để xem range/variance.
3. Test case “đúng rõ ràng”, “sai rõ ràng”, “gần đúng”.
4. Đảm bảo không có exception im lặng (silent fail).

---

## F. Mapping nhanh: thuật toán -> data format

- `DPOTrainer`: preference (`prompt/chosen/rejected` ưu tiên explicit prompt).
- `ORPOTrainer`: preference (`prompt/chosen/rejected`).
- `GRPOTrainer`: prompt-only + metadata cho reward.

---

## G. Anti-pattern cần tránh

- Reward phụ thuộc parser mơ hồ (regex yếu, dễ false positive).
- Dùng dữ liệu “chosen/rejected” lẫn logic GRPO mà không tách rõ mục tiêu.
- Trộn nhiều chat template trong cùng 1 run.
- Reward scale không chuẩn hóa khiến 1 hàm lấn át toàn bộ hàm còn lại.

---

## H. Kết luận

Trong RL training với Unsloth, chất lượng không chỉ nằm ở model size hay GPU.
Nó nằm trước hết ở hợp đồng dữ liệu + hợp đồng reward.
Nếu hai hợp đồng này chặt, GRPO mới hội tụ bền và có thể mở rộng production.

