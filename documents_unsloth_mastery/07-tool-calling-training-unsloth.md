# 07 - Quy trình training LLM cho tool calling bằng Unsloth

Tài liệu này tách rõ:
1. Notebook nào đang thiên về inference/demo.
2. Cách xây pipeline train tool-calling đúng chuẩn với Unsloth.

---

## A. Phân tích notebook tool-calling hiện có

## A1) `Qwen2.5_Coder_(1.5B)-Tool_Calling.ipynb`
Notebook này chủ yếu demo:
- khai báo tool schema/function definitions,
- prompt model để sinh JSON call,
- grammar-constrained decoding để ép output JSON,
- parse JSON và gọi function Python.

Điểm chính: đây là **framework demo orchestration + inference**, chưa phải full training notebook.

## A2) `FunctionGemma_(270M)-Multi-Turn-Tool-Calling.ipynb`
Notebook tập trung multi-turn tool use:
- define tools,
- parse tool call markers,
- thực thi tool rồi feed kết quả lại cuộc hội thoại.

Cũng nghiêng về **agentic inference loop** hơn là huấn luyện từ dataset tool-calling.

---

## B. Vậy train tool-calling nên làm thế nào?

## B1) Mục tiêu train
Model cần học 2 năng lực:
1. **Tool selection**: chọn đúng tool.
2. **Argument grounding**: điền args đúng schema và ngữ cảnh.

## B2) Dữ liệu cần có
Mỗi mẫu nên gồm:
- hội thoại đầu vào,
- tool schema khả dụng,
- tool call mục tiêu (name + arguments),
- (tuỳ chọn) tool result + assistant final response.

## B3) Format mẫu khuyến nghị
Nên chuẩn hóa theo dạng conversational messages, có role rõ ràng (`system/user/assistant/tool`).

---

## C. Pipeline training đề xuất (Unsloth)

## C1) Stage 1 - SFT cho tool-call format
- Load model bằng Unsloth.
- Áp chat template thống nhất.
- Train SFT để model học cú pháp tool call và format output ổn định.

## C2) Stage 2 - Preference/RL (tuỳ mục tiêu)
- Dùng DPO/ORPO nếu có cặp output tốt/xấu.
- Dùng GRPO nếu có reward tự động đánh giá đúng tool + đúng args.

## C3) Stage 3 - Evaluation đặc thù tool-calling
Đo tối thiểu:
- Tool name accuracy,
- Argument exact match rate,
- JSON/schema validity,
- End-to-end success rate (tool call + final answer đúng).

---

## D. Reward design cho tool-calling GRPO

Có thể tách 4 reward:
1. `schema_reward`: output parse được + đúng schema.
2. `tool_choice_reward`: chọn đúng tool theo ground truth.
3. `argument_reward`: args đúng key/type/value.
4. `execution_reward`: gọi tool chạy thành công và ra kết quả mong muốn.

Nếu chỉ dùng 1 reward tổng hợp, rất khó debug khi model fail.

---

## E. Lưu ý quan trọng khi triển khai

- Chat template phải đồng nhất giữa train và infer.
- Tool schema versioning phải rõ (thêm/xóa field là breaking change).
- Parser phải fail-fast, tránh parser “nuông chiều” làm reward ảo.
- Phải có tập test adversarial (prompt mơ hồ, thông tin thiếu, args nhiễu).

---

## F. Khung thực hành gợi ý cho người mới

1. Bắt đầu với 2-3 tool đơn giản, schema nhỏ.
2. Làm SFT để model nói đúng format trước.
3. Thêm eval tự động cho exact match.
4. Sau khi ổn định mới thêm GRPO để tối ưu độ đúng/ngắn gọn/an toàn.

---

## G. Kết luận

Notebook tool-calling hiện tại của Unsloth cho bạn “khung orchestration” rất tốt.
Để thành pipeline training thực thụ, bạn cần bổ sung data curriculum + eval + reward discipline.
Làm đúng thứ tự SFT -> (DPO/ORPO/GRPO) -> robust eval sẽ cho kết quả bền hơn nhiều.

