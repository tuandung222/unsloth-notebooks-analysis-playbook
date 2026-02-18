# 02 - FastModel của Unsloth là gì, inference ra sao, và tương thích Hugging Face như thế nào?

Tài liệu này trả lời 3 câu hỏi thực chiến:
1. `FastModel/FastLanguageModel` thực chất là gì trong kiến trúc Unsloth.
2. Inference flow thực sự chạy như thế nào.
3. Mức độ tương thích với API training/inference của Hugging Face (transformers/PEFT/TRL).

---

## A. FastModel/FastLanguageModel: bản chất kiến trúc

## A1) Không phải model architecture mới
`FastLanguageModel` không tạo ra một architecture mới kiểu “Unsloth-Transformer”.
Nó là lớp loader/patching tối ưu cho các model HF phổ biến (Llama, Qwen, Mistral, Gemma...).

## A2) Vai trò chính
Trong `unsloth/models/loader.py`, `FastLanguageModel.from_pretrained(...)` làm các việc:
- chọn backend/model class phù hợp,
- cấu hình quantization mode (`4bit`, `8bit`, `fp8`, `16bit`...),
- gắn các hook tối ưu train/infer,
- và trong một số nhánh sẽ chuyển sang `FastModel.from_pretrained(...)` (ví dụ `load_in_8bit`, `full_finetuning`, hoặc có `qat_scheme`).

=> Nói ngắn gọn: `FastLanguageModel` là “entrypoint thông minh” cho LLM use-case.

## A3) `for_training` / `for_inference`
Trong `unsloth/models/llama.py` (và vision variant), model được gắn method:
- `model.for_training(...)`
- `model.for_inference(...)`

Mục tiêu là chuyển state nội bộ về đúng chế độ, thay vì người dùng tự chắp vá nhiều cờ.

## A4) patch `generate`
Unsloth patch `model.generate` thành wrapper nhanh (`unsloth_fast_generate`) trong nhiều trường hợp.
Do đó API call vẫn giống HF (`model.generate(...)`), nhưng bên trong đã được tối ưu.

---

## B. Inference flow thực tế

## B1) Flow chuẩn
1. Load model bằng `FastLanguageModel.from_pretrained(...)`.
2. Nếu vừa train xong, gọi `FastLanguageModel.for_inference(model)`.
3. Chuẩn bị input qua tokenizer/chat template.
4. Gọi `model.generate(...)` như transformers chuẩn.

## B2) Vì sao vẫn “giống Hugging Face”
Unsloth chủ động giữ contract API quen thuộc:
- tokenizer và prompt formatting vẫn theo HF ecosystem,
- generate signature giữ gần chuẩn,
- output decode và post-processing không buộc đổi framework.

Kết quả: team có thể migrate dần từ HF thuần sang Unsloth mà ít sửa code nghiệp vụ.

## B3) vLLM path (khi dùng online RL/sinh mẫu lớn)
Loader và RL patch của Unsloth có các nhánh tích hợp vLLM để tăng tốc generation ở online training (GRPO/online methods).
Điểm chính: tăng throughput phần rollout/generation vốn là bottleneck.

---

## C. Tương thích với HF training API

## C1) Với PEFT/LoRA
Pattern chính trong notebook Unsloth:
- load bằng `FastLanguageModel.from_pretrained(...)`,
- gắn adapter bằng `FastLanguageModel.get_peft_model(...)`.

Về mặt tư duy vẫn là LoRA flow quen thuộc của PEFT, nhưng Unsloth thêm tối ưu memory/perf.

## C2) Với TRL trainers
Notebook DPO/ORPO/GRPO dùng trực tiếp `DPOTrainer`, `ORPOTrainer`, `GRPOTrainer` của TRL.
Điều này cho thấy mức tương thích cao ở trainer-level, miễn là data format đúng chuẩn TRL.

## C3) Với transformers Trainer/SFTTrainer
SFT notebook dùng `SFTTrainer` (TRL) trên model/tokenizer do Unsloth load.
Tức là contract training loop không bị phá, chủ yếu đổi layer “model loading + optimization”.

---

## D. Tương thích với HF inference API

## D1) Mức tương thích thực dụng
- Cùng tokenizer paradigm.
- Cùng `generate`-centric flow.
- Có thể stream bằng `TextStreamer` trong nhiều notebook.

## D2) Điểm cần chú ý
- Trước infer thường nên gọi `for_inference`.
- Nếu dùng chat template khác nhau giữa train/infer, chất lượng giảm mạnh dù code vẫn chạy.
- Với RL và vLLM, cần kiểm tra version compatibility theo stack hiện tại.

---

## E. Kết luận kỹ thuật

`FastModel/FastLanguageModel` của Unsloth nên hiểu là:
- một lớp “performance-aware compatibility layer” trên top Hugging Face ecosystem,
- giữ API gần chuẩn HF để giảm migration cost,
- đồng thời thêm patch/hook để tăng hiệu năng train/infer, đặc biệt trong workload RL online.

