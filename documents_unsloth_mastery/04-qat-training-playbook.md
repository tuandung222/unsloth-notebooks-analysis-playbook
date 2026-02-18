# 04 - Playbook QAT training với Unsloth (từ cơ bản đến vận hành)

Tài liệu này dựa trên notebook `Qwen3_(4B)_Instruct-QAT.ipynb` và code Unsloth liên quan `qat_scheme`.

---

## A. QAT trong Unsloth là gì?

QAT (Quantization-Aware Training) là huấn luyện với mô phỏng lượng tử hóa trong lúc train để model thích nghi trước khi convert cuối.

Trong Unsloth, QAT xuất hiện ở 2 điểm chính:
1. Cấu hình `qat_scheme` khi gắn LoRA hoặc full-finetuning path.
2. Bước convert cuối qua `torchao.quantization.quantize_(..., QATConfig(step="convert"))`.

---

## B. Pattern notebook QAT của Unsloth

## B1) Load model
Notebook QAT load model bằng `FastLanguageModel.from_pretrained(...)`.

## B2) Gắn LoRA + bật QAT
`FastLanguageModel.get_peft_model(..., qat_scheme="int4", ...)`

Điểm quan trọng:
- Đây là QAT + LoRA path.
- Trong loader source có cảnh báo: một số kiểu `qat_scheme` chỉ hợp lệ với `full_finetuning=True` nếu truyền ở `from_pretrained`; còn QAT với LoRA thì nên truyền trong `get_peft_model`.

## B3) Train SFT
Notebook dùng `SFTTrainer` + dataset hội thoại đã format template.

## B4) Convert fake quant -> real quantized modules
Sau train:
`quantize_(model, QATConfig(step="convert"))`

Bước này giảm overhead fake-quant và chuẩn bị model để export/runtime.

---

## C. Quy trình QAT chuẩn thực chiến

## C1) Phase 0: Chuẩn bị
- Chọn model base phù hợp VRAM.
- Chuẩn hóa dataset chat template.
- Chuẩn bị eval set nhỏ nhưng đại diện.

## C2) Phase 1: Baseline không QAT
- Chạy SFT ngắn (smoke) không QAT.
- Lưu metrics chất lượng/loss làm mốc.

## C3) Phase 2: QAT run
- Bật `qat_scheme` (ví dụ `int4`).
- Hạ batch/sequence nếu cần để ổn định.
- Theo dõi kỹ loss và chất lượng đầu ra.

## C4) Phase 3: Convert + export
- Gọi convert step của QAT.
- Export theo mục tiêu deploy (`torchao`, merged, hoặc nhánh khác).

## C5) Phase 4: So sánh hậu lượng tử
- So sánh trước/sau conversion:
  - quality regression,
  - latency,
  - VRAM/RAM footprint.

---

## D. Hyperparameter khuyến nghị khởi đầu

- `learning_rate` thấp hơn run SFT thường nếu thấy dao động lớn.
- `max_steps` ngắn cho smoke test trước.
- `gradient_accumulation_steps` để giữ effective batch ổn định.
- ưu tiên logging dày giai đoạn đầu để bắt sớm divergence.

---

## E. Failure modes hay gặp trong QAT

1. **Train chạy nhưng chất lượng tụt mạnh sau convert**
   - Thường do data/template chưa sạch hoặc QAT schedule quá gắt.
2. **OOM**
   - Giảm `max_seq_length`, batch, hoặc chuyển model nhỏ hơn.
3. **Export xong nhưng inference lệch hành vi**
   - Kiểm tra tokenizer/chat template parity và EOS settings.

---

## F. Checklist trước khi đưa QAT vào production

- Có baseline non-QAT để đối chiếu.
- Có eval fixed prompts + pass/fail tiêu chí.
- Có benchmark latency/memory thực tế trên target hardware.
- Có rollback plan về merged non-QAT hoặc checkpoint trước.

---

## G. Kết luận

QAT với Unsloth là đường đi hợp lý khi bạn cần trade-off tốt giữa hiệu năng và footprint.
Nhưng bắt buộc làm theo chu kỳ: baseline -> QAT train -> convert -> benchmark parity.
Nếu bỏ qua bước parity test, rủi ro chất lượng sau deploy rất cao.

