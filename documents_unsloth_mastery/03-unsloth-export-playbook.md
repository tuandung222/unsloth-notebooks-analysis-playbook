# 03 - Export model với Unsloth: merged, GGUF, TorchAO, ONNX và chiến lược deploy

Tài liệu này tổng hợp các đường export quan trọng sau khi train bằng Unsloth.

---

## A. Bức tranh tổng quát

Trong `unsloth/save.py`, Unsloth patch thêm nhiều method lên model object:
- `save_pretrained_merged` / `push_to_hub_merged`
- `save_pretrained_gguf` / `push_to_hub_gguf`
- `save_pretrained_torchao`

Điểm mạnh: từ một checkpoint train (thường là LoRA), bạn có nhiều đích deploy khác nhau.

---

## B. Export phổ biến nhất: merged Hugging Face model

## B1) Khi dùng
Dùng khi bạn muốn chạy bằng stack HF/TGI/vLLM chuẩn.

## B2) Ý tưởng
- Merge adapter vào base model,
- xuất ra format HF quen thuộc (weights + tokenizer + config),
- dễ dùng nhất cho ecosystem Python/serving hiện đại.

## B3) Trade-off
- File lớn hơn LoRA-only.
- Nhưng inference/runtime đơn giản hơn vì không phải load adapter riêng.

---

## C. Export GGUF (llama.cpp/Ollama ecosystem)

## C1) Khi dùng
Dùng khi target là edge/local inference qua `llama.cpp` hoặc công cụ phụ thuộc GGUF.

## C2) Những gì Unsloth làm
Luồng GGUF trong `save.py` gồm các bước kiểu:
1. lưu model ở dạng trung gian,
2. gọi converter HF->GGUF,
3. quantize GGUF theo method (Q8_0, Q4_K_M, ...),
4. trả về danh sách file GGUF đã tạo.

## C3) Lưu ý thực chiến
- GGUF conversion và quantization khá nặng disk + thời gian.
- Nên giữ 1 bản full/merged chất lượng cao để làm source “gốc”, rồi tạo nhiều biến thể GGUF từ đó.
- Cần kiểm tra chat template/tokenizer parity để tránh lệch hành vi.

---

## D. Export TorchAO (đặc biệt quan trọng với QAT)

## D1) Khi dùng
Dùng cho flow quantization-aware hoặc torchao deployment path.

## D2) Liên hệ với QAT
`save_pretrained_torchao` có logic riêng cho model đã train với `qat_scheme`.
Nghĩa là export không chỉ “ghi file”, mà còn đảm bảo mapping đúng với trạng thái quantization của model.

## D3) Gợi ý
Nếu pipeline của bạn dùng QAT + torchao runtime, nên ưu tiên kiểm thử end-to-end theo đường này thay vì ép qua đường khác quá sớm.

---

## E. ONNX: hiện trạng và cách làm an toàn

## E1) Hiện trạng trong mã Unsloth
Trong source hiện tại, không thấy helper ONNX first-class kiểu `save_pretrained_onnx` của Unsloth.

## E2) Cách làm khuyến nghị
1. Xuất model merged HF trước.
2. Dùng toolchain ONNX bên ngoài (transformers export / optimum / onnxruntime tooling) để convert.
3. Chạy bộ test parity (logits/generation sanity) giữa HF và ONNX.

=> Tức là ONNX nên xem là “downstream conversion path”, không phải core export API của Unsloth.

---

## F. Export “nhiều định dạng” theo pipeline chuẩn

## F1) Quy trình đề xuất
1. Train -> lưu LoRA adapter định kỳ.
2. Chọn checkpoint tốt nhất theo eval.
3. Xuất `merged` làm source-of-truth.
4. Từ merged, tạo nhánh deploy:
   - GGUF cho local/edge,
   - TorchAO cho quant runtime tương ứng,
   - ONNX (nếu stack backend cần).
5. Chạy benchmark chất lượng + latency + memory cho từng định dạng.

## F2) Checklist bắt buộc
- Chat template đồng nhất.
- Tokenizer files đầy đủ.
- EOS/BOS token config đúng.
- Prompt parity test (cùng seed, cùng temperature) để phát hiện lệch hành vi.

---

## G. Quy tắc chọn định dạng nhanh

- Cần tương thích cao với HF stack: dùng `merged`.
- Cần chạy trên llama.cpp/Ollama: dùng `GGUF`.
- Cần luồng QAT/torchao: dùng `save_pretrained_torchao`.
- Cần backend ONNX: convert từ merged bằng tool ngoài + test parity kỹ.

