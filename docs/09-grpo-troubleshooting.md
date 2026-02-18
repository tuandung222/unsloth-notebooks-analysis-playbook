# 09 - Troubleshooting GRPO (quick reference)

## 1) Train không bắt đầu / lỗi import
### Triệu chứng
- Lỗi `ImportError`, `ModuleNotFoundError`, hoặc trùng version `trl/transformers`.

### Cách xử lý
- Dùng đúng block install trong notebook gốc.
- Không trộn pip cũ và mới trong cùng runtime.
- Restart runtime sau khi thay major versions.

## 2) Reward sai do parsing
### Triệu chứng
- Output trông đúng nhưng reward luôn thấp.

### Cách xử lý
- In raw completion đầu tiên mỗi vài steps.
- Test regex độc lập với 5-10 sample output.
- Cho phép optional whitespace/EOS trong regex nếu cần.

## 3) OOM khi GRPO
### Triệu chứng
- `CUDA out of memory` khi trainer gọi generation.

### Cách xử lý theo thứ tự ưu tiên
1. Giảm `num_generations`.
2. Giảm `max_seq_length` hoặc `max_completion_length`.
3. Giảm `per_device_train_batch_size`.
4. Dùng 4bit hoặc FP8 variant notebook.
5. Dùng GPU lớn hơn (A100/H100) nếu train model lớn.

## 4) Reward hacking
### Triệu chứng
- Reward tăng nhưng chất lượng thực tế không tăng.

### Cách xử lý
- Thêm reward chống gian lận (anti-cheating checks).
- Tách format reward và correctness reward.
- Thêm negative examples hoặc penalties khi output “lách luật”.

## 5) Environment RL bị timeout
### Triệu chứng
- Nhiều mẫu bị timeout khi execute strategy.

### Cách xử lý
- Giảm thời gian cho mỗi episode.
- Phạt nặng non-terminating strategy.
- Buộc output function ngắn, rõ contract.

## 6) Model sau RL kém hơn baseline
### Triệu chứng
- Response degrade, mất khả năng trả lời chung.

### Cách xử lý
- Giảm learning rate.
- Giảm số steps và reward extremes.
- Thử tăng dữ liệu prompt đa dạng hơn.
- Chỉ fine-tune LoRA rank nhỏ trước, tăng dần.

## 7) Quy tắc debug 80/20
- 80% lỗi GRPO nằm ở: output contract + reward parser + memory budget.
- Chỉ debug 3 thứ này trước khi đụng tới thuật toán phức tạp.
