# 06 - Guide mở rộng & đóng góp

## Khi nào sửa ở đâu?
- Sửa nội dung chuẩn: `original_template/`.
- Không sửa tay hàng loạt trong `nb/` nếu có thể regenerate.
- Dùng script pipeline để giữ đồng bộ.

## Quy trình khuyến nghị
1. Chọn một notebook template đại diện.
2. Cập nhật nội dung + verify chạy được.
3. Regenerate notebooks/scripts bằng pipeline.
4. Soát diff trên `python_scripts/` để review logic dễ hơn.
5. Cập nhật bảng link README nếu có notebook mới.

## Checklist chất lượng
- Cài đặt tương thích Colab/Kaggle/local.
- Hyperparameters có chú thích rõ (memory vs quality tradeoff).
- Reward function không mơ hồ, có test tối thiểu cho parser/regex.
- Save/export path nhất quán (LoRA, merged, GGUF nếu có).

## Những cải tiến nên ưu tiên cho người mới
- Tạo “starter subset” 10 notebook cốt lõi.
- Chuẩn hoá naming cho notebook RL để dễ tìm.
- Thêm benchmark mini chuẩn cho từng loại task.
- Thêm docs “debug playbook” cho lỗi OOM/version mismatch.
