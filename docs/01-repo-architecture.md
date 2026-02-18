# 01 - Kiến trúc repo gốc

## Cấu trúc thư mục
- `README.md`: bảng link notebook theo category.
- `original_template/`: notebook nguồn “chuẩn biên tập”.
- `nb/`: notebook publish (Colab/Kaggle/HF Course).
- `python_scripts/`: bản `.py` export từ notebook (review diff dễ hơn).
- `update_all_notebooks.py`: script orchestration cập nhật toàn bộ notebook + README.
- `scripts/fix_templates.py`: script vá nhanh cho template theo issue cụ thể.

## Luồng tạo nội dung
1. Chỉnh template trong `original_template/`.
2. Chạy pipeline cập nhật (`update_all_notebooks.py`).
3. Script cập nhật announcement, installation cell, metadata, link/badge, chuẩn hoá text.
4. Đồng bộ sang `nb/` và convert sang `python_scripts/`.
5. Regenerate section danh mục trong `README.md`.

## Vai trò của `update_all_notebooks.py`
Script này làm nhiều việc cùng lúc:
- Chuẩn hoá nội dung markdown/code cells.
- Cập nhật package pin (`transformers`, `trl`,...) theo policy mới.
- Tự suy luận metadata notebook từ tên file (model/type/architecture).
- Cảnh báo package bị mất trong install cell.
- Convert notebook -> script và ẩn một số cell debug/memory.

## Ý nghĩa thiết kế
- Repo dùng mô hình “generated artifacts”: template là source-of-truth, `nb/` + `python_scripts/` là outputs.
- Điều này giúp maintain hàng trăm notebook đồng bộ, giảm sửa tay dễ lỗi.
