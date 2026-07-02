# KHUNG SLIDE THUYẾT TRÌNH
## Phân loại xe ô tô (Car / Bus / Truck) bằng Deep Learning + R-CNN

Ước tính: 12–15 slide, ~10-12 phút thuyết trình

---

### Slide 1 — Trang bìa
- Tên đề tài, tên thành viên, lớp, giảng viên hướng dẫn.

### Slide 2 — Đặt vấn đề
- Vì sao cần phân loại xe tự động? (giao thông thông minh, thu phí, giám sát)
- 2 đề tài con: (2) Phân loại xe + (18) Tìm hiểu R-CNN.

### Slide 3 — Mục tiêu đồ án
- Xây dựng model phân loại 3 loại xe: Car, Bus, Truck.
- Áp dụng kiến trúc R-CNN để detect + classify.
- So sánh với model classification thuần.

### Slide 4 — Tổng quan dữ liệu
- Nguồn dữ liệu, số lượng ảnh mỗi lớp (bảng/biểu đồ cột).
- Vấn đề mất cân bằng dữ liệu (Bus ít hơn).

### Slide 5 — Quy trình tiền xử lý dữ liệu
- Chia train/val.
- Data Augmentation (liệt kê ngắn gọn + hình minh họa trước/sau).

### Slide 6 — Vấn đề: Cần bounding box cho R-CNN
- Dataset gốc chỉ có nhãn phân loại.
- Giải pháp: dùng YOLOv8 tự động sinh bounding box (Pascal VOC XML).
- Sơ đồ: Ảnh → YOLO detect → box confidence cao nhất → gán nhãn theo thư mục.

### Slide 7 — Lý thuyết: Họ mô hình R-CNN
- R-CNN → Fast R-CNN → Faster R-CNN (bảng so sánh ngắn: tốc độ, cách đề xuất vùng).
- Nhấn mạnh lý do chọn Faster R-CNN.

### Slide 8 — Kiến trúc Faster R-CNN áp dụng
- Sơ đồ pipeline: Backbone (MobileNetV3+FPN) → RPN → ROI Pooling → Classifier + Box Regressor.
- Giải thích ngắn từng khối.

### Slide 9 — Mô hình baseline: Classification thuần
- MobileNetV2 → EfficientNetV2S, Transfer Learning, class weights.
- Nêu ngắn gọn để làm nền so sánh.

### Slide 10 — Kết quả huấn luyện
- Biểu đồ Train/Val Loss (ket_qua_training_rcnn.png).
- Nhận xét hội tụ.

### Slide 11 — Kết quả dự đoán minh họa
- Ảnh test có bounding box (Car/Bus/Truck) kèm % confidence.
- Bảng kết quả 3 ảnh mẫu.

### Slide 12 — Demo giao diện
- Screenshot giao diện Tkinter (chọn ảnh → hiển thị kết quả).
- (Nếu thuyết trình trực tiếp: demo live).

### Slide 13 — So sánh 2 phương pháp
- Bảng so sánh Classification vs Faster R-CNN (tiêu chí: độ phức tạp, tốc độ, khả năng mở rộng).

### Slide 14 — Kết luận & Hướng phát triển
- Kết quả đạt được.
- Hạn chế: tốc độ train CPU, dữ liệu còn ít.
- Hướng mở rộng: video real-time, tăng dữ liệu, GPU.

### Slide 15 — Cảm ơn / Hỏi đáp
- Lời cảm ơn, mở phần Q&A.

---

## LƯU Ý KHI LÀM SLIDE
- Mỗi slide tối đa 5-6 dòng chữ, ưu tiên hình ảnh/sơ đồ.
- Dùng đúng ảnh thực tế đã train (ket_qua_training_rcnn.png, output_rcnn/*.jpg) — không dùng ảnh minh họa giả.
- Phân công: 1 người trình bày lý thuyết R-CNN (Topic 18), 1 người trình bày phần thực nghiệm phân loại xe (Topic 2), nếu làm nhóm.
