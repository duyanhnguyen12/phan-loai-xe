# BÁO CÁO ĐỒ ÁN CUỐI KỲ

## Đề tài
**Phân tích và xây dựng mô hình học sâu phân loại xe ô tô**
(xe con 5 chỗ, xe tải, xe bus) — kết hợp tìm hiểu mô hình **R-CNN**

---

## MỤC LỤC

1. Mở đầu
2. Cơ sở lý thuyết
3. Dữ liệu và tiền xử lý
4. Phương pháp thực hiện
5. Kết quả thực nghiệm
6. Tìm hiểu chuyên sâu: R-CNN
7. Đánh giá và so sánh
8. Kết luận và hướng phát triển
9. Tài liệu tham khảo
10. Phụ lục

---

## 1. MỞ ĐẦU

### 1.1. Lý do chọn đề tài
- Nhu cầu thực tế: giám sát giao thông, phân luồng xe, thu phí tự động, đếm xe...
- Bài toán phân loại/nhận diện xe là ứng dụng tiêu biểu của Computer Vision.

### 1.2. Mục tiêu đề tài
- Xây dựng mô hình học sâu phân loại 3 loại xe: **Car (xe con 5 chỗ)**, **Bus**, **Truck**.
- Áp dụng kiến trúc **R-CNN / Faster R-CNN** để vừa phát hiện (detect) vừa phân loại xe trong ảnh.
- So sánh với mô hình classification thuần (CNN backbone: EfficientNetV2/MobileNetV2).

### 1.3. Phạm vi thực hiện
- Ảnh tĩnh (không xử lý video/real-time).
- 3 lớp đối tượng: Bus, Car, Truck.

---

## 2. CƠ SỞ LÝ THUYẾT

### 2.1. Mạng nơ-ron tích chập (CNN)
- Khái niệm convolution, pooling, feature map.
- Vai trò của CNN trong trích xuất đặc trưng ảnh.

### 2.2. Transfer Learning & Fine-tuning
- Sử dụng model pretrained trên ImageNet (MobileNetV2 / EfficientNetV2).
- Kỹ thuật fine-tuning: đóng băng (freeze) một phần layer, mở (unfreeze) các layer cuối.

### 2.3. Bài toán Object Detection vs Classification
| | Classification | Detection |
|---|---|---|
| Input | Ảnh 1 đối tượng | Ảnh nhiều đối tượng |
| Output | 1 nhãn | Nhiều bounding box + nhãn |
| Ví dụ model | MobileNetV2, EfficientNet | YOLO, R-CNN, SSD |

### 2.4. Họ mô hình R-CNN (trọng tâm Topic 18)
- **R-CNN (2014):** Region Proposal (Selective Search) + CNN trích đặc trưng từng vùng + SVM phân loại. Nhược điểm: chậm, xử lý từng vùng riêng lẻ.
- **Fast R-CNN (2015):** Đưa toàn ảnh qua CNN 1 lần, dùng ROI Pooling để trích đặc trưng từng vùng đề xuất → nhanh hơn nhiều.
- **Faster R-CNN (2015):** Thay Selective Search bằng **Region Proposal Network (RPN)** — mạng nơ-ron tự học cách đề xuất vùng, giúp toàn bộ pipeline train end-to-end.
- Kiến trúc Faster R-CNN sử dụng trong đồ án:
  ```
  Ảnh đầu vào
      ↓
  Backbone (MobileNetV3 + FPN)  → trích xuất feature map
      ↓
  RPN (Region Proposal Network) → đề xuất vùng có khả năng chứa vật thể
      ↓
  ROI Pooling/Align            → chuẩn hóa kích thước từng vùng
      ↓
  Classifier + Box Regressor    → phân loại (Bus/Car/Truck) + tinh chỉnh bounding box
  ```

---

## 3. DỮ LIỆU VÀ TIỀN XỬ LÝ

### 3.1. Nguồn dữ liệu
- Tự thu thập + bổ sung từ Kaggle/Roboflow.
- Thống kê số lượng ảnh mỗi lớp (Bus/Car/Truck) — train/val.

### 3.2. Vấn đề mất cân bằng dữ liệu (Class Imbalance)
- Bus có ít ảnh hơn Car/Truck → áp dụng **class weights** khi train.

### 3.3. Gán nhãn Bounding Box (Annotation)
- Vì dataset gốc chỉ có nhãn phân loại (không có tọa độ), đồ án dùng **YOLOv8 pretrained** để tự động sinh bounding box (`auto_annotate.py`), lưu theo chuẩn **Pascal VOC XML**.
- Quy trình: ảnh → YOLO detect → lấy box có confidence cao nhất → gán nhãn theo thư mục gốc (đảm bảo đúng nhãn, không phụ thuộc YOLO).

### 3.4. Data Augmentation
- RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast, RandomTranslation.
- Mục đích: chống overfitting khi dữ liệu còn hạn chế.

---

## 4. PHƯƠNG PHÁP THỰC HIỆN

### 4.1. Mô hình 1 — Classification thuần (baseline)
- Backbone: MobileNetV2 → nâng cấp EfficientNetV2S.
- Fine-tune các layer cuối, Dropout + L2 regularization chống overfit.
- File liên quan: `train.py`, `predict.py`, `evaluate.py`.

### 4.2. Mô hình 2 — Faster R-CNN (trọng tâm đồ án)
- Backbone: MobileNetV3-Large + FPN (chọn thay ResNet50 để giảm thời gian train trên CPU).
- Custom `FastRCNNPredictor` cho 4 lớp (background + Bus + Car + Truck).
- Optimizer: SGD, learning rate scheduler StepLR.
- File liên quan: `auto_annotate.py`, `train_rcnn.py`, `predict_rcnn.py`.

### 4.3. Công cụ hỗ trợ
- Giao diện desktop (Tkinter) cho phép chọn ảnh, hiển thị bounding box và % confidence trực quan (`app_rcnn_gui.py`).

---

## 5. KẾT QUẢ THỰC NGHIỆM

### 5.1. Kết quả huấn luyện
- Biểu đồ Train/Val Loss theo epoch *(chèn `ket_qua_training_rcnn.png`)*.
- Nhận xét xu hướng hội tụ, có/không overfitting.

### 5.2. Kết quả dự đoán trên ảnh test
| Ảnh | Nhãn thực | Dự đoán | Confidence |
|-----|-----------|---------|-----------|
| test_car.jpg | Car | Car | 85.9% |
| test_bus.jpg | Bus | Bus | 97.7% |
| test_truck.jpg | Truck | Truck | 94.4% |

### 5.3. Ảnh minh họa
- Chèn ảnh output có bounding box từ thư mục `output_rcnn/`.

---

## 6. TÌM HIỂU CHUYÊN SÂU: R-CNN (Topic 18)

### 6.1. Vì sao chọn Faster R-CNN thay vì R-CNN gốc
- R-CNN gốc quá chậm (Selective Search + CNN riêng từng vùng ~2000 vùng/ảnh).
- Faster R-CNN phù hợp triển khai thực tế, vẫn giữ được độ chính xác cao.

### 6.2. Vai trò của Region Proposal Network (RPN)
- Thay thế thuật toán truyền thống bằng mạng học được (learnable), train chung với toàn bộ mô hình.

### 6.3. Ưu điểm / Hạn chế khi áp dụng vào bài toán xe
- Ưu điểm: phát hiện được nhiều xe trong 1 ảnh, có tọa độ chính xác.
- Hạn chế: cần bounding box để train (đồ án khắc phục bằng auto-annotate); tốc độ train/inference chậm hơn classification thuần trên CPU.

---

## 7. ĐÁNH GIÁ VÀ SO SÁNH

| Tiêu chí | Classification (EfficientNetV2) | Faster R-CNN |
|----------|----------------------------------|---------------|
| Bài toán phù hợp | Ảnh 1 xe | Ảnh nhiều xe |
| Độ phức tạp cài đặt | Đơn giản | Phức tạp hơn |
| Cần bounding box | Không | Có (tự động sinh) |
| Tốc độ train (CPU) | Nhanh | Chậm hơn |
| Khả năng mở rộng | Hạn chế | Cao (đếm xe, giám sát) |

---

## 8. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 8.1. Kết luận
- Đã xây dựng thành công 2 mô hình: classification và Faster R-CNN.
- Faster R-CNN đáp ứng đúng yêu cầu đề tài, vừa phân loại vừa định vị xe.

### 8.2. Hướng phát triển
- Tăng dữ liệu, cân bằng lớp Bus.
- Thử nghiệm backbone mạnh hơn (ResNet50) khi có GPU phù hợp.
- Mở rộng sang video/real-time (đếm xe qua camera).

---

## 9. TÀI LIỆU THAM KHẢO
- Girshick, R. et al. "Rich feature hierarchies for accurate object detection" (R-CNN), 2014.
- Girshick, R. "Fast R-CNN", 2015.
- Ren, S. et al. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", 2015.
- Tài liệu PyTorch/TorchVision — Faster R-CNN.
- Tài liệu TensorFlow/Keras — Transfer Learning.

---

## 10. PHỤ LỤC
- Code nguồn (link GitHub repo).
- Hướng dẫn chạy chương trình (README).
- Ảnh kết quả bổ sung.
