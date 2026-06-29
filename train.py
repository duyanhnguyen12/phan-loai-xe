import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Load data - Bật shuffle=True cho cả hai tập để đồng bộ nhãn hoàn hảo
train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=42
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/val",
    image_size=(224, 224),
    batch_size=32,
    shuffle=True, # Sửa thành True để tránh lệch batch nhãn
    seed=42
)

# Kiểm tra thứ tự nhãn thực tế
CLASS_NAMES = train_ds.class_names
print("\n[INFO] Thứ tự các lớp thực tế Keras nhận diện:", CLASS_NAMES)

# Tối ưu tốc độ load (Bỏ .cache() trống để ép quét lại dữ liệu sạch ngoài ổ đĩa)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Data Augmentation chống học vẹt
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomBrightness(0.15),
])

# 2. Xây dựng mô hình Fine-tuning (Mở băng một phần)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Bật trainable để tinh chỉnh trọng số cho phù hợp cấu trúc xe thực tế
base_model.trainable = True

# Khóa các tầng trích xuất cạnh cơ bản, chỉ mở 40 tầng cao cấp phía sau để học hình khối xe
for layer in base_model.layers[:-40]:
    layer.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=True) # training=True giúp cập nhật các tầng BatchNormalization
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x) # Tăng dropout lên một chút chống overfitting
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# HẠ THẤP LEARNING RATE (Bắt buộc khi Fine-tuning để tránh làm nát trọng số ImageNet)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Giảm xuống 0.0001
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', # Theo dõi val_loss chính xác hơn val_accuracy
    patience=5,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/best_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("\n=== BẮT ĐẦU TRAINING MỚI (FINE-TUNING) ===")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[early_stopping, checkpoint]
)

# Vẽ và lưu đồ thị kết quả mới
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("ket_qua_training.png")
print("\n✅ Đã lưu đồ thị mới vào file ket_qua_training.png")