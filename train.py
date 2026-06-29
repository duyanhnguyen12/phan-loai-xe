import tensorflow as tf
import matplotlib.pyplot as plt

# Load data
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
    shuffle=False
)

# Tên các lớp
CLASS_NAMES = train_ds.class_names
print("Các lớp:", CLASS_NAMES)

# Tối ưu tốc độ load
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# Data Augmentation (quan trọng vì dataset nhỏ)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.1),
])

# Xây dựng model Transfer Learning
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # Chuẩn hóa đúng cho MobileNetV2
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,           # Dừng nếu 5 epoch không cải thiện
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/best_model.keras",
    monitor='val_accuracy',
    save_best_only=True,  # Chỉ lưu model tốt nhất
    verbose=1
)

# Huấn luyện
print("\n=== BẮT ĐẦU TRAINING ===")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stopping, checkpoint]
)

# Lưu model cuối
model.save("models/xe_classifier.keras")
print("Đã lưu model!")

# Kết quả tốt nhất
best_acc = max(history.history['val_accuracy'])
print(f"\n✅ Val Accuracy tốt nhất: {best_acc*100:.1f}%")

# Vẽ đồ thị kết quả
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy theo epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss theo epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("ket_qua_training.png")
plt.show()
print("Đã lưu đồ thị ket_qua_training.png")