import tensorflow as tf
import matplotlib.pyplot as plt
from keras.regularizers import l2

# 1. Load data
train_ds = tf.keras.utils.image_dataset_from_directory(
    "../data/train",
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=42
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "../data/val",
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=42
)

CLASS_NAMES = train_ds.class_names
print("\n[INFO] Thứ tự các lớp:", CLASS_NAMES)

# 2. Tính class weights để bù cho Bus bị ít ảnh hơn
counts = [len(tf.io.gfile.listdir(f"../data/train/{c}")) for c in CLASS_NAMES]
total = sum(counts)
class_weight = {i: total / (len(counts) * cnt) for i, cnt in enumerate(counts)}
print(f"[INFO] Số ảnh mỗi class: { {CLASS_NAMES[i]: counts[i] for i in range(len(CLASS_NAMES))} }")
print(f"[INFO] Class weights: { {CLASS_NAMES[i]: round(class_weight[i], 3) for i in range(len(CLASS_NAMES))} }")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# 3. Data Augmentation mạnh hơn
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomBrightness(0.15),
    tf.keras.layers.RandomContrast(0.15),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])

# 4. Base model: EfficientNetV2S (chính xác hơn MobileNetV2, vẫn nhẹ)
base_model = tf.keras.applications.EfficientNetV2S(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Khóa toàn bộ, chỉ mở 50 tầng cuối để fine-tune
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
x = base_model(x, training=True)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=l2(0.001))(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5. Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/best_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

print("\n=== BẮT ĐẦU TRAINING (EfficientNetV2S + Class Weights + Augmentation mạnh) ===")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    class_weight=class_weight,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# 6. Vẽ đồ thị
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
print("\n✅ Đã lưu đồ thị vào ket_qua_training.png")
