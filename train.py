import os
import tensorflow as tf
from tensorflow.keras import layers, models
# B0からB3にアップグレード
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler



# --- 1. 設定 ---
img_size = 300  # EfficientNetB3の推奨サイズは300x300です
batch_size = 20  # B3はメモリを消費するため少し下げます
model_save_path = '/kaggle/working/best_inu_neko_model_v2.keras'

# ディレクトリ確認（既存のパスを使用）
base_dir = '/kaggle/input/dogsvscat/PetImages' 

# --- 2. データセット構築（フィルタリング付き） ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir, validation_split=0.2, subset="training", seed=123,
    image_size=(img_size, img_size), batch_size=batch_size, label_mode='binary'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir, validation_split=0.2, subset="validation", seed=123,
    image_size=(img_size, img_size), batch_size=batch_size, label_mode='binary'
)

# 空ファイル等のエラーを無視
train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
val_ds = val_ds.apply(tf.data.experimental.ignore_errors())

# --- 3. 強力なデータ拡張レイヤー ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1), # 追加：色の濃淡の変化に強くする
    layers.RandomBrightness(0.1), # 追加：明るさの変化に強くする
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 4. モデル構築 (EfficientNetB3) ---
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

inputs = layers.Input(shape=(img_size, img_size, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x) # 追加：学習を安定させる
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x) # 過学習防止
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

# --- 5. コールバック ---
# 学習率を徐々に下げるスケジューラ
import math

def scheduler(epoch, lr):
    if epoch < 5:
        return float(lr) # 明示的にfloatに変換
    else:
        # tf.math.exp ではなく math.exp を使うことで純粋な数値として返します
        return float(lr * math.exp(-0.1))

callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ModelCheckpoint(filepath=model_save_path, save_best_only=True, monitor='val_accuracy'),
    LearningRateScheduler(scheduler)
]
# --- 6. Phase 1: 転移学習 ---
print("\nPhase 1: EfficientNetB3 転移学習")
base_model.trainable = False
# label_smoothingを導入して過学習を抑制
model.compile(optimizer=Adam(1e-3), 
              loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1), 
              metrics=['accuracy'])
model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=callbacks)

# --- 7. Phase 2: ファインチューニング（全層） ---
print("\nPhase 2: 全層ファインチューニング")
base_model.trainable = True
# BN層は凍結したままにする（定石）
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=Adam(1e-5), 
              loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1), 
              metrics=['accuracy'])
model.fit(train_ds, epochs=15, validation_data=val_ds, callbacks=callbacks)
