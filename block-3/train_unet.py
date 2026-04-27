import os
import re
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# =====================================================
# CONFIG
# =====================================================
IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 50
NUM_CLASSES = 6

DATASET = "dataset"

TRAIN_DIR = os.path.join(DATASET, "train")
VAL_DIR   = os.path.join(DATASET, "valid")
TEST_DIR  = os.path.join(DATASET, "test")


# =====================================================
# CLEAN ROBOFLOW FILENAMES
# =====================================================
def extract_key(filename):
    name = os.path.splitext(filename)[0]
    name = re.sub(r"_jpg\.rf\..*", "", name)
    name = re.sub(r"_png\.rf\..*", "", name)
    name = name.replace("_mask", "")
    return name


# =====================================================
# PAIRING
# =====================================================
def get_pairs(folder):
    files = os.listdir(folder)

    images = {}
    masks = {}

    for f in files:
        path = os.path.join(folder, f)

        if f.endswith("_mask.png"):
            key = extract_key(f.replace("_mask.png", ".png"))
            masks[key] = path

        elif f.lower().endswith((".jpg", ".jpeg", ".png")) and "_mask" not in f:
            key = extract_key(f)
            images[key] = path

    return [(images[k], masks[k]) for k in images if k in masks]


# =====================================================
# LOAD IMAGE / MASK
# =====================================================
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img


def load_mask(path):
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(
        mask,
        (IMG_SIZE, IMG_SIZE),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return tf.cast(mask, tf.int32)


def process(img_path, mask_path):
    return load_image(img_path), load_mask(mask_path)


def augment(img, mask):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    return img, mask


# =====================================================
# DATASET
# =====================================================
def create_dataset(folder, training=False):
    pairs = get_pairs(folder)

    if len(pairs) == 0:
        raise ValueError(f"No pairs found in {folder}")

    img_paths = np.array([p[0] for p in pairs], dtype=str)
    mask_paths = np.array([p[1] for p in pairs], dtype=str)

    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(200)

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = create_dataset(TRAIN_DIR, training=True)
val_ds   = create_dataset(VAL_DIR)
test_ds  = create_dataset(TEST_DIR)


# =====================================================
# DICE ONLY (SAFE METRIC)
# =====================================================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    y_true = tf.one_hot(tf.cast(y_true, tf.int32), NUM_CLASSES)
    y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), NUM_CLASSES)

    inter = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    return (2. * inter + smooth) / (union + smooth)


# =====================================================
# U-NET
# =====================================================
def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    return x


def encoder(x, f):
    c = conv_block(x, f)
    p = layers.MaxPooling2D()(c)
    return c, p


def decoder(x, skip, f):
    x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, f)
    return x


def build_unet():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))

    s1, p1 = encoder(inputs, 64)
    s2, p2 = encoder(p1, 128)
    s3, p3 = encoder(p2, 256)
    s4, p4 = encoder(p3, 512)

    b = conv_block(p4, 1024)

    d1 = decoder(b, s4, 512)
    d2 = decoder(d1, s3, 256)
    d3 = decoder(d2, s2, 128)
    d4 = decoder(d3, s1, 64)

    outputs = layers.Conv2D(NUM_CLASSES, 1, activation="softmax")(d4)

    return Model(inputs, outputs)


model = build_unet()


# =====================================================
# COMPILE (FIXED - NO MeanIoU)
# =====================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", dice_coef]
)


# =====================================================
# CALLBACKS
# =====================================================
callbacks = [
    ModelCheckpoint("best_unet.keras", save_best_only=True, monitor="val_loss"),
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=4, factor=0.5, verbose=1)
]


# =====================================================
# TRAIN
# =====================================================
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# =====================================================
# TEST
# =====================================================
print("Evaluating...")
print(model.evaluate(test_ds))

model.save("final_unet.keras")