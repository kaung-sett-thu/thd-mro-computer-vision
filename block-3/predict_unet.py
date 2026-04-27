import os
import re
import cv2
import numpy as np
import tensorflow as tf

# =====================================================
# CONFIG
# =====================================================
IMG_SIZE = 512
MODEL_PATH = "best_unet.keras"

INPUT_FOLDER = "predict_images"
OUTPUT_FOLDER = "predictions"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# =====================================================
# CLASS COLORS (RGB)
# =====================================================
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (200, 200, 200),
    2: (255, 0, 0),
    3: (0, 0, 255),
    4: (0, 255, 0),
    5: (40, 40, 40)
}

# =====================================================
# PREPROCESS
# =====================================================
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# =====================================================
# DECODE MASK
# =====================================================
def decode_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for k, v in CLASS_COLORS.items():
        color_mask[mask == k] = v

    return color_mask


# =====================================================
# CLEAN FILENAME (Roboflow-safe)
# =====================================================
def clean_name(name):
    name = re.sub(r"_jpg\.rf\..*?(\.jpg|\.png)$", r"\1", name)
    name = re.sub(r"_png\.rf\..*?(\.png)$", r"\1", name)
    return name


# =====================================================
# INFERENCE LOOP
# =====================================================
for file in os.listdir(INPUT_FOLDER):

    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(INPUT_FOLDER, file)

    img = cv2.imread(path)
    if img is None:
        print(f"Skipping corrupted file: {file}")
        continue

    h, w = img.shape[:2]

    # preprocess
    input_img = preprocess(img)

    # predict
    pred = model.predict(input_img, verbose=0)
    pred_mask = np.argmax(pred[0], axis=-1).astype(np.uint8)

    # decode
    color_mask = decode_mask(pred_mask)

    # resize back
    color_mask = cv2.resize(color_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # overlay
    overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

    # save
    name = os.path.splitext(file)[0]
    name = clean_name(name)

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{name}_mask.png"), color_mask)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{name}_overlay.png"), overlay)

print("✅ Predictions saved to predictions/")