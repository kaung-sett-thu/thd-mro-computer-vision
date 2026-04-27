from ultralytics import YOLO
import os

# paths and datasets info
DATA_YAML = "../dataset/data.yaml"
TEST_IMAGES = "../dataset/test/images"
RESULTS_DIR = "../runs/detect/object_detection_train"

# initiate yolov8 model
model = YOLO("yolov8n.pt")

# base model

# model.train(
#     # training parameters
#     data=DATA_YAML,
#     epochs=20,
#     imgsz=640,
#     batch=16,
#     project=RESULTS_DIR,
#     name="exp_2",
#     exist_ok=True,

#     # augmentation parameters
#     degrees=10,      # rotation
#     translate=0.1,   # shift
#     scale=0.5,       # zoom
#     shear=2.0,       # shear

#     fliplr=0.5,      # horizontal flip
#     flipud=0.0,      # vertical flip
# )

# decrease the learning rate

model.train(
    # training parameters
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    batch=16,
    project=RESULTS_DIR,
    name="exp_3",
    exist_ok=True,

    lr0=0.001,            # lower learning rate
    lrf=0.01,

    # augmentation parameters
    degrees=10,      # rotation
    translate=0.1,   # shift
    scale=0.5,       # zoom
    shear=2.0,       # shear

    fliplr=0.5,      # horizontal flip
    flipud=0.0,      # vertical flip
)

# validate the model 
best_model_path = os.path.join(RESULTS_DIR, "exp_3/weights/best.pt")
metrics = model.val(model=best_model_path, data=DATA_YAML)
print("Validation metrics:", metrics)

# test the model using test images
TEST_IMAGES = "../dataset/test/images"  # folder with test images
output_dir = os.path.join(RESULTS_DIR, "exp_3/predictions")

results = model.predict(
    source=TEST_IMAGES,
    conf=0.25,       # confidence threshold
    save=True,       # save annotated images
    save_txt=True,   # save predictions in txt files
    project=output_dir,
    name="predictions",
    exist_ok=True
)

print("Inference done! Check predictions in:", output_dir)