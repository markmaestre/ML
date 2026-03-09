from ultralytics import YOLO
import os
import yaml

def validate_dataset(data_yaml_path):
    """Check if dataset YAML and folders exist and contain files."""
    
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Data YAML file not found: {data_yaml_path}")

    with open(data_yaml_path) as f:
        data_cfg = yaml.safe_load(f)

    for key in ["train", "val"]:
        path = data_cfg.get(key)

        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"'{key}' folder does not exist: {path}")

        # count files inside folder including subfolders
        file_count = sum(len(files) for _, _, files in os.walk(path))
        if file_count == 0:
            raise FileNotFoundError(f"'{key}' folder is empty: {path}")

    print("✅ Dataset paths are valid.")
    return data_cfg


def train_model(
    data_yaml="datasets/data.yaml",
    model_name="yolov8n.pt",
    save_name="waste",
    epochs=50,
    batch_size=16,
    img_size=640,
    workers=2   # safer default for Windows
):

    # Validate dataset first
    validate_dataset(data_yaml)

    print("🚀 Loading YOLO model...")
    model = YOLO(model_name)

    print("🔥 Starting training...")

    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        name=save_name,

        # performance + stability
        cache=True,
        workers=workers,
        amp=True,
        cos_lr=True,
        patience=30,

        # augmentation improvements
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,

        # saving
        save=True,
        save_period=5,
        verbose=True,

        # learning rate
        lr0=0.01,
        lrf=0.01,
        pretrained=True,
        resume=False
    )

    print(f"✅ Training completed! Model saved under runs/detect/{save_name}")


if __name__ == "__main__":
    train_model(
        epochs=50,
        batch_size=16,
        img_size=640
    )
