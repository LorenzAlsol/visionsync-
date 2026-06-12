import urllib.request
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "onnx")

MODELS = {
    "yolov8n.onnx": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx",
}

def download():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, url in MODELS.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f"  [skip] {filename} already exists.")
            continue
        print(f"  [download] {filename} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  [ok] Saved to {dest}")

if __name__ == "__main__":
    print("Downloading VisionSync models...")
    download()
    print("Done.")