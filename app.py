from fastapi import FastAPI, Form
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pyrealsense2 as rs
import numpy as np
import cv2
import base64

position_map = {
    1: (0, 0),
    2: (1, 0),
    3: (2, 0),
    4: (0, 1),
    5: (1, 1),
    6: (2, 1),
    7: (0, 2),
    8: (1, 2),
    9: (2, 2),
    10: (0, 3),
    11: (1, 3),
    12: (2, 3),
    13: (0, 4),
    14: (1, 4),
    15: (2, 4), 
}

VERTICAL_LINES = [520, 600, 680, 760]  # Vertical (x) positions
HORIZONTAL_LINES = [160, 240, 320, 400, 480, 560]  # Horizontal (y) positions
PREPROCESS  = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])


app = FastAPI()

model = None
pipeline = None

def load_model():
    global model
    model = models.resnet50()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load("./model/best_model_depth.pth"))
    model.eval()
    print("Classification model loaded")

def init_camera():
    global pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipeline.start(config)
    print("Camera settings initialized")
    
    
def capture_depth_image():
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    
    if not depth_frame:
        raise RuntimeError("Could not capture depth image from the camera.")
    
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    _, buffer = cv2.imencode('.png', depth_image)
    print("Image captured")
    return base64.b64encode(buffer).decode("utf-8")

def classify_image(model, image, preprocess):    
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        probability = model(image_tensor)
    return round(probability.item(), 4)

def load_image_from_path(image_path):
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to load image from {image_path}: {e}")

def split_and_classify(model, image, vertical_lines, horizontal_lines):
    classifications = []
    part_number = 1

    for i in reversed(range(len(horizontal_lines) - 1)):  # Start from bottom
        for j in reversed(range(len(vertical_lines) - 1)):  # Start from right
            left = vertical_lines[j]
            right = vertical_lines[j + 1]
            upper = horizontal_lines[i]
            lower = horizontal_lines[i + 1]

            part = image.crop((left, upper, right, lower))

            classification = classify_image(model, part, PREPROCESS)
            classifications.append(classification)

            part_number += 1

    return classifications

@app.get("/get_depth_image")
def get_depth_image():
    try:
        base64_image = capture_depth_image()
        print("Image base64 returned as response")
        return {"image": base64_image}
    except RuntimeError as e:
        return {"error": str(e)}
    
    
@app.post("/glas_config")
async def glass_config(image: str = Form(...)):
    global model
    image_data = base64.b64decode(image)
    
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    file_path = "./images/image.png"
    cv2.imwrite(file_path, img)
    
    img = load_image_from_path(file_path)
    classifications = split_and_classify(model, img, VERTICAL_LINES, HORIZONTAL_LINES)
    indices = [index for index, value in enumerate(classifications) if value > 0.5]     # give only indices if classified over 50%
    indices = [index + 1 for index in indices]                                          # Begin index with 1
    
    print("----")
    print(indices)
    print("----")
    
    return [{"x": position_map[i][0], "y": position_map[i][1]} for i in indices]
    
    
@app.on_event("shutdown")
def shutdown_event():
    global pipeline
    print("Shutting down RealSense pipeline...")
    pipeline.stop()

@app.on_event("startup")
async def startup_event():
    load_model()
    init_camera()
