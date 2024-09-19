import os
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pyrealsense2 as rs

def load_custom_model(model_path):
    model = models.resnet50()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def classify_image(model, image, preprocess):
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output)
    
    return round(probability.item(), 4)

def split_and_classify(image, vertical_lines, horizontal_lines, model, preprocess):
    classifications = []
    part_number = 1

    for i in reversed(range(len(horizontal_lines) - 1)):  # Start from bottom
        for j in reversed(range(len(vertical_lines) - 1)):  # Start from right
            left = vertical_lines[j]
            right = vertical_lines[j + 1]
            upper = horizontal_lines[i]
            lower = horizontal_lines[i + 1]

            part = image.crop((left, upper, right, lower))
            part.save(os.path.join("images", f"part_{part_number}.png"))

            classification = classify_image(model, part, preprocess)
            classifications.append(classification)

            part_number += 1

    return classifications

def load_image_from_path(image_path):
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to load image from {image_path}: {e}")

def capture_image():
    pipeline = rs.pipeline()
    config = rs.config()
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    

    pipeline.start(config)
    
    try:
        frames = pipeline.wait_for_frames()
        frame = frames.get_color_frame()
        
        if not frame:
            raise RuntimeError("Could not capture an image from the camera.")
        
        image_data = np.asanyarray(frame.get_data())
        image = Image.fromarray(image_data)
        return image
    
    finally:
        pipeline.stop()

def main():
    test_image = "./test/gt_2024-09-18_17-00-47_74.png"
    model_path = 'model/best_model_depth.pth'
    vertical_lines = [520, 600, 680, 760]  # Vertical (x) positions
    horizontal_lines = [160, 240, 320, 400, 480, 560]  # Horizontal (y) positions
    
    preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = load_custom_model(model_path)
    
    # image = capture_image()
    image = load_image_from_path(test_image)
    
    classifications = split_and_classify(image, vertical_lines, horizontal_lines, model, preprocess)
    
    print(f"Classifications: {classifications}")
    

if __name__ == "__main__":
    main()
