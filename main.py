import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import gdown
import torchvision
import torchvision.models as models
import torch.nn as nn
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
mask_model_url = 'https://drive.google.com/file/d/1lzy-3ex6CxWrAs1J_g4DF5VS3Q0iP-4x/view?usp=sharing'
classification_model_url = "https://drive.google.com/file/d/1XF47KFp7Clq98wfppCl33mxKF2h96kN0/view?usp=sharing"
def download_file_from_gdrive(url, output_path):
    file_id = url.split('/d/')[1].split('/')[0]
    gdrive_url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(gdrive_url, output_path, quiet=False, fuzzy=True)

mask_model_path = 'mask_model.pth'
classification_model_path = "best.pth"
download_file_from_gdrive(mask_model_url, mask_model_path)
download_file_from_gdrive(classification_model_url, classification_model_path)

classification_model =models.resnet50()
classification_model.fc = nn.Sequential(nn.Dropout(0.5),nn.Linear(classification_model.fc.in_features,1024),nn.ReLU(),nn.Dropout(0.5),nn.Linear(1024, 4))

state_dict = torch.load(classification_model_path, map_location=device,weights_only=True)
classification_model.load_state_dict(state_dict)
classification_model.to(device)

mask_model = torch.load(mask_model_path, map_location=device,weights_only=False)
mask_model.eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])

def mask_images(file_path):
    image = Image.open(file_path).convert("RGB")
    tensor_image = torchvision.transforms.ToTensor()(image).to(device)

    with torch.no_grad():
        outputs = mask_model(tensor_image.unsqueeze(0))
    masks = outputs[0]['masks'].cpu().detach().numpy()
    scores = outputs[0]['scores'].cpu().detach().numpy()

    max_score_idx = np.argmax(scores)
    best_mask = masks[max_score_idx].squeeze()

    mask_pil = Image.fromarray((best_mask * 255).astype(np.uint8))
    mask_pil = mask_pil.resize(image.size, resample=Image.BILINEAR)
    mask_pil = mask_pil.convert("L")
    mask_np = np.array(mask_pil)
    rows = np.any(mask_np > 128, axis=1)
    cols = np.any(mask_np > 128, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    cropped_mask = mask_pil.crop((x_min, y_min, x_max, y_max))
    cropped_image_np = np.array(cropped_image)
    cropped_mask_np = np.array(cropped_mask)
    masked_image_np = np.where(cropped_mask_np[..., None] > 128, cropped_image_np, 0)
    masked_image = Image.fromarray(masked_image_np)
    return masked_image

def grade_classification(masked_image):
    masked_image = transform(masked_image.convert("RGB")).unsqueeze(0).to(device)
    classification_model.eval()
    
    with torch.no_grad():
        outputs = classification_model(masked_image)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    prob_list = probs.cpu().numpy().tolist()[0] 
    prob_list = [round(p, 2) for p in prob_list]
    return "result.jpg", prob_list

def process_folder(input_folder):
    images = []
    predictions = []
    for idx, filename in enumerate(os.listdir(input_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg','.tif')):
            file_path = os.path.join(input_folder, filename)
            print(f"Processing image {idx}: {filename}")
    
            masked_image = mask_images(file_path)
            _, prob_list = grade_classification(masked_image)
            images.append(masked_image)
            predictions.append(prob_list)
    output_image = create_prediction_grid(images, predictions)
    output_image_path = "prediction_grid.jpg"
    output_image.save(output_image_path)
    print(f"Grid of predictions saved at: {output_image_path}")



def create_prediction_grid(images, predictions, grid_size=(4, 4)):
    image_label = ["Naive", "Mild", "Moderate", "Severe"]
    img_width, img_height = 224, 224 
    grid_width, grid_height = grid_size
    grid_image = Image.new('RGB', (img_width * grid_width, img_height * grid_height + 40), 'white')
    
    for i, (image, prediction) in enumerate(zip(images, predictions)):
        row = i // grid_width
        col = i % grid_width
        x_offset = col * img_width
        y_offset = row * (img_height + 40) 
        resized_image = image.resize((img_width, img_height))
        grid_image.paste(resized_image, (x_offset, y_offset))
        max_prob_idx = prediction.index(max(prediction))
        predicted_class = image_label[max_prob_idx]
        predicted_prob = round(max(prediction), 2)
        draw = ImageDraw.Draw(grid_image)
        prediction_text = f"Class: {predicted_class}"
        draw.text((x_offset, y_offset + img_height + 5), prediction_text, fill="black")
    
    return grid_image

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify the folder containing images.")
    else:
        input_folder = sys.argv[1]
        process_folder(input_folder)