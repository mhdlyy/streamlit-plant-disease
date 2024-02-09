import streamlit as st
from PIL import Image
from torchvision import models, transforms
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2

def get_labels():
    return [
        'Apple scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper leaf',
        'Bell_pepper leaf spot', 'Blueberry leaf', 'Cherry leaf', 'Corn gray leaf spot',
        'Corn leaf blight', 'Corn rust leaf', 'Peach leaf', 'Potato leaf early blight',
        'Potato leaf late blight', 'Raspberry leaf', 'Soyabean leaf',
        'Squash powdery mildew leaf', 'Strawberry leaf', 'Tomato early blight leaf',
        'Tomato septoria leaf spot', 'Tomato leaf', 'Tomato leaf bacterial spot',
        'Tomato leaf late blight', 'Tomato leaf mosaic virus',
        'Tomato leaf yellow virus', 'Tomato mold leaf', 'Grape leaf', 'Grape leaf black rot'
    ]

def get_transforms(image_size):
    inference_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    return inference_transform

def denormalize(
    x,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
):
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(x, 0, 1)

def getPlantDiseaseName(output_class):
    class_name = get_labels()[int(output_class)]
    plant_name = class_name.split(' ')[0]
    disease_name = ' '.join(class_name.split(' ')[1:])

    return [plant_name, disease_name]

def inference(model, image):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.eval()
    
    with torch.no_grad():
        image = image.to(DEVICE)

        # Forward pass.
        outputs = model(image)

    # Softmax probabilities.
    predictions = F.softmax(outputs, dim=1).cpu().numpy()

    # Predicted class number.
    output_class = np.argmax(predictions)
    
    return [getPlantDiseaseName(output_class), np.max(predictions)]
    
@st.cache_resource
def get_model():
    model = models.resnet50()

    for params in model.parameters():
        params.requires_grad = False

    num_in_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_in_ftrs, 27)

    weights = torch.load('best_model.pth', map_location=torch.device('cpu')) # paling di sini salah jg... krn format isi .pth beda dg yg dulu 
    
    if weights is not None:
        model.load_state_dict(weights['model_state_dict'])
    
    return model

def predict(image_path):
    # file yg diupload hrs disave dulu biar tdk error. Dunno why
    img = Image.open(image_path)
    # img = img.save("img.jpg")
    # image = cv2.imread("img.jpg")
    image=np.asarray(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transforms = get_transforms(224)
    image = transforms(image)
    image = torch.unsqueeze(image, 0)

    return inference(get_model(), image)