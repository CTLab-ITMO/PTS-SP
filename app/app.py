import streamlit as st
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import inception_v3
import clip
import matplotlib.pyplot as plt
from models import CLIPModel, InceptionModel


# Define the 4 params logistic regression function
def log4pl(x, A, B, C, D):
    return ((A - D) / (1.0 + ((x / C) ** B))) + D


def inv_log4pl(y, A, B, C, D):
    return C * ((A - D) / (y - D) - 1) ** (1 / B)


# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"

fcnn_clip_model = CLIPModel().to(device)
fcnn_clip_model.load_state_dict(
    torch.load("weights/model_clip_fc.pt", map_location=torch.device(device))
)
fcnn_inception_model = InceptionModel().to(device)
fcnn_inception_model.load_state_dict(
    torch.load("weights/model_incept_v3_fc.pt", map_location=torch.device(device))
)

clip_model, _ = clip.load("ViT-B/32", device=device)

# Load InceptionV3 model
inception_model = inception_v3(pretrained=True).to(device)
inception_model.fc = torch.nn.Identity()
inception_model.eval()

# Define image transformations
preprocess_inception = transforms.Compose(
    [
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Function to get CLIP embeddings
def get_clip_embedding(image):
    image = image.to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
    return image_features


# Function to get InceptionV3 embeddings
def get_inception_embedding(image):
    image = image.to(device)
    with torch.no_grad():
        image = preprocess_inception(image).to(device)
        image_features = inception_model(image)
    return image_features


# Streamlit UI
st.title("Logistic Regression Plot with Image Embeddings")

# Input 4 images
uploaded_files = st.file_uploader(
    "Выберите 4 изображения", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if len(uploaded_files) == 4:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize images if needed
            transforms.ToTensor(),  # Convert PIL images to tensors
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    # Display uploaded images
    st.write("Uploaded Images:")
    images = []
    col1, col2 = st.columns(2)
    for i, uploaded_file in enumerate(uploaded_files):
        with col1 if i < 2 else col2:
            image = Image.open(uploaded_file)
            resized_image = image.resize((150, 150))  # Resizing to fit the grid
            st.image(resized_image, caption=uploaded_file.name)
            image = transform(image)
            images.append(image)
    images = torch.stack(images)

    # Input mAP value
    mAP_value = st.number_input("Input mAP value:", step=0.01)
    with torch.no_grad():
        fcnn_clip_model.eval()
        print(images.shape)
        _, _, C, D = (fcnn_clip_model(get_clip_embedding(images)).cpu().numpy() - 10).T
        fcnn_inception_model.eval()
        A, B, _, _ = (
            fcnn_inception_model(get_inception_embedding(images)).cpu().numpy() - 10
        ).T

    if mAP_value < D and mAP_value > A:
        # Plot logistic regression
        x = np.arange(1, 10000, 0.01)
        y = log4pl(x, A, B, C, D)
        plt.plot(x, y, label="Predicted curve")
        plt.scatter(
            inv_log4pl(mAP_value, A, B, C, D),
            mAP_value,
            color="red",
            label="Intersection Point",
        )
        plt.xlabel("Training Dataset Size")
        plt.ylabel("mAP Value")
        plt.title("Logistic Regression Fit")
        print(A[0])
        plt.text(
            0.95,
            0.05,
            f"A={A[0]:.4f}, B={B[0]:.4f},\nC={C[0]:.4f}, D={D[0]:.4f},\nTrain size={int(inv_log4pl(mAP_value, A, B, C, D)[0])}",
            fontsize=10,
            verticalalignment="bottom",
            bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5},
        )
        plt.xscale("log")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)
    else:
        print("Check")
