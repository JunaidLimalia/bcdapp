from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
import cv2  # <- move import here

app = Flask(__name__)
CORS(app)  # Allow React frontend access

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure static/ directory exists
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Recreate model
def create_model():
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 2)
    )
    return model

# Load model
model = create_model()
checkpoint = torch.load("model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Grad-CAM hook storage
gradients = None
activations = None

def save_gradcam(image_tensor, model, target_layer, class_idx):
    global gradients, activations

    def forward_hook(module, input, output):
        global activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        global gradients
        gradients = grad_output[0]

    # Register hooks
    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    # Forward & backward pass
    model.zero_grad()
    output = model(image_tensor)
    class_score = output[0, class_idx]
    class_score.backward()

    # Grad-CAM heatmap
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activation = activations[0]

    for i in range(activation.shape[0]):
        activation[i, :, :] *= pooled_gradients[i]

    heatmap = activation.mean(dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Convert original tensor to PIL image
    original_image = image_tensor.squeeze().cpu()
    original_image = to_pil_image(original_image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) +
                                  torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))

    # Save Grad-CAM heatmap as PNG
    gradcam_path = os.path.join(STATIC_DIR, "gradcam.png")
    plt.imshow(heatmap, cmap='jet')
    plt.axis('off')
    plt.savefig(gradcam_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Save superimposed Grad-CAM on original image
    original_cv = np.array(original_image)
    heatmap_cv = cv2.resize(heatmap, (original_cv.shape[1], original_cv.shape[0]))
    heatmap_cv = np.uint8(255 * heatmap_cv)
    heatmap_color = cv2.applyColorMap(heatmap_cv, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * 0.4 + original_cv
    superimposed_path = os.path.join(STATIC_DIR, "superimposed.jpg")
    cv2.imwrite(superimposed_path, superimposed_img)

    # Clean up hooks
    handle_fw.remove()
    handle_bw.remove()

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    # Run Grad-CAM
    save_gradcam(img_tensor, model, model.layer4[-1], predicted.item())

    class_names = ["Benign", "Malignant"]
    prediction = class_names[predicted.item()]
    return jsonify({
        "prediction": prediction,
        "gradcam": "/static/gradcam.png",
        "superimposed": "/static/superimposed.jpg"
    })

if __name__ == "__main__":
    app.run(debug=True)