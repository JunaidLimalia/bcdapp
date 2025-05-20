from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import os
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import numpy as np
import cv2
import base64

app = Flask(__name__, static_folder="frontend/build", static_url_path="/")
# app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Define and load model
def create_model():
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 3)
    )
    return model

model = create_model()
checkpoint = torch.load("best_model_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None

        self.target_layer.register_forward_hook(self.save_features)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_features(self, module, input, output):
        self.features = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        model_output = self.model(input_image)
        if target_class is None:
            target_class = model_output.argmax(dim=1)

        self.model.zero_grad()
        model_output[0][target_class].backward()

        gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (gradients * self.features).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, input_image.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy()
    
class AblationCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.device = next(model.parameters()).device

        self.model.eval()
        self.features = None

        # Capture output of target layer
        def forward_hook(module, input, output):
            self.features = output.detach()

        self.target_layer.register_forward_hook(forward_hook)

    def forward_to_layer(self, x):
        """Manual forward pass up to the target layer (layer4[-1])"""
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)  # self.features will be captured here
        return x

    def forward_from_layer(self, features):
        """Continue forward pass from after target layer"""
        x = self.model.avgpool(features)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x

    def generate_cam(self, input_image, target_class=None):
        self.features = None  # Clear previous features
        self.model.eval()

        with torch.no_grad():
            _ = self.forward_to_layer(input_image)

        feature_maps = self.features[0]  # [C, H, W]
        n_channels = feature_maps.shape[0]

        with torch.no_grad():
            original_output = self.forward_from_layer(self.features)
        if target_class is None:
            target_class = original_output.argmax(dim=1)

        original_score = original_output[0][target_class].item()
        importance_scores = torch.zeros(n_channels, device=self.device)

        for i in range(n_channels):
            ablated = self.features.clone()
            ablated[0, i] = 0  # Zero out one channel
            with torch.no_grad():
                ablated_output = self.forward_from_layer(ablated)
            ablated_score = ablated_output[0][target_class].item()
            importance_scores[i] = original_score - ablated_score

        cam = torch.zeros(feature_maps.shape[1:], device=self.device)
        for i in range(n_channels):
            cam += importance_scores[i] * feature_maps[i]

        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                            input_image.shape[2:],
                            mode='bilinear',
                            align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy()

def generate_base64_gradcam(image_pil, image_tensor, model, target_class):
    orig_image = np.array(image_pil)
    grad_cam = GradCAM(model, model.layer4[-1])
    cam = grad_cam.generate_cam(image_tensor, target_class)

    # Heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (orig_image.shape[1], orig_image.shape[0]))

    # Superimposed
    superimposed = cv2.addWeighted(orig_image, 0.6, heatmap, 0.4, 0)

    # Encode both to base64
    _, heatmap_buf = cv2.imencode('.png', cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    _, super_buf = cv2.imencode('.jpg', cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))

    heatmap_b64 = base64.b64encode(heatmap_buf).decode('utf-8')
    super_b64 = base64.b64encode(super_buf).decode('utf-8')

    return heatmap_b64, super_b64

def generate_base64_ablationcam(image_pil, image_tensor, model, target_class):
    orig_image = np.array(image_pil)
    ablation_cam = AblationCAM(model, model.layer4[-1])
    cam = ablation_cam.generate_cam(image_tensor, target_class)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (orig_image.shape[1], orig_image.shape[0]))

    superimposed = cv2.addWeighted(orig_image, 0.6, heatmap, 0.4, 0)

    _, heatmap_buf = cv2.imencode('.png', cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    _, super_buf = cv2.imencode('.jpg', cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))

    heatmap_b64 = base64.b64encode(heatmap_buf).decode('utf-8')
    super_b64 = base64.b64encode(super_buf).decode('utf-8')

    return heatmap_b64, super_b64

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        response = jsonify({"message": "CORS preflight"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        gradcam_b64, superimposed_b64 = generate_base64_gradcam(img, img_tensor, model, predicted.item())
        ablationcam_b64, ablation_superimposed_b64 = generate_base64_ablationcam(img, img_tensor, model, predicted.item())

        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        benign_prob = float(probabilities[0]) * 100
        malignant_prob = float(probabilities[1]) * 100
        irrelevant_prob = float(probabilities[2]) * 100
        confidence = max(benign_prob, malignant_prob, irrelevant_prob)
        predicted_class = ("Benign" if predicted.item() == 0 else "Malignant" if predicted.item() == 1 else "Irrelevant")

        text_explanation = (
            f"The model predicts this case as {predicted_class} with {confidence:.2f}% confidence. "
            f"Estimated probabilities: {benign_prob:.2f}% benign, {malignant_prob:.2f}% malignant, {irrelevant_prob:.2f}% irrelevant. "
            f"These reflect the model's analysis of tissue patterns and structures in the image."
        )

        response = jsonify({
            "prediction": predicted_class,
            "gradcam": f"data:image/png;base64,{gradcam_b64}",
            "superimposed": f"data:image/jpeg;base64,{superimposed_b64}",
            "ablationcam": f"data:image/png;base64,{ablationcam_b64}",
            "ablationSuperimposed": f"data:image/jpeg;base64,{ablation_superimposed_b64}",
            "textExplanation": text_explanation
        })
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

# Serve React app
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

@app.route("/", methods=["GET"])
def root():
    return "App is running", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)