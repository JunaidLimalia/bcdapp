from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

app = Flask(__name__, static_folder="frontend/build", static_url_path="/")
# app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Ensure static directory exists
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

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
        nn.Linear(512, 2)
    )
    return model

model = create_model()
checkpoint = torch.load("model.pth", map_location=device)
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

def visualize_gradcam(image_pil, image_tensor, model, target_class):
    orig_image = np.array(image_pil)

    grad_cam = GradCAM(model, model.layer4[-1])
    cam = grad_cam.generate_cam(image_tensor, target_class)

    # Save Grad-CAM heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (orig_image.shape[1], orig_image.shape[0]))
    gradcam_path = os.path.join(STATIC_DIR, "gradcam.png")
    cv2.imwrite(gradcam_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

    # Save superimposed image
    superimposed = cv2.addWeighted(orig_image, 0.6, heatmap, 0.4, 0)
    superimposed_path = os.path.join(STATIC_DIR, "superimposed.jpg")
    cv2.imwrite(superimposed_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        visualize_gradcam(img, img_tensor, model, predicted.item())

        # Probability calculation
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        benign_prob = float(probabilities[0]) * 100
        malignant_prob = float(probabilities[1]) * 100
        confidence = max(benign_prob, malignant_prob)
        predicted_class = "Benign" if predicted.item() == 0 else "Malignant"

        # Summary explanation
        text_explanation = (
            f"The model predicts this case as {predicted_class} with a confidence of {confidence:.2f}%. "
            f"It estimates a {benign_prob:.2f}% likelihood that the sample is benign and {malignant_prob:.2f}% "
            f"for malignant. These probabilities reflect the model's level of certainty after analyzing the "
            f"tissue patterns, structure, and intensity distribution in the input image."
        )

        return jsonify({
            "prediction": predicted_class,
            "gradcam": "/static/gradcam.png",
            "superimposed": "/static/superimposed.jpg",
            "textExplanation": text_explanation
    })

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
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=8080)



# Uncomment the following lines if you want to run a simple Flask app without React
# from flask import Flask

# app = Flask(__name__)

# @app.route("/")
# def hello():
#     return "Hello from Flask!"

# if __name__ == "__main__":
#     import os
#     port = int(os.environ.get("PORT", 8080))
#     app.run(host="0.0.0.0", port=port)