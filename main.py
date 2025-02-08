import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
from torchvision import transforms, models
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import uuid


app = Flask(__name__)

# Class names
class_names = ['Covid-19', 'Emphysema', 'Healthy', 'Pneumonia', 'Tuberculosis', 'Random']

# Load the pre-trained ResNet101 model
model = models.resnet101(pretrained=False)
model.fc = torch.nn.Linear(in_features=2048, out_features=len(class_names))
checkpoint_path = "resnet101_state_dict.pth"  
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Generate visualizations for Grad-CAM and other images
def generate_visualizations(image_path):
    original_image = cv2.imread(image_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Grayscale image
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Histogram equalized image
    equalized_image = cv2.equalizeHist(grayscale_image)

    # Edge detection result
    edges_image = cv2.Canny(grayscale_image, 50, 150)

    # Segmented image
    _, segmented_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)

    # Grad-CAM Visualization
    input_tensor = preprocess_image(image_path)
    target_layer = model.layer4[-1]  # Last layer of ResNet101
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    # Normalize the Grad-CAM output to range [0, 1]
    grayscale_cam = np.maximum(grayscale_cam, 0)
    grayscale_cam = grayscale_cam / np.max(grayscale_cam)

    # Apply the Grad-CAM heatmap on the image with a red color map
    input_image_rgb_resized = cv2.resize(original_image_rgb, (224, 224))

    # Using a custom colormap (ensure red regions are highlighted)
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    grad_cam_image = cv2.addWeighted(input_image_rgb_resized, 0.7, heatmap, 0.3, 0)

    # Mask for ROI extraction (high-confidence areas are those with grayscale_cam > 0.5)
    roi_mask = (grayscale_cam > 0.5).astype(np.uint8)
    roi = cv2.bitwise_and(input_image_rgb_resized, input_image_rgb_resized, mask=roi_mask)

    # Generate unique file name
    file_id = str(uuid.uuid4())

    # Save images to server and return file paths
    visualization_paths = {
        "original": f'/uploads/{file_id}_original.png',
        "grayscale": f'/uploads/{file_id}_grayscale.png',
        "equalized": f'/uploads/{file_id}_equalized.png',
        "edges": f'/uploads/{file_id}_edges.png',
        "segmented": f'/uploads/{file_id}_segmented.png',
        "grad_cam": f'/uploads/{file_id}_grad_cam.png',
        "roi": f'/uploads/{file_id}_roi.png'
    }

    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)

    # Save the visualizations as images
    cv2.imwrite(f'uploads/{file_id}_original.png', original_image)
    cv2.imwrite(f'uploads/{file_id}_grayscale.png', grayscale_image)
    cv2.imwrite(f'uploads/{file_id}_equalized.png', equalized_image)
    cv2.imwrite(f'uploads/{file_id}_edges.png', edges_image)
    cv2.imwrite(f'uploads/{file_id}_segmented.png', segmented_image)
    cv2.imwrite(f'uploads/{file_id}_grad_cam.png', grad_cam_image)
    cv2.imwrite(f'uploads/{file_id}_roi.png', roi)

    return visualization_paths

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_id = str(uuid.uuid4())
    file_path = os.path.join('uploads', f'{image_id}.png')
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)

    input_tensor = preprocess_image(file_path)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
        confidence_score = torch.nn.functional.softmax(output, dim=1)[0, predicted_class].item() * 100

    result = {
        "predicted_class": class_names[predicted_class],
        "confidence_score": f"{confidence_score:.2f}%",
    }

    if class_names[predicted_class] == "Random":
        result["message"] = "Sorry... You inserted a Non X-ray image. Please try again with a chest x-ray image. Thank you."
        result["visualizations"] = None
    else:
        visualizations = generate_visualizations(file_path)
        result["visualizations"] = visualizations

    return jsonify(result)

# To serve the images from the 'uploads' folder correctly
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)
