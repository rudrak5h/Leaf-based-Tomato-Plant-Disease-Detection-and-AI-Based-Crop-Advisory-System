import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
from advisory import ADVISORY_DATA

# 1. Setup Model Architecture (Must match your training code)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the classes exactly as found in your training data
class_names = [
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", 
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", 
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot", 
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus", 
    "Tomato_healthy"
]

def load_model():
    model = models.resnet50(weights=None) # Load structure
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load('tomato_model2.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# 2. Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Prediction Function
def predict_and_advise(img):
    img = Image.fromarray(img)
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, index = torch.max(probabilities, 0)
    
    label = class_names[index]
    conf_score = f"{confidence.item() * 100:.2f}%"
    
    # Get advisory data from your advisory.py
    info = ADVISORY_DATA.get(label, {})
    
    advisory_text = (
        f"### 📋 AI Crop Advisory\n"
        f"**Description:** {info.get('description', 'N/A')}\n\n"
        f"**Causes:** {info.get('causes', 'N/A')}\n\n"
        f"**Prevention:** {info.get('prevention', 'N/A')}\n\n"
        f"**Treatment:** {info.get('treatment', 'N/A')}"
    )
    
    return label, conf_score, advisory_text

# 4. Gradio Interface Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🍅 Tomato Leaf Disease Detection & AI Advisory System")
    gr.Markdown("Upload an image of a tomato leaf to get an instant diagnosis and treatment plan.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image()
            btn = gr.Button("Analyze Leaf")
        
        with gr.Column():
            out_label = gr.Textbox(label="Detected Disease")
            out_conf = gr.Textbox(label="Confidence Level")
            out_advisory = gr.Markdown(label="Advisory Details")
    
    btn.click(predict_and_advise, inputs=input_img, outputs=[out_label, out_conf, out_advisory])

demo.launch(share=True)