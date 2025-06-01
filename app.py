import gradio as gr
from transformers import pipeline, CLIPProcessor, CLIPModel
from datasets import load_dataset
from PIL import Image
import torch

# 1. Labels laden aus dem originalen Food-101-Dataset
food101_dataset = load_dataset("food101", split="train[:1%]")
labels_food101 = food101_dataset.features["label"].names

# 2. Eigener ViT-Classifier (hier musst du den Modelnamen anpassen, wenn du es hochgeladen hast)
vit_classifier = pipeline("image-classification", model="jarinschnierl/vit-base-food101")

# 3. Zero-Shot CLIP vorbereiten
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# 4. Klassifizierungsfunktion
def classify_food(image):
    # ViT Klassifikation
    vit_results = vit_classifier(image)
    vit_output = {result['label']: result['score'] for result in vit_results}

    # CLIP Zero-Shot Klassifikation
    inputs = clip_processor(text=[f"a photo of {label}" for label in labels_food101], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1)[0]
    
    clip_output = {label: float(probs[i]) for i, label in enumerate(labels_food101)}
    clip_output = dict(sorted(clip_output.items(), key=lambda x: x[1], reverse=True)[:5])  # Top 5 anzeigen

    return {"ViT Classification": vit_output, "CLIP Zero-Shot Classification": clip_output}

# 5. Gradio UI
iface = gr.Interface(
    fn=classify_food,
    inputs=gr.Image(type="filepath"),
    outputs=gr.JSON(),
    title="Food Classification Comparison",
    description="Vergleich zwischen trainiertem ViT-Modell und CLIP Zero-Shot auf Food-101 Kategorien.",
)

iface.launch()
