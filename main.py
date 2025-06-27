import streamlit as st
from PIL import Image
import torch
import timm
import numpy as np
import os

# --- Model and preprocessing setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Vision Transformer model
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)  # num_classes=0 removes final classification layer
model = model.eval().to(device)

# Correct preprocessing function
def preprocess_img(pil_img):
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
    ])
    return tf(pil_img).unsqueeze(0).to(device)

# Improved embedding extraction
def get_embedding(pil_img):
    img_tensor = preprocess_img(pil_img)
    with torch.no_grad():
        features = model.forward_features(img_tensor)
        # Use class token (first token) as image embedding
        embedding = features[:, 0].cpu().numpy()
    return embedding.squeeze()

# --- Face recognition system ---
def load_known_embeddings(known_dir="known_faces"):
    known_embeddings = []
    known_names = []
    
    if not os.path.exists(known_dir):
        os.makedirs(known_dir)
        return np.empty((0, 768)), []
    
    for fname in os.listdir(known_dir):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            try:
                img_path = os.path.join(known_dir, fname)
                img = Image.open(img_path).convert('RGB')
                emb = get_embedding(img)
                known_embeddings.append(emb)
                known_names.append(os.path.splitext(fname)[0])
            except Exception as e:
                st.warning(f"Error processing {fname}: {str(e)}")
    
    if known_embeddings:
        return np.array(known_embeddings), known_names
    return np.empty((0, 768)), []

def recognize(face_img, known_embeddings, known_names, threshold=0.6):
    query_emb = get_embedding(face_img)
    
    if known_embeddings.shape[0] == 0:
        return "No known faces", 0.0
    
    # Compute cosine similarity
    norm_query = np.linalg.norm(query_emb)
    norm_known = np.linalg.norm(known_embeddings, axis=1)
    similarities = np.dot(known_embeddings, query_emb) / (norm_known * norm_query + 1e-8)
    
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]
    
    if best_similarity > threshold:
        return known_names[best_idx], best_similarity
    return "Unknown", best_similarity

# --- Streamlit UI ---
st.title("Face Recognition with Vision Transformer")

# Configuration sidebar
with st.sidebar:
    st.header("Configuration")
    similarity_threshold = st.slider("Recognition Threshold", 0.0, 1.0, 0.6, 0.05)
    known_dir = st.text_input("Known Faces Directory", "known_faces")
    st.info("Add face images to the 'known_faces' directory. Filename = person's name")

# Camera input
img_file_buffer = st.camera_input("Take a picture for recognition")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Captured Image", use_column_width=True)
    
    # Load known faces and recognize
    known_embeddings, known_names = load_known_embeddings(known_dir)
    name, confidence = recognize(image, known_embeddings, known_names, similarity_threshold)
    
    with col2:
        st.subheader("Recognition Result")
        st.metric("Identity", name)
        st.metric("Confidence", f"{confidence:.2f}")
        
        # Show known faces if recognized
        if name != "Unknown" and name != "No known faces":
            st.subheader("Known Faces")
            try:
                # Try common image extensions
                extensions = ['.jpg', '.jpeg', '.png']
                found = False
                for ext in extensions:
                    known_face_path = os.path.join(known_dir, f"{name}{ext}")
                    if os.path.exists(known_face_path):
                        known_face = Image.open(known_face_path)
                        st.image(known_face, caption=f"Registered: {name}", width=200)
                        found = True
                        break
                if not found:
                    st.warning("Registered image not found")
            except Exception as e:
                st.error(f"Error loading registered image: {str(e)}")
