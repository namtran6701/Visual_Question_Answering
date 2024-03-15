import streamlit as st
from transformers.utils import logging
from transformers import AutoProcessor, BlipForQuestionAnswering
from PIL import Image
import warnings

# Suppress warnings (optional)
logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

# Set page configuration
st.set_page_config(layout="centered")

# Load the model and processor
@st.cache_resource  # Cache for efficiency
def load_models():
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
    return model, processor

def display_title():
    st.markdown("""
    <h1 style='text-align: center; color: #3498db;'>Visual Question Answering with 
    <a href='https://huggingface.co/Salesforce/blip-vqa-base' target='_blank'>BLIP</a></h1> 
    <h3 style='text-align: center;'>Select an example image or upload your own and ask a question about it!</h3>
    """, unsafe_allow_html=True)

def display_selected_image(selected_image_path):
    if selected_image_path:
        image = Image.open(selected_image_path)
        st.image(image, caption="Selected Image", width=400)

def process_image_and_question(image_file, selected_image_path, question, model, processor):
    if (image_file is not None or selected_image_path) and question:
        if image_file:
            image = Image.open(image_file)
        elif selected_image_path:
            image = Image.open(selected_image_path)

        inputs = processor(image, question, return_tensors="pt")
        output = model.generate(**inputs)
        answer = processor.decode(output[0], skip_special_tokens=True)
        
        st.image(image, caption="Uploaded Image", width=400)
        st.subheader("Answer:")
        st.write(answer)
        

def main():
    display_title()

    # Image examples
    image_paths = ["pic_1.png", "pic_2.png", "pic_3.jpg"]  # Provide paths to your example images

    # Dropdown for selecting example images
    selected_image_path = st.selectbox("Select an example image:", image_paths)

    # Display selected image
    display_selected_image(selected_image_path)

    # File uploader for uploading custom images
    image_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    # Input question
    question = st.text_input("Enter your question:")

    # Load models
    model, processor = load_models()

    # Process image and question
    process_image_and_question(image_file, selected_image_path, question, model, processor)

    # Author information at the bottom
    st.markdown("---")
    st.caption("Made by Nam Tran (Meow)")

if __name__ == "__main__":
    main()
