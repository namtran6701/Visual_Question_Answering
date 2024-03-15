# Visual Question Answering with BLIP

This Streamlit app allows users to perform visual question answering (VQA) using the BLIP model. Users can select example images or upload their own images, ask questions about the images, and receive answers generated by the BLIP model.

## Getting Started
To run the app locally, follow these steps:

1. Clone this repository to your local machine
git clone https://github.com/your-username/your-repository.git

2. Install the required dependencies using pip
pip install -r requirements.txt

3. Run the Streamlit app
streamlit run app.py

4. Open your web browser and navigate to http://localhost:8501 to access the app.

Usage

1. Upon opening the app, users are presented with a title and instructions to select an example image or upload their own image.

2. Users can select an example image from the dropdown menu or upload their own image using the file uploader.

3. Users can enter a question related to the selected or uploaded image in the text input field.

4. After entering the question, the app displays the selected/uploaded image and generates an answer to the question using the BLIP model.

5. Users can view the generated answer below the image.

