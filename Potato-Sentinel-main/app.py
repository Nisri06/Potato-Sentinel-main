from flask import Flask, render_template, request, redirect
import os
from werkzeug.utils import secure_filename
from io import BytesIO
import base64
import tensorflow as tf
import joblib
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)  # Change to INFO or ERROR in production
logger = logging.getLogger(__name__)

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

CLASS_NAMES = ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']

CLASS_DESCRIPTIONS = {
    'Potato___Early_blight': '''
    Early Blight is a common fungal disease caused by *Alternaria solani* that affects potato plants. 
    It typically starts as small, dark spots with concentric rings on older leaves, often leading to yellowing and premature leaf drop. 
    The disease can progress rapidly in humid, warm conditions, significantly reducing potato yields. 
    Early blight can also lead to the formation of lesions on tubers, affecting their marketability and storage life. 
    Proper management includes crop rotation, resistant varieties, and timely fungicide applications.
    ''',

    'Potato___healthy': '''
    Healthy potato plants exhibit vibrant green leaves and robust growth. The plant's foliage is free from any spots, discolorations, or lesions, 
    indicating that it is not affected by any fungal or bacterial diseases. A healthy potato plant has well-developed stems and roots, 
    and the tubers are firm, with no signs of decay, discoloration, or other diseases. Proper soil nutrition, adequate irrigation, 
    and pest control are crucial for maintaining plant health and maximizing yield.
    ''',

    'Potato___Late_blight': '''
    Late Blight, caused by the oomycete *Phytophthora infestans*, is one of the most destructive diseases in potato cultivation. 
    It manifests as dark, water-soaked lesions on leaves and stems, which quickly expand and result in rapid tissue death. 
    This disease thrives in cool, wet conditions, and if left unchecked, can lead to complete crop loss. 
    Infected tubers may develop soft rot and become unfit for consumption or storage. Effective management strategies include 
    resistant cultivars, early detection, and regular fungicide applications, especially during rainy periods.
    '''
}


def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    logger.debug(f"Checking file extension for: {filename}")
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_stream):
    try:
        logger.debug("Starting image preprocessing.")
        
        # Load image and resize it to the target size
        img = tf.keras.preprocessing.image.load_img(img_stream, target_size=(256, 256))
        logger.debug("Image loaded and resized to (256, 256).")
        
        # Convert image to array
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        logger.debug(f"Image converted to array with shape: {img_array.shape} and dtype: {img_array.dtype}.")
        
        # Expand dimensions for batch processing
        img_array = np.expand_dims(img_array, axis=0)
        logger.debug(f"Image array shape after expanding dimensions: {img_array.shape}.")
        
        return img_array
    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}")
        raise

def predict_image(img_stream, ensemble_model, rf_model, scaler):
    try:
        logger.debug("Starting prediction on image...")

        # Step 1: Preprocess the image
        logger.debug("Starting image preprocessing...")
        img_array = preprocess_image(img_stream)
        logger.debug(f"Completed image preprocessing. Resulting shape: {img_array.shape}, dtype: {img_array.dtype}")

        # Step 2: Make predictions with the ensemble model
        logger.debug("Starting prediction with the ensemble (CNN) model...")
        cnn_features = ensemble_model.predict(img_array)
        logger.debug(f"Completed ensemble model prediction. Output shape: {cnn_features.shape}")

        # Step 3: Transform CNN features for the random forest model
        logger.debug("Starting feature scaling with scaler for Random Forest model...")
        rf_features = scaler.transform(cnn_features)
        logger.debug(f"Completed feature scaling. Scaled features shape: {rf_features.shape}, sample values: {rf_features[:5]}")

        # Step 4: Make predictions with the random forest model
        logger.debug("Starting prediction with the Random Forest model...")
        rf_predictions = rf_model.predict(rf_features)
        logger.debug(f"Completed Random Forest prediction. Prediction output: {rf_predictions}")

        # Step 5: Map prediction to class name
        logger.debug("Mapping prediction to class name...")
        predicted_class = CLASS_NAMES[rf_predictions[0]]
        logger.debug(f"Prediction result mapped to class name: {predicted_class}")

        return predicted_class

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

def get_class_description(predicted_class):
    try:
        logger.debug(f"Fetching description for class: {predicted_class}")
        description = CLASS_DESCRIPTIONS.get(predicted_class, "Description not available.")
        return description
    except Exception as e:
        logger.error(f"Error during description retrieval: {e}")
        raise

@app.route('/')
def index_page():
    """Serve the AI page."""
    logger.debug("Rendering home page")
    return render_template('home.html')

@app.route('/home')
def home_page():
    """Serve the home page."""
    logger.debug("Rendering home page")
    return render_template('home.html')

@app.route('/about')
def about_page():
    logger.debug("Rendering about page")
    return render_template('about.html')

@app.route('/ai')
def ai_page():
    logger.debug("Rendering AI page")
    return render_template('ai.html')

@app.route('/faq')
def faq_page():
    logger.debug("Rendering FAQ page")
    return render_template('faq.html')

@app.route('/tnc')
def tnc_page():
    logger.debug("Rendering Terms & Conditions page")
    return render_template('tnc.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle the file upload, load models, and run prediction."""
    logger.debug("Handling file upload")
    
    if 'file' not in request.files:
        logger.error("No file part in the request")
        return redirect(request.url)
    
    file = request.files['file']
    logger.debug(f"File selected: {file.filename}")

    if file.filename == '':
        logger.error("No file selected")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        logger.debug(f"Valid file extension for {file.filename}")

        # Use BytesIO to handle the file in memory
        file_stream = BytesIO(file.read())
        filename = secure_filename(file.filename)
        logger.debug(f"File read into memory: {filename}")

        try:
            # Reload models every time an image is uploaded
            logger.debug("Loading ensemble model...")
            ensemble_model = tf.keras.models.load_model('ensemble_model.h5')
            logger.debug("Ensemble model loaded successfully.")

            logger.debug("Loading random forest model...")
            rf_model = joblib.load('rf_model.pkl')
            logger.debug("Random Forest model loaded successfully.")

            logger.debug("Loading scaler...")
            scaler = joblib.load('scaler.pkl')
            logger.debug("Scaler loaded successfully.")
            
            # Read the image and convert to base64
            image_data = base64.b64encode(file_stream.getvalue()).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{image_data}"
            logger.debug("Image converted to base64")

            # Run the prediction on the in-memory file stream
            prediction = predict_image(file_stream, ensemble_model, rf_model, scaler)
            logger.debug(f"Prediction received: {prediction}")

            # Convert prediction to user-friendly name
            friendly_class_name = {
                'Potato___Early_blight': 'Early Blight',
                'Potato___healthy': 'Healthy',
                'Potato___Late_blight': 'Late Blight'
            }.get(prediction, 'Unknown')
            logger.debug(f"Friendly class name: {friendly_class_name}")

            description = get_class_description(prediction)
            logger.debug(f"Class description: {description}")

            # Pass the prediction to the result page
            return render_template('result.html', predicted_class=prediction, image_url=image_url, description=description, friendly_class_name=friendly_class_name)

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "An error occurred during prediction."
        
    else:
        logger.error(f"Invalid file extension: {file.filename}")
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
