from flask import Flask, request, render_template 
import numpy as np 
import cv2
import tensorflow as tf 
from werkzeug.utils import secure_filename 
from tensorflow.keras.models import load_model 
from tensorflow.keras.losses import binary_crossentropy
import os 

def dice_loss(y_true, y_pred):     
    numerator = 2 * tf.reduce_sum(y_true * y_pred)     
    denominator = tf.reduce_sum(y_true + y_pred)     
    return 1 - numerator / denominator  

def dice_coef(y_true, y_pred, smooth=1):     
    intersection = tf.reduce_sum(y_true * y_pred)     
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)  


def iou(y_true, y_pred, smooth=1):     
    intersection = tf.reduce_sum(y_true * y_pred)     
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection     
    return (intersection + smooth) / (union + smooth)  

def preprocess_image(image_path):     
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)     
    img = cv2.resize(img, (256, 256))     
    img = img / 255.0     
    img = np.expand_dims(img, axis=0)     
    return img  

def predict_image(image_path):     
    input_image = preprocess_image(image_path)     
    predicted_mask = model.predict(input_image)     
    return input_image[0], predicted_mask[0]  

def calculate_volume(mask, pixel_resolution):     
    num_nonzero_pixels = np.count_nonzero(mask)     
    volume = num_nonzero_pixels * pixel_resolution     
    return volume  

class AppConfig:     
    DEBUG = True     
    TESTING = False  
    
app = Flask(__name__, template_folder='templates') 
app.config.from_object(AppConfig) 

model_path = os.path.join(app.root_path, 'model2.h5') 
model = load_model(model_path, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef, 'iou': iou})
@app.route('/') 
def index():     
    return render_template('model.html')  

@app.route('/predict', methods=['POST']) 
def predict():     
    if 'image' not in request.files:         
        flash('No file part', 'error')         
        return redirect(request.url)     
    
    img = request.files['image']
        
    if img.filename == '':
        raise ValueError("No file selected") # Save the uploaded image
    img_path = "static/uploaded_image.jpg" 
    img.save(img_path)
          
        
    input_image, predicted_mask = predict_image(img_path)     
    thresholded_mask = (predicted_mask > 0.5).astype(np.uint8) * 255     
    resized_mask = cv2.resize(thresholded_mask, (input_image.shape[1], input_image.shape[0]))     
    
    if np.sum(resized_mask) == 0:         
        message = "No tumor detected"         
        cv2.putText(input_image, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)         
        cv2.imwrite('temp.png', input_image)         
        cv2.imshow('Overlay', input_image)         
        cv2.waitKey(0)         
        cv2.destroyAllWindows()         
        os.remove('temp.png')         
        return render_template('model.html', prediction_value=message)     
    
    else:         
        pixel_resolution = 1  # Adjust based on your data         
        volume = calculate_volume(thresholded_mask, pixel_resolution)         
        shaded_mask = np.zeros_like(input_image)         
        shaded_mask[resized_mask > 0] = [0, 0, 255]         
        overlay = cv2.addWeighted(input_image, 0.7, shaded_mask, 0.3, 0)         
        cv2.imwrite('temp.png', overlay)         
        cv2.imshow('Overlay', overlay) 
                
        cv2.waitKey(0)         
        cv2.destroyAllWindows()         
        os.remove('temp.png')         
        return render_template('model.html', prediction_value="Volume of the mask: {:.2f} cubic units".format(volume))

    
if __name__ == "__main__":     
    app.run(host='localhost', port=5000, debug=True)
