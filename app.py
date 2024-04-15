import numpy as np
import tensorflow 
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os

app = Flask(__name__, template_folder='templates')

model = load_model("classification.h5")

DATA_FOLDER = os.path.join(os.path.expanduser("~"), "Desktop", "DATA")

target_size = (224, 224)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':

        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
       
        if file.filename == '':
            return redirect(request.url)
        if file:
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(DATA_FOLDER, filename)
            file.save(file_path)

           
            img = image.load_img(file_path, target_size=target_size)
            
            x = image.img_to_array(img)
            
            x = np.expand_dims(x, axis=0)
            
            x = preprocess_input(x)

          
            preds = model.predict(x)
            
            preds = np.argmax(preds, axis=1)

            
            if preds == 0:
                prediction = "The leaf is diseased cotton leaf"
            elif preds == 1:
                prediction = "The leaf is diseased cotton plant"
            elif preds == 2:
                prediction = "The leaf is fresh cotton leaf"
            else:
                prediction = "The leaf is fresh cotton plant"

            return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
