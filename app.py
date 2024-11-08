from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Charger le modèle pré-entraîné (model.h5)
model = load_model('model.h5')

# Chemin pour enregistrer les images téléchargées
target_img = os.path.join(os.getcwd(), 'static/images')
if not os.path.exists(target_img):
    os.makedirs(target_img)

# Extensions de fichiers autorisées
ALLOWED_EXT = {'jpg', 'jpeg', 'png'}

# Fonction pour vérifier si le type de fichier est autorisé
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# Fonction pour charger et préparer l'image dans le bon format
def read_image(filename):
    # Charger l'image et la redimensionner à (32, 32) comme attendu par le modèle
    img = image.load_img(filename, target_size=(32, 32))
    # Convertir l'image en tableau numpy et ajouter une dimension pour le traitement par lot
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # Normaliser les valeurs des pixels à [0, 1]
    x = x / 255.0
    return x

# Route pour la page d'accueil
@app.route('/')
def index_view():
    return render_template('index.html')

# Route pour prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message="No file part")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', message="No selected file")
    
    if file and allowed_file(file.filename):
        # Enregistrer le fichier téléchargé
        filename = file.filename
        file_path = os.path.join(target_img, filename).replace('\\', '/')
        file.save(file_path)

        # Préparer l'image pour le modèle
        img_array = read_image(file_path)

        # Faire la prédiction
        class_prediction = model.predict(img_array)
        predicted_class = np.argmax(class_prediction[0])
        class_names = ["apple", "banana", "orange"]  # Mettez à jour ceci si votre modèle a plus de classes ou différentes
        predicted_class_name = class_names[predicted_class]

        return render_template('predict.html', 
                               predicted_class=predicted_class_name,  
                               user_image=file_path)
    else:
        return render_template('index.html', message="Invalid file type. Please upload a jpg, jpeg, or png image.")

# Route pour le favicon
@app.route('/favicon.ico')
def favicon():
    return '', 204  # Pas de contenu pour éviter l'erreur 404

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)
