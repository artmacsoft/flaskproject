from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')

# Function to get RoBERTa embeddings for a given text
def get_roberta_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Function to compute cosine similarity
def compute_similarity_roberta(job_description, candidate_profile, tokenizer, model):
    job_embedding = get_roberta_embeddings(job_description, tokenizer, model)
    candidate_embedding = get_roberta_embeddings(candidate_profile, tokenizer, model)
    similarity_score = cosine_similarity(job_embedding, candidate_embedding)
    return similarity_score[0][0]

# Define the anomaly detection model
def build_autoencoder(input_dim, encoding_dim=2):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder

# Anomaly detection
def detect_anomalies(autoencoder, profile_data, scaler, threshold=0.05):
    profile_data_scaled = scaler.transform(profile_data)
    reconstructed_data = autoencoder.predict(profile_data_scaled)
    reconstruction_error = np.mean(np.square(profile_data_scaled - reconstructed_data), axis=1)
    anomalies = reconstruction_error > threshold
    return anomalies, reconstruction_error

# Route for uploading and processing files
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        job_description = request.form['job_description']
        uploaded_files = request.files.getlist("file[]")
        profiles = []
        for file in uploaded_files:
            if file and file.filename:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                with open(file_path, 'r') as f:
                    profiles.append(f.read())
        
        # Example real profiles (these should be replaced with actual data)
        real_profiles = np.array([[5, 3.5, 2], [4, 4, 2], [3, 5, 1]])  #Profile 1: [5, 3, 1] (5 years of experience, 3 technologies, Bachelor’s Degree)
                                                                       #Profile 2: [4, 2, 2] (4 years, 2 technologies, Master’s Degree)
                                                                       #Profile 3: [6, 4, 1] (6 years, 4 technologies, Bachelor’s Degree)
        scaler = StandardScaler()
        real_profiles_scaled = scaler.fit_transform(real_profiles)
        autoencoder = build_autoencoder(input_dim=real_profiles.shape[1])
        autoencoder.fit(real_profiles_scaled, real_profiles_scaled, epochs=50, batch_size=8, shuffle=True)

        results = []
        for profile in profiles:
            similarity_score = compute_similarity_roberta(job_description, profile, tokenizer, roberta_model)
            profile_features = np.array([[7, 1, 3]])  # Example features for profile
            anomalies, reconstruction_error = detect_anomalies(autoencoder, profile_features, scaler)
            anomaly_score = reconstruction_error[0]
            final_score = 0.7 * similarity_score - 0.3 * anomaly_score
            results.append({
                'profile': profile,
                'similarity_score': similarity_score,
                'anomaly_score': anomaly_score,
                'final_score': final_score,
                'is_anomalous': anomalies[0]
            })
        
        return render_template('results.html', results=results)
    
    return render_template('upload.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
