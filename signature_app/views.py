import os
import joblib
import numpy as np
from django.shortcuts import render
from .utils import extract_features_expert

# 1. PATH SETUP: Find the models in the 'ml_models' folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'ml_models')

# 2. LOAD MODELS: Load them globally for speed
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'pd_authenticator.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
except Exception as e:
    print(f"Error loading models: {e}")

# View for the initial upload page
def upload_page(request):
    user_ids = range(1, 41)  # Creates list for User 1 to User 40
    return render(request, 'upload.html', {'user_ids': user_ids})

# View to process the authentication
def authenticate_signature(request):
    if request.method == 'POST':
        # Get data from the form
        claimed_id = int(request.POST.get('user_id'))
        uploaded_file = request.FILES.get('signature_file')
        
        if uploaded_file:
            # Step A: Extract features using our expert utility
            feats = extract_features_expert(uploaded_file)
            
            if feats is None:
                return render(request, 'result.html', {
                    'message': "Error: Could not process file. Ensure it is a valid SVC2004 .txt file.",
                    'color': 'gray'
                })

            # Step B: Scale the features
            scaled_feats = scaler.transform([feats])
            
            # Step C: Predict probability (Confidence)
            probs = model.predict_proba(scaled_feats)[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            
            # Step D: Decode the index back to the real User ID
            actual_id = label_encoder.inverse_transform([pred_idx])[0]
            
            # Step E: Authentication Logic
            threshold = 0.85 # 85% confidence required for 'YES'
            
            if actual_id == claimed_id and confidence >= threshold:
                msg = f"YES, it is the authenticated signature of User {claimed_id}"
                color = "green"
                status = "Success"
            elif actual_id == claimed_id and confidence < threshold:
                msg = f"Pattern matches User {claimed_id}, but confidence is too low ({confidence*100:.1f}%). Please sign again."
                color = "orange"
                status = "Uncertain"
            else:
                msg = f"NO, it is NOT the authenticated signature of User {claimed_id}"
                color = "red"
                status = "Failed"
                
            return render(request, 'result.html', {
                'message': msg, 
                'color': color, 
                'confidence': f"{confidence*100:.2f}%",
                'status': status
            })
            
    return render(request, 'upload.html')