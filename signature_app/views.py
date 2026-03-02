import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # This hides info/warning logs
import joblib
import json
import numpy as np
from django.shortcuts import render, redirect
from django.http import JsonResponse
from .utils import extract_from_file, calculate_14_features, generate_signature_plot, get_reference_signature
import tensorflow as tf
from django.shortcuts import render
from django.conf import settings
from PIL import Image

# ==========================================
# 1. MODEL LOADING
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'ml_models')

try:
    model = joblib.load(os.path.join(MODEL_DIR, 'pd_authenticator.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
except Exception as e:
    print(f"CRITICAL ERROR: {e}")

# ==========================================
# 2. NAVIGATION VIEWS
# ==========================================
def index_hub(request):
    return render(request, 'index.html')

def upload_page(request):
    return render(request, 'upload.html', {'user_ids': range(1, 41)})

def live_page(request):
    return render(request, 'live_pad.html', {'user_ids': range(1, 41)})

# ==========================================
# 3. AUTHENTICATION LOGIC
# ==========================================

# FILE UPLOAD HANDLER
def authenticate_file(request):
    if request.method == 'POST':
        claimed_id = int(request.POST.get('user_id'))
        uploaded_file = request.FILES.get('signature_file')
        
        if uploaded_file:
            raw_data = []
            uploaded_file.seek(0)
            for line in uploaded_file:
                parts = line.decode('utf-8').strip().split()
                if len(parts) == 7:
                    raw_data.append([float(p) for p in parts])
            
            raw_array = np.array(raw_data)
            if len(raw_array) < 15:
                return render(request, 'result.html', {'message': "Invalid File", 'color': 'gray'})

            features = calculate_14_features(raw_array)
            result_data = run_inference(features, claimed_id, raw_array)
            return render(request, 'result.html', result_data)
            
    return render(request, 'upload.html')

# LIVE PAD HANDLER (This was the missing one!)
def authenticate_live(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            claimed_id = int(body.get('user_id'))
            coords = np.array(body.get('data')) # From JavaScript Canvas
            
            if len(coords) < 15:
                return JsonResponse({'message': 'Signature too short', 'status': 'Error'})

            features = calculate_14_features(coords)
            result_data = run_inference(features, claimed_id, coords)
            
            # Since this is an AJAX call, we return JSON
            return JsonResponse(result_data)
        except Exception as e:
            return JsonResponse({'message': str(e), 'status': 'Error'})
    return JsonResponse({'message': 'Invalid Request'}, status=400)

# ==========================================
# 4. SHARED INFERENCE HELPER
# ==========================================
def run_inference(features, claimed_id, raw_array):
    scaled_feats = scaler.transform([features])
    probs = model.predict_proba(scaled_feats)[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    actual_id = label_encoder.inverse_transform([pred_idx])[0]
    
    input_img = generate_signature_plot(raw_array)
    REF_DIR = os.path.join(BASE_DIR, 'reference_signatures')
    reference_img = get_reference_signature(claimed_id, REF_DIR)
    
    threshold = 0.65
    status = "Authenticated" if (actual_id == claimed_id and confidence >= threshold) else "Denied"
    color = "green" if status == "Authenticated" else "red"
    
    return {
        'message': f"Result for User {claimed_id}",
        'status': status,
        'color': color,
        'confidence': f"{confidence*100:.2f}%",
        'input_img': input_img,
        'ref_img': reference_img,
        'claimed_id': claimed_id
    }





# Load model once at server start
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'signature_cnn_model.h5')
SIGNATURE_MODEL = tf.keras.models.load_model(MODEL_PATH)

def authenticate_by_image(request):
    # Your exact list from Colab
    USER_MAPPING = [
        'U1', 'U10', 'U11', 'U12', 'U13', 'U14', 'U15', 'U16', 'U17', 'U18', 'U19', 
        'U2', 'U20', 'U21', 'U22', 'U23', 'U24', 'U25', 'U26', 'U27', 'U28', 'U29', 
        'U3', 'U30', 'U31', 'U32', 'U33', 'U34', 'U35', 'U36', 'U37', 'U38', 'U39', 
        'U4', 'U40', 'U5', 'U6', 'U7', 'U8', 'U9'
    ]

    if request.method == 'POST' and request.FILES.get('signature_image'):
        try:
            file = request.FILES['signature_image']
            img = Image.open(file).convert('RGB').resize((128, 128))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

            predictions = SIGNATURE_MODEL.predict(img_array)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])

            user_id = USER_MAPPING[predicted_idx].replace('U', 'User ')
            
            # Store result in session for the next page
            request.session['auth_result'] = {
                'identified_user': user_id,  # Changed from 'user' to 'identified_user'
                'confidence': f"{confidence * 100:.2f}%",
                'color': "green" if confidence > 0.80 else "orange"
            }
            return redirect('show_results')

        except Exception as e:
            return render(request, 'live_pad.html', {'error': str(e)})

    return render(request, 'live_pad.html')

def show_results(request):
    data = request.session.get('auth_result')
    if not data:
        
        return redirect('authenticate_image') 
    return render(request, 'result.html', data)