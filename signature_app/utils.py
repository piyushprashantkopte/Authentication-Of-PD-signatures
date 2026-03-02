import matplotlib
matplotlib.use('Agg')  # This tells Matplotlib NOT to use a GUI/Tkinter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
import io
import os
import base64

# --- 1. THE MATH ENGINE (Expected by views.py) ---
def calculate_14_features(arr):
    """
    Standardizes the 14-feature extraction logic for both 
    live coordinates and file-based data.
    """
    try:
        # Mapping: 0:x, 1:y, 2:t, 3:status, 4:p, 5:az, 6:al
        x, y, t, p = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 4]

        dt = np.diff(t) + 1e-6
        dist = np.hypot(np.diff(x), np.diff(y))
        velocity = dist / dt
        
        # Calculate Acceleration and Jerk
        accel = np.diff(velocity) / dt[:-1]
        jerk = np.diff(accel) / (dt[:-2] + 1e-6)
        
        # 1. Number of micro-pauses
        pauses = np.sum(velocity < 0.1)
        
        # 2. Signal Energy & Entropy
        energy = np.sum(velocity**2)
        hist, _ = np.histogram(velocity, bins=10, density=True)
        sig_entropy = entropy(hist + 1e-6)
        
        # 3. Pressure-Velocity Correlation
        p_v_corr = np.corrcoef(velocity, p[1:])[0, 1] if np.std(p) > 0 else 0

        # Return the exact 14 features in order
        return [
            np.mean(velocity), np.std(velocity), np.max(velocity), # 1, 2, 3
            energy, sig_entropy,                                  # 4, 5
            np.mean(p), np.std(p), p_v_corr,                      # 6, 7, 8
            np.mean(np.abs(jerk)) if len(jerk) > 0 else 0,         # 9
            pauses,                                               # 10
            np.max(x) - np.min(x),                                 # 11
            np.max(y) - np.min(y),                                 # 12
            np.max(t) - np.min(t),                                 # 13
            len(arr)                                               # 14
        ]
    except Exception as e:
        print(f"Math Calculation Error: {e}")
        return None

# --- 2. THE FILE EXTRACTOR (Expected by views.py) ---
def extract_from_file(file_stream):
    """Processes uploaded .txt files and returns the 14 features."""
    try:
        data_rows = []
        for line in file_stream:
            parts = line.decode('utf-8').strip().split()
            if len(parts) == 7:
                try:
                    data_rows.append([float(p) for p in parts])
                except ValueError:
                    continue
        
        if len(data_rows) < 15:
            return None

        # Convert to numpy and pass to the math engine
        return calculate_14_features(np.array(data_rows))
    except Exception as e:
        print(f"File Extraction Error: {e}")
        return None

# --- 3. VISUALIZATION ENGINE ---
def generate_signature_plot(arr):
    try:
        # Switch to the Agg backend locally if not done globally
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        x, y = arr[:, 0], arr[:, 1]
        
        # Clear any existing plots to prevent overlaying
        plt.clf() 
        
        plt.figure(figsize=(5, 2))
        plt.plot(x, -y, color='#1a73e8', linewidth=1.5)
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        
        # CRITICAL: Close the plot to prevent memory leaks and thread errors
        plt.close('all') 
        
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Plotting error: {e}")
        return None

# --- 4. REFERENCE LOADER ---
def get_reference_signature(user_id, reference_folder):
    try:
        filename = f"U{user_id}S1.TXT" 
        path = os.path.join(reference_folder, filename)
        
        data = []
        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 7:
                    data.append([float(p) for p in parts])
        
        return generate_signature_plot(np.array(data))
    except:
        return None