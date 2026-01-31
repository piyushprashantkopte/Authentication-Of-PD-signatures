import numpy as np
from scipy.stats import entropy

def extract_features_expert(file_stream):
    try:
        data_rows = []
        # Process the uploaded file stream
        for line in file_stream:
            parts = line.decode('utf-8').strip().split()
            if len(parts) == 7:
                try:
                    data_rows.append([float(p) for p in parts])
                except ValueError:
                    continue
        
        if len(data_rows) < 15:
            return None

        arr = np.array(data_rows)
        # Mapping: 0:x, 1:y, 2:t, 3:status, 4:p, 5:az, 6:al
        x, y, t, p = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 4]

        # --- High-Accuracy Kinematics (Matches your 14-feature training) ---
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

        # This list MUST have exactly 14 items in this specific order
        return [
            np.mean(velocity), np.std(velocity), np.max(velocity), # 1, 2, 3
            energy, sig_entropy,                                  # 4, 5
            np.mean(p), np.std(p), p_v_corr,                     # 6, 7, 8
            np.mean(np.abs(jerk)) if len(jerk) > 0 else 0,        # 9
            pauses,                                               # 10
            np.max(x) - np.min(x),                                # 11 (Width)
            np.max(y) - np.min(y),                                # 12 (Height)
            np.max(t) - np.min(t),                                # 13 (Duration)
            len(arr)                                              # 14 (Total Points)
        ]
    except Exception as e:
        print(f"Extraction Error: {e}")
        return None