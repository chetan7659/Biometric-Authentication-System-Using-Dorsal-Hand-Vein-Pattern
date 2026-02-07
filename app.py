import streamlit as st
import cv2
import numpy as np
import os
import csv
from sklearn.preprocessing import normalize
from scipy.spatial import distance
from cancellability.feature_rdmtransform import transformMeximumCurvatureRDM as MCRG
from classification import KFA, Projection, Normo

# --- Configuration ---
DATASET_PATH = './sample dataset/veinpattern/'
KEY1_PATH = './create_csv/Key01.csv'
KEY2_PATH = './create_csv/Key02.csv'
IMG_SIZE = (100, 100)

st.set_page_config(page_title="Dorsal Hand Vein Auth", layout="wide")

@st.cache_resource
def load_keys():
    """Lengths keys from CSV files."""
    keys = []
    for path in [KEY1_PATH, KEY2_PATH]:
        key_mat = []
        if os.path.exists(path):
            with open(path) as f:
                reader = csv.reader(f)
                for row in reader:
                    key_mat.append([float(x) for x in row])
            # Normalize
            key_mat = np.array(key_mat)
            key_mat = np.nan_to_num(key_mat) 
            key_mat = normalize(key_mat)
            keys.append(key_mat)
        else:
             st.error(f"Key file not found: {path}")
             return None, None
    return keys[0], keys[1]

@st.cache_data
def train_model():
    """
    Loads dataset, extracts features, and trains the KFA model.
    """
    KeyMat1, KeyMat2 = load_keys()
    if KeyMat1 is None: return None, None, None, None

    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset path not found: {DATASET_PATH}")
        return None, None, None, None

    subjects = os.listdir(DATASET_PATH)
    # Filter for directories
    subjects = [s for s in subjects if os.path.isdir(os.path.join(DATASET_PATH, s))]
    
    # Check if we have enough keys
    if len(KeyMat1) < len(subjects):
         st.warning(f"Warning: Only {len(KeyMat1)} keys available but {len(subjects)} subjects found. Truncating subjects list.")
         subjects = subjects[:len(KeyMat1)]

    data_matrix = []
    ids_train = []
    
    k, N = IMG_SIZE
    
    total_subjects = len(subjects)
    progress_text = "Training Progress"
    my_bar = st.progress(0, text=progress_text)

    for i, subject in enumerate(subjects):
        subject_path = os.path.join(DATASET_PATH, subject)
        images = os.listdir(subject_path)
        
        Key1 = KeyMat1[i].reshape(k * N, 1)
        Key2 = KeyMat2[i].reshape(k * N, 1)

        for img_name in images:
            img_path = os.path.join(subject_path, img_name)
            img = cv2.imread(img_path, 0)
            if img is None: continue
            
            img = np.asarray(img, dtype=float)
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
            
            # Feature Extraction & Transformation
            # Note: MCRG returns TfM (transformed feature) and fv (original scaled)
            transformed_feature, _ = MCRG(img, Key1, Key2)
            
            # Flatten to 1D vector
            vec = transformed_feature.flatten() 
            data_matrix.append(vec)
            ids_train.append(i) 
            
        my_bar.progress((i + 1) / total_subjects, text=progress_text)
            
    # Prepare training data
    if not data_matrix:
        st.error("No valid images found for training.")
        return None, None, None, None
        
    train_data = np.array(data_matrix).T # (Features, Samples)
    train_data = np.nan_to_num(train_data) # Fix NaNs
    
    st.success(f"Training on {train_data.shape[1]} samples from {len(subjects)} subjects.")
    
    try:
        # KFA expects ids to be a list/array of class labels
        # n = number of components. Max is classes-1.
        n_components = len(subjects) - 1
        if n_components < 1: n_components = 1
        
        model = KFA.perform_kfa_PhD(train_data, ids_train, 'fpp', n_components)
    except Exception as e:
        st.error(f"Training failed: {e}")
        return None, None, None, None

    return model, subjects, (KeyMat1, KeyMat2), ids_train

# --- UI ---
st.title("Biometric Authentication System")
st.markdown("## Dorsal Hand Vein Based Cancellable Authentication")
st.markdown("This application demonstrates secure, cancellable biometric authentication using dorsal hand vein patterns.")

# Loading
with st.spinner("Loading System & Training KFA Model... This may take a minute."):
    model, subjects, keys, ids_train_list = train_model()

if model is None:
    st.stop()

st.divider()

# Testing Interface
col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Upload Probe Image")
    uploaded_file = st.file_uploader("Upload Hand Image", type=['jpg', 'png', 'bmp', 'jpeg'])
    
    st.header("2. Claim Identity")
    claim_idx = st.selectbox("Select Subject ID to Verify Against", range(len(subjects)), format_func=lambda x: f"Subject {x+1}: {subjects[x]}")

with col2:
    st.header("Authentication Result")
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 0) # Load as grayscale
        
        st.image(opencv_image, caption="Probe Image", width=300)
        
        if st.button("Verify Identity", type="primary"):
            # logic
            k, N = IMG_SIZE
            KeyMat1, KeyMat2 = keys
            
            # 1. Transform Probe using Claimed User's Key
            # This is the "Cancellable" part - we apply the transform using the key of the claim.
            Key1 = KeyMat1[claim_idx].reshape(k*N, 1)
            Key2 = KeyMat2[claim_idx].reshape(k*N, 1)
            
            probe_processed = np.asarray(opencv_image, dtype=float)
            probe_processed = cv2.resize(probe_processed, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
            
            feat_img, _ = MCRG(probe_processed, Key1, Key2)
            
            # Need to reshape for projection (column vector)
            # wait, projection expects (feature_dim, n_samples)
            # Feature dim is 100*100 = 10000? MCRG returns shape?
            # MCRG returns TfM (median filtered image). Shape 100x100? No, check MCRG.
            # MCRG line 39: fvs = np.array(img).reshape(10000, 1).
            # MCRG returns TfM which is likely same shape after arithmetic.
            # So 10000 elements. flatten() gives (10000,).
            
            processed_vec = feat_img.flatten() # 1D array (10000,)
            test_data_col = processed_vec.reshape(-1, 1) # (10000, 1)
            
            # 2. Project into Subspace
            # nonlinear_subspace_projection_PhD expects X as (features, samples)
            try:
                proj_vec = Projection.nonlinear_subspace_projection_PhD(test_data_col, model)
                v = np.array(proj_vec).flatten()
            except Exception as e:
                st.error(f"Projection failed: {e}")
                st.stop()
            
            # 3. Euclidean Distance
            # We compare the projected probe 'v' against enrolled vectors 'u'
            # But WHICH enrolled vectors?
            # For Verification: We compare against enrolled vectors of the CLAIMED subject.
            # Get indices of training samples belonging to claim_idx
            
            relevant_indices = [i for i, x in enumerate(ids_train_list) if x == claim_idx]
            
            min_dist = float('inf')
            
            dt = model.train # (n_components, n_samples)
            
            distances = []
            
            for idx in relevant_indices:
                u = np.array(dt[:, idx]).flatten()
                d = distance.euclidean(v, u)
                distances.append(d)
                if d < min_dist:
                   min_dist = d
            
            st.metric("Minimum Distance", f"{min_dist:.2f}")
            
            # Threshold (from main.py)
            THRESHOLD = 100.0
            
            if min_dist <= THRESHOLD:
                st.success(f"ACCESS GRANTED! Verified as Subject {claim_idx+1}")
                st.snow()
            else:
                st.error(f"ACCESS DENIED. Mismatch with Subject {claim_idx+1}")
                st.info(f"Distance {min_dist:.2f} exceeds threshold {THRESHOLD}")

    else:
        st.info("Upload an image to start.")
