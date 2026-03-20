# 🧠 Step-by-Step Guide: Rebuild Dorsal Hand Vein Biometric Authentication System

## 📌 Project Overview

This system authenticates a person using the **vein pattern on the back of their hand**.
It works in 5 stages:

```
Raw Hand Image
     ↓
[STAGE 1] Pre-Processing  → Remove hair noise from image
     ↓
[STAGE 2] Feature Extraction → Extract vein pattern (Maximum Curvature Method)
     ↓
[STAGE 3] Cancellability → Transform features using a secret key (privacy protection)
     ↓
[STAGE 4] Classification → Kernel Fisher Analysis (KFA) to build a model
     ↓
[STAGE 5] Authentication → Euclidean distance comparison → GRANT or DENY
```

---

## 🗂️ Final Folder Structure You Will Build

```
my_biometric_project/
│
├── requirements.txt                  ← Python dependencies
├── __init__.py
├── main.py                           ← Run authentication pipeline
├── app.py                            ← Streamlit web UI
├── run_app.bat                       ← Launch script
│
├── sample dataset/
│   └── veinpattern/
│       ├── s01/  (10 images each)
│       ├── s02/
│       └── ...
│
├── PreProcessing_FeatureExtraction/
│   ├── __init__.py
│   ├── normalize.py                  ← Min-max normalization
│   ├── preprocessing.py              ← Hair removal
│   ├── profile_curvature.py          ← Gaussian curvature (Step 1)
│   ├── detect_vein_center_assign_score.py  ← Vein scoring (Step 2)
│   ├── connect_center.py             ← Connect vein centers (Step 3)
│   ├── label.py                      ← Binarize vein pattern
│   └── extract_feature.py            ← Main pipeline combining all steps
│
├── cancellability/
│   ├── __init__.py
│   ├── feature_rdmtransform.py       ← Random Distance Transform (main)
│   └── feature_xortransform.py       ← XOR Transform (alternative)
│
├── classification/
│   ├── __init__.py
│   ├── Normo.py                      ← Column normalization utility
│   ├── KernelMatrix.py               ← Compute kernel matrices
│   ├── KFA.py                        ← Kernel Fisher Analysis
│   └── Projection.py                 ← Project test data into subspace
│
└── create_csv/
    ├── __init__.py
    ├── Key01.csv                     ← Secret key matrix 1
    ├── Key02.csv                     ← Secret key matrix 2
    ├── Xor_Key.csv                   ← XOR key matrix
    ├── key_01.py                     ← Script to generate Key01.csv
    ├── key_02.py                     ← Script to generate Key02.csv
    ├── xor_key.py                    ← Script to generate Xor_Key.csv
    └── create_database.py            ← Process images → CSV datasets
```

---

## 🔧 STEP 1: Setup Environment

### 1.1 Create a new project folder
```
mkdir my_biometric_project
cd my_biometric_project
```

### 1.2 Create virtual environment
```
python -m venv venv
venv\Scripts\activate
```

### 1.3 Create `requirements.txt`
```
opencv-python
numpy
scipy
scikit-learn
pandas
streamlit
```

### 1.4 Install dependencies
```
pip install -r requirements.txt
```

---

## 📁 STEP 2: Create Root Files

### 2.1 Create `__init__.py` (root)
```python
# Root package init
```

### 2.2 Create `run_app.bat`
```bat
@echo off
call venv\Scripts\activate
streamlit run app.py
pause
```

---

## 📁 STEP 3: Build `PreProcessing_FeatureExtraction/` Module

This module extracts the vein pattern from a raw dorsal hand image.

### 3.1 Create `PreProcessing_FeatureExtraction/__init__.py`
```python
# PreProcessing_FeatureExtraction package
```

### 3.2 Create `PreProcessing_FeatureExtraction/normalize.py`

**Purpose:** Scale any array to a [low, high] range.

```python
import numpy as np

def normalize_data(x, low=0, high=1, data_type=None):
    x = np.asarray(x, dtype=float)
    min_x, max_x = np.min(x), np.max(x)
    x = x - float(min_x)
    if max_x - min_x == 0:
        x = np.zeros_like(x)
    else:
        x = x / float((max_x - min_x))
    x = x * (high - low) + low
    if data_type is None:
        return np.asarray(x, dtype=float)
    return np.asarray(x, dtype=data_type)
```

### 3.3 Create `PreProcessing_FeatureExtraction/preprocessing.py`

**Purpose:** Remove hair from the dorsal hand image using a Mexican Hat kernel (convolution filter).

```python
import os
import cv2
from scipy.signal import convolve2d
from PreProcessing_FeatureExtraction.normalize import normalize_data

dir = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir, 'MexicanHatKernalData')

def remove_hair(image, mexican_kernel_size, low=1, high=4):
    try:
        read_kernel = cv2.imread(
            os.path.join(filepath, f'Kernel {mexican_kernel_size}.jpg'), 0
        )
    except FileNotFoundError:
        print('please choose correct size of kernel')
    normalized_kernel = normalize_data(read_kernel, low, high)
    hair_remove = convolve2d(image, normalized_kernel, mode='same', fillvalue=0)
    return hair_remove
```

> ⚠️ **Note:** You need the `MexicanHatKernalData/` folder with kernel images (Kernel 3.jpg, Kernel 6.jpg, etc.) — copy these from the original project.

### 3.4 Create `PreProcessing_FeatureExtraction/profile_curvature.py`

**Purpose (Step 1-1):** Compute the curvature (kappa) of the image in 4 directions (horizontal, vertical, +45°, -45°) using Gaussian steerable filters. Veins appear as valleys (negative curvature).

```python
import math
import numpy as np
import scipy.ndimage as Image

def compute_curvature(image, sigma):
    # Build 2D Gaussian filter
    winsize = np.ceil(4 * sigma)
    window = np.arange(-winsize, winsize + 1)
    X, Y = np.meshgrid(window, window)
    G = 1.0 / (2 * math.pi * sigma ** 2)
    G *= np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))

    # First and second derivatives of G
    G1_0 = (-X / (sigma ** 2)) * G
    G2_0 = ((X ** 2 - sigma ** 2) / (sigma ** 4)) * G
    G1_90 = G1_0.T
    G2_90 = G2_0.T
    hxy = ((X * Y) / (sigma ** 8)) * G

    # Convolve image with derivative filters
    image_g1_0  = 0.1  * Image.convolve(image, G1_0,  mode='nearest')
    image_g2_0  = 10   * Image.convolve(image, G2_0,  mode='nearest')
    image_g1_90 = 0.1  * Image.convolve(image, G1_90, mode='nearest')
    image_g2_90 = 10   * Image.convolve(image, G2_90, mode='nearest')
    fxy         =        Image.convolve(image, hxy,   mode='nearest')

    # Diagonal directions (steerable filter combination)
    image_g1_45  = 0.5 * np.sqrt(2) * (image_g1_0 + image_g1_90)
    image_g1_m45 = 0.5 * np.sqrt(2) * (image_g1_0 - image_g1_90)
    image_g2_45  = 0.5 * image_g2_0 + fxy + 0.5 * image_g2_90
    image_g2_m45 = 0.5 * image_g2_0 - fxy + 0.5 * image_g2_90

    # Kappa (curvature) formula: d²P/dz² / (1 + (dP/dz)²)^(3/2)
    return np.dstack([
        (image_g2_0   / ((1 + image_g1_0   ** 2) ** 1.5)),
        (image_g2_90  / ((1 + image_g1_90  ** 2) ** 1.5)),
        (image_g2_45  / ((1 + image_g1_45  ** 2) ** 1.5)),
        (image_g2_m45 / ((1 + image_g1_m45 ** 2) ** 1.5)),
    ])
```

### 3.5 Create `PreProcessing_FeatureExtraction/detect_vein_center_assign_score.py`

**Purpose (Steps 1-2, 1-3, 1-4):** Find vein centers and assign a probabilistic score based on curvature width and depth.

```python
import numpy

def profile_score_1d(profile_1d):
    """Assign score to vein centers in a 1D profile."""
    threshold_1d = (profile_1d > 0).astype(int)
    diff = threshold_1d[1:] - threshold_1d[:-1]
    starts = numpy.argwhere(diff > 0)
    starts += 1
    ends = numpy.argwhere(diff < 0)
    ends += 1
    if threshold_1d[0]:
        starts = numpy.insert(starts, 0, 0)
    if threshold_1d[-1]:
        ends = numpy.append(ends, len(profile_1d))

    score_1d = numpy.zeros_like(profile_1d)
    if starts.size == 0 and ends.size == 0:
        return score_1d
    for start, end in zip(starts, ends):
        maximum = numpy.argmax(profile_1d[int(start):int(end)])
        score_1d[start + maximum] = profile_1d[start + maximum] * (end - start)
    return score_1d


def compute_vein_score(k):
    """Compute vein center scores in all 4 directions from kappa."""
    score = numpy.zeros(k.shape, dtype='float64')

    # Horizontal
    for index in range(k.shape[0]):
        score[index, :, 0] += profile_score_1d(k[index, :, 0])

    # Vertical
    for index in range(k.shape[1]):
        score[:, index, 1] += profile_score_1d(k[:, index, 1])

    # +45 degrees
    curve = k[:, :, 2]
    i, j = numpy.indices(curve.shape)
    for index in range(-curve.shape[0] + 1, curve.shape[1]):
        score[i == (j - index), 2] += profile_score_1d(curve.diagonal(index))

    # -45 degrees
    curve = numpy.flipud(k[:, :, 3])
    Vud = numpy.flipud(score)
    for index in reversed(range(curve.shape[1] - 1, -curve.shape[0], -1)):
        Vud[i == (j - index), 3] += profile_score_1d(curve.diagonal(index))

    return score
```

### 3.6 Create `PreProcessing_FeatureExtraction/connect_center.py`

**Purpose (Step 2):** Connect vein centers to form a continuous vein pattern and remove noise.

```python
import numpy

def connect_profile_1d(vein_prob_1d):
    return numpy.amin([
        numpy.amax([vein_prob_1d[3:-1], vein_prob_1d[4:]], axis=0),
        numpy.amax([vein_prob_1d[1:-3], vein_prob_1d[:-4]], axis=0)
    ], axis=0)


def connect_centres(vein_score):
    """Connect vein centres in all 4 directions."""
    connected_center = numpy.zeros(vein_score.shape, dtype='float64')
    temp = (vein_score[..., 0] + vein_score[..., 1] +
            vein_score[..., 2] + vein_score[..., 3])
    vein_score = temp

    # Horizontal
    for index in range(vein_score.shape[0]):
        connected_center[index, 2:-2, 0] = connect_profile_1d(vein_score[index, :])

    # Vertical
    for index in range(vein_score.shape[1]):
        connected_center[2:-2, index, 1] = connect_profile_1d(vein_score[:, index])

    i, j = numpy.indices(vein_score.shape)
    border = numpy.zeros((2,), dtype='float64')

    # +45 degrees
    for index in range(-vein_score.shape[0] + 5, vein_score.shape[1] - 4):
        connected_center[:, :, 2][i == (j - index)] = numpy.hstack(
            [border, connect_profile_1d(vein_score.diagonal(index)), border])

    # -45 degrees
    Vud = numpy.flipud(vein_score)
    Cdud = numpy.flipud(connected_center[:, :, 3])
    for index in reversed(range(vein_score.shape[1] - 5, -vein_score.shape[0] + 4, -1)):
        Cdud[:, :][i == (j - index)] = numpy.hstack(
            [border, connect_profile_1d(Vud.diagonal(index)), border])

    return connected_center
```

### 3.7 Create `PreProcessing_FeatureExtraction/label.py`

**Purpose (Step 3):** Binarize the connected vein pattern using median thresholding.

```python
import numpy

def binaries(G):
    """Threshold vein pattern using median value."""
    median = numpy.median(G[G > 0])
    Gbool = G > median
    return Gbool.astype(numpy.float64)
```

### 3.8 Create `PreProcessing_FeatureExtraction/extract_feature.py`

**Purpose:** Combines all 3 steps above into one `vein_pattern()` function.

```python
import numpy as np
from PreProcessing_FeatureExtraction.connect_center import connect_centres
from PreProcessing_FeatureExtraction.detect_vein_center_assign_score import compute_vein_score
from PreProcessing_FeatureExtraction.label import binaries
from PreProcessing_FeatureExtraction.normalize import normalize_data
from PreProcessing_FeatureExtraction.preprocessing import remove_hair
from PreProcessing_FeatureExtraction.profile_curvature import compute_curvature


def vein_pattern(image, kernel_size, sigma):
    """
    Full Maximum Curvature pipeline:
    1. Remove hair noise
    2. Normalize image
    3. Compute curvature (kappa) in 4 directions
    4. Assign probabilistic vein scores
    5. Connect vein centers
    6. Binarize using median threshold
    7. Multiply with original image for thick veins
    """
    data = np.asarray(image, dtype=float)
    filter_data = remove_hair(data, kernel_size)
    preprocessed_data = normalize_data(filter_data, 0, 255)
    kappa = compute_curvature(preprocessed_data, sigma=sigma)
    score = compute_vein_score(kappa)
    conect_score = connect_centres(score)
    threshold = binaries(np.amax(conect_score, axis=2))
    vein = np.multiply(image, threshold, dtype=float)
    return vein
```

---

## 📁 STEP 4: Build `cancellability/` Module

This module **protects the biometric template** using a secret key — so if the key is stolen, you can revoke and reissue a new key.

### 4.1 Create `cancellability/__init__.py`
```python
# cancellability package
```

### 4.2 Create `cancellability/feature_rdmtransform.py`

**Purpose:** Random Distance Transform — the main cancellable transform. It computes the Euclidean distance between the feature vector and the key vector in 2D space, then applies a median filter.

```python
import numpy as np
import scipy.ndimage

def rescale(fvs, a, b):
    m = fvs.min()
    M = fvs.max()
    if M - m == 0:
        return np.full_like(fvs, a)
    return (b - a) * (fvs - m) / (M - m) + a


def transformMeximumCurvatureRDM(img, Key1, Key2):
    """
    Apply Random Distance Transform to the image feature vector.
    
    img   : 100x100 grayscale image
    Key1  : secret key 1 (10000x1)
    Key2  : secret key 2 (10000x1)
    
    Returns:
        TfM : transformed (cancellable) feature vector
        fv  : original scaled feature vector
    """
    # Flatten image to 1D (10000,1) and scale to [1, 100]
    fvs = np.array(img).reshape(10000, 1)
    fvs = rescale(fvs, 1, 100)
    fv = fvs

    # Split feature vector into X1 (first half) and Y1 (second half)
    fvsParts = np.split(fvs, 2)
    X1 = fvsParts[0]   # shape (5000, 1)
    Y1 = fvsParts[1]   # shape (5000, 1)

    # Split keys
    Key1 = np.array(Key1)
    X2 = np.split(Key1, 2)[0]   # first half of Key1

    Key2 = np.array(Key2)
    Y2 = np.split(Key2, 2)[0]   # first half of Key2

    # Random Distance Transform: Euclidean distance in 2D
    # Tf[i] = sqrt((X2[i] - X1[i])^2 + (Y2[i] - Y1[i])^2)
    Tf = np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2)

    # Apply median filter to smooth the result
    TfM = scipy.ndimage.median_filter(
        Tf, size=5, mode='reflect'
    )

    fv = np.array(fv).reshape(10000, 1)
    return TfM, fv
```

### 4.3 Create `cancellability/feature_xortransform.py`

**Purpose:** XOR Transform — alternative cancellable method using bitwise XOR with a random key matrix.

```python
import cv2
import numpy as np
import scipy.ndimage

def transform_RG_XOR(image, key):
    """XOR the image with a random key, then apply median filter."""
    RG = key
    fv = cv2.bitwise_xor(image.astype(np.uint8), RG.astype(np.uint8))
    fvs = scipy.ndimage.median_filter(
        fv, size=(5, 5), mode='reflect'
    )
    return fvs
```

---

## 📁 STEP 5: Build `classification/` Module

This module trains a **Kernel Fisher Analysis (KFA)** model and uses it for authentication.

### 5.1 Create `classification/__init__.py`
```python
# classification package
```

### 5.2 Create `classification/Normo.py`

**Purpose:** Normalize columns of a matrix to unit L2 norm.

```python
import numpy as np
from sklearn.preprocessing import normalize

def normc(Mat):
    """Normalize columns of matrix to unit L2 norm."""
    assert len(Mat.shape) == 2
    if Mat.dtype != float:
        Mat = np.asarray(Mat, dtype=float)
    return normalize(Mat, norm='l2', axis=0)
```

### 5.3 Create `classification/KernelMatrix.py`

**Purpose:** Compute kernel matrices (polynomial, fractional power polynomial, or sigmoid) between two data matrices.

```python
import numpy as np

def compute_kernel_matrix_PhD(X, Y, kernel_type, kernel_args):
    """
    Compute kernel matrix between X and Y.
    
    Supported kernels:
      'poly' : k(x,y) = (x'y + c)^d         [kernel_args = [c, d]]
      'fpp'  : k(x,y) = sign(x'y+c)|x'y+c|^p [kernel_args = [c, p]]
      'tanh' : k(x,y) = tanh(x'y + c)         [kernel_args = c]
    """
    X = np.array(X)
    Y = np.array(Y)

    if kernel_type == 'poly':
        kermat = (np.dot(X.T, Y) + kernel_args[0]) ** kernel_args[1]
    elif kernel_type == 'fpp':
        dot = np.dot(X.T, Y) + kernel_args[0]
        kermat = np.sign(dot) * (abs(dot) ** kernel_args[1])
    elif kernel_type == 'tanh':
        kermat = np.tanh(np.dot(X.T, Y) + kernel_args)

    return kermat
```

### 5.4 Create `classification/KFA.py`

**Purpose:** Kernel Fisher Analysis — finds a low-dimensional subspace that maximizes between-class separation while minimizing within-class scatter (kernel version of LDA).

```python
import numpy as np
from classification import KernelMatrix
from classification import Normo


def find(ele, arr):
    return [i for i in range(len(arr)) if ele == arr[i]]


def return_W(W, ids):
    """Build within-class weight matrix W."""
    id_unique = np.unique(ids)
    for uid in id_unique:
        ind = find(uid, ids)
        x, y = np.meshgrid(ind, ind)
        elem_val = 1 / len(ind)
        for p in range(len(x)):
            for q in range(len(x)):
                W[x[p][q]][y[p][q]] = elem_val
    return W


class Model(object):
    def __init__(self):
        pass


def perform_kfa_PhD(X, ids, kernel_type, n):
    """
    Train a Kernel Fisher Analysis model.
    
    X           : (features x samples) training data matrix
    ids         : list of class labels for each sample
    kernel_type : 'poly', 'fpp', or 'tanh'
    n           : number of KFA components to retain
    
    Returns model with:
      model.K     - training kernel matrix
      model.W     - transformation matrix (eigenvectors)
      model.train - projected training features
      model.X     - original training data
      ... and more
    """
    model = Model()

    # Set kernel parameters
    if kernel_type == 'poly':
        kernel_args = [0, 2]
    elif kernel_type == 'fpp':
        kernel_args = [0, .8]
    elif kernel_type == 'tanh':
        kernel_args = 0

    [a, b] = X.shape

    # Step 1: Compute training kernel matrix
    K = KernelMatrix.compute_kernel_matrix_PhD(X, X, kernel_type, kernel_args)
    model.K = K

    # Step 2: Center the kernel matrix
    J = np.ones((b, b)) / b
    Kc = K - np.dot(J, K) - np.dot(K, J) + np.dot(np.dot(J, K), J)
    model.J = J

    # Step 3: Build within-class weight matrix W
    W = np.zeros((b, b))
    W = return_W(W, ids)

    # Step 4: Tikhonov regularization + solve eigenproblem
    epsi = 1e-10 * np.min(np.dot(Kc, Kc))
    Crit = np.linalg.solve(
        (np.dot(Kc, Kc) + epsi * np.identity(b)),
        (np.dot(np.dot(Kc, W), Kc))
    )

    # Step 5: SVD to get eigenvectors
    [U, V, L] = np.linalg.svd(Crit)
    Alpha = Normo.normc(U[:, 0:n])

    # Step 6: Normalize Alpha to unit length in feature space F
    R = np.dot(np.dot(Alpha.T, Kc), Alpha)
    norms = np.real(np.diag(R))
    for i in range(n):
        Alpha[:, i] = Alpha[:, i] / np.sqrt(norms[i])

    # Build model
    model.W = Alpha
    model.dim = n
    V = V.reshape(len(V), 1)
    model.eigs = V
    model.typ = kernel_type
    model.args = kernel_args
    model.X = X

    # Project training data into KFA subspace
    model.train = np.dot(Alpha.T, Kc)

    return model
```

### 5.5 Create `classification/Projection.py`

**Purpose:** Project a new (test) image into the trained KFA subspace for comparison.

```python
import numpy as np
from classification import KernelMatrix


def nonlinear_subspace_projection_PhD(X, model):
    """
    Project test data X into the KFA subspace defined by model.
    
    X     : (features x 1) test sample
    model : trained KFA model
    
    Returns feat : projected feature vector
    """
    [a, b] = X.shape
    [c, d] = model.J.shape

    # Compute test kernel matrix (between test and training data)
    K = KernelMatrix.compute_kernel_matrix_PhD(X, model.X, model.typ, model.args)

    # Center the test kernel matrix
    Jt = np.ones((b, 1)) / c
    J  = np.ones((c, 1))
    K = (K
         - np.dot(np.dot(Jt, J.T), model.K)
         - np.dot(K, (1/c) * np.dot(J, J.T))
         + np.dot(Jt, np.dot(J.T, np.dot(model.K, (1/c) * np.dot(J, J.T)))))

    K = K.T  # transpose for correct orientation

    # Project into subspace: feat = W^T * K
    feat = np.dot(model.W.T, K)
    return feat
```

---

## 📁 STEP 6: Build `create_csv/` Module

### 6.1 Create `create_csv/__init__.py`
```python
# create_csv package
```

### 6.2 Create `create_csv/key_01.py`

**Purpose:** Generate a random key matrix (85 subjects × 10000 values) and save as `Key01.csv`.

```python
import numpy as np
import pandas as pd

seed = 7
np.random.seed(seed)

# 85 subjects, image size 100x100 = 10000 values per subject
Key = np.random.uniform(1, 100, (85, 10000))
df = pd.DataFrame(Key)
df.to_csv('Key01.csv', index=False, header=False)
print('Key01.csv created successfully.')
```

### 6.3 Create `create_csv/key_02.py`

```python
import numpy as np
import pandas as pd

seed = 42
np.random.seed(seed)

Key = np.random.uniform(1, 100, (85, 10000))
df = pd.DataFrame(Key)
df.to_csv('Key02.csv', index=False, header=False)
print('Key02.csv created successfully.')
```

### 6.4 Create `create_csv/xor_key.py`

```python
import numpy as np
import pandas as pd

seed = 2
np.random.seed(seed)

# XOR key: integer values for bitwise XOR, shape (85, 100x100)
Key = np.random.randint(0, 256, (85, 10000))
df = pd.DataFrame(Key)
df.to_csv('Xor_Key.csv', index=False, header=False)
print('Xor_Key.csv created successfully.')
```

---

## 📁 STEP 7: Create `main.py`

**Purpose:** The main authentication pipeline — loads images, extracts features, trains KFA, and authenticates a test image.

```python
print('\nLoad images and compute features. This may take a while.')

import csv
import os
import warnings
import cv2
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import normalize

from cancellability.feature_rdmtransform import transformMeximumCurvatureRDM as MCRG
from classification import KFA, Projection

warnings.filterwarnings("ignore")

# --- Config ---
input_folder_path = './sample dataset/veinpattern/'
k, N = 100, 100

# --- Load Key Matrices ---
def load_key(path):
    mat = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            mat.append([float(x) for x in row])
    return normalize(np.array(mat))

KeyMat  = load_key('./create_csv/Key01.csv')
KeyMat1 = load_key('./create_csv/Key02.csv')
print('Key Matrices Loaded')

# --- Load Dataset ---
subjects = os.listdir(input_folder_path)
R = len(subjects)
S_Path = [os.path.join(input_folder_path, s) for s in subjects]
sub_folder_content = [os.listdir(p) for p in S_Path]
C = len(sub_folder_content[0])

# --- Feature Extraction (Training) ---
ID = []
dataMatrixRG = None

for x in range(R):
    print(f'Subject: {x + 1}')
    for y in range(C):
        ID.append(x)
        ImgPath = os.path.join(S_Path[x], sub_folder_content[x][y])
        Img = cv2.imread(ImgPath, 0)
        Img = np.asarray(Img, dtype=np.float64)
        Img = cv2.resize(Img, (k, N), interpolation=cv2.INTER_CUBIC)

        Key  = KeyMat[x].reshape(k * N, 1)
        Key1 = KeyMat1[x].reshape(k * N, 1)

        transformedFV, _ = MCRG(Img, Key, Key1)

        if dataMatrixRG is None:
            dataMatrixRG = np.column_stack((transformedFV)).T
        else:
            dataMatrixRG = np.column_stack((dataMatrixRG, transformedFV))

print('Feature Extraction completed')
train_data = np.nan_to_num(dataMatrixRG)
ids_train = ID

# --- Train KFA Model ---
model = KFA.perform_kfa_PhD(train_data, ids_train, 'fpp', len(ids_train))

# --- Test: Authenticate subject 0, sample index 5 ---
ids_test = 0
ImgPath = os.path.join(S_Path[ids_test], sub_folder_content[ids_test][5])
Img2 = cv2.imread(ImgPath, 0)
Img2 = np.asarray(Img2, dtype=float)
Img2 = cv2.resize(Img2, (k, N), interpolation=cv2.INTER_CUBIC)

Key  = KeyMat[ids_test].reshape(k * N, 1)
Key1 = KeyMat1[ids_test].reshape(k * N, 1)
transformedFV1, _ = MCRG(Img2, Key, Key1)

test_data = np.array(transformedFV1).T
testfeature = Projection.nonlinear_subspace_projection_PhD(test_data, model)

# --- Compare distances ---
dt = model.train
v = np.array(testfeature).flatten()
THRESHOLD = 100.0

for x in range(len(ids_train)):
    u = np.array(dt[:, x]).flatten()
    di = distance.euclidean(v, u)
    if di <= THRESHOLD:
        print(f'Access GRANTED for subject {ids_train[x] + 1}')
    else:
        print(f'Access DENIED for subject {ids_train[x] + 1}')
```

---

## 📁 STEP 8: Create `app.py` (Streamlit Web UI)

**Purpose:** A web interface where users can upload a hand image and verify their identity.

> Copy the full `app.py` from the original project — it uses all the modules above.

Key flow in `app.py`:
1. `load_keys()` → loads Key01.csv and Key02.csv
2. `train_model()` → loops all subjects, calls `MCRG()`, builds `train_data`, trains `KFA`
3. User uploads image → `MCRG()` transforms it → `Projection` projects it → Euclidean distance → GRANT/DENY

---

## 🚀 STEP 9: Run the Project

### Run authentication pipeline:
```
python main.py
```

### Run web UI:
```
streamlit run app.py
```
or double-click `run_app.bat`

---

## 🔑 KEY CONCEPTS SUMMARY

| Concept | What it does | File |
|---|---|---|
| **Hair Removal** | Mexican Hat kernel convolution removes hair noise | `preprocessing.py` |
| **Profile Curvature** | Gaussian steerable filter finds vein valleys (kappa) | `profile_curvature.py` |
| **Vein Scoring** | Assigns probability score to vein center locations | `detect_vein_center_assign_score.py` |
| **Connect Centers** | Links vein centers for continuous pattern | `connect_center.py` |
| **Binarize** | Median threshold → binary vein mask | `label.py` |
| **RDM Transform** | Euclidean distance with secret key = cancellable template | `feature_rdmtransform.py` |
| **Kernel Matrix** | Measures similarity between samples in kernel space | `KernelMatrix.py` |
| **KFA** | Finds best subspace to separate identities | `KFA.py` |
| **Projection** | Projects test image into KFA subspace | `Projection.py` |
| **Distance Match** | Euclidean distance < threshold → GRANTED | `main.py` / `app.py` |

---

## ⚠️ IMPORTANT NOTES

1. **Dataset**: You need 850 infrared dorsal hand images (85 subjects × 10 images each) in `sample dataset/veinpattern/s01/`, `s02/`, etc.
2. **MexicanHatKernalData**: Copy the kernel image files from the original project into `PreProcessing_FeatureExtraction/MexicanHatKernalData/`.
3. **Key CSVs**: Run `key_01.py`, `key_02.py`, `xor_key.py` inside `create_csv/` to generate the key files before running `main.py`.
4. **Image size**: All images are resized to **100×100** pixels internally.
5. **Threshold**: The authentication threshold is **100.0** (Euclidean distance). Tune this based on your dataset.
