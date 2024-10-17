import os
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
from skimage.color import rgb2lab
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import silhouette_score
import requests
import zipfile
import io

st.set_page_config(layout="wide")

# CSS untuk styling halaman
def set_custom_css():
    st.markdown(
        """
        <style>
        .stApp { background-color: #808080; }
        .main { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .centered-title { display: flex; justify-content: center; align-items: center; 
                          height: 100px; font-size: 48px; font-weight: bold; 
                          text-align: center; color: #FFFFFF; }
        .subtitle { text-align: center; font-size: 18px; margin-bottom: 20px; color: #FFFFFF; }
        </style>
        """, unsafe_allow_html=True,
    )

set_custom_css()

st.markdown("<div class='centered-title'>K-Means Clustering pada Citra Udara</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Disusun oleh Aliya Rahmania (140810210005), Adinda Salsabila (140810210017), Devi Humaira (140810220015) untuk UTS Data Mining 2024</div>", 
    unsafe_allow_html=True
)

class KMeansModel:
    def __init__(self):
        self.centroids = {}  # Menyimpan centroid untuk setiap K
        self.max_k = 5

    def extract_features(self, image: np.ndarray):
        """Ekstrak fitur dari gambar."""
        image = cv2.fastNlMeansDenoisingColored(image, None, 30, 30, 7, 21)
        lab_image = rgb2lab(image)
        h, w, _ = image.shape
        pixels = lab_image.reshape(-1, 3)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        coords = np.column_stack([x.flatten() / w, y.flatten() / h])
        features = np.hstack([pixels, coords])
        return features

    def fit(self, images):
        """Latih model dengan K dari 2 hingga 5 dan simpan centroidnya."""
        all_features = np.vstack([self.extract_features(img) for img in images])

        for k in range(2, self.max_k + 1):
            random_indices = np.random.choice(all_features.shape[0], k, replace=False)
            centroids = all_features[random_indices]

            for _ in range(100):  # Maksimal 100 iterasi
                labels = self._assign_clusters(all_features, centroids)
                new_centroids = np.array([all_features[labels == i].mean(axis=0) for i in range(k)])
                if np.all(new_centroids == centroids):
                    break
                centroids = new_centroids

            self.centroids[k] = centroids  # Simpan centroid untuk K

    def _assign_clusters(self, features, centroids):
        distances = np.linalg.norm(features[:, None] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, image: np.ndarray, k: int):
        """Lakukan prediksi cluster pada gambar."""
        if k not in self.centroids:
            raise ValueError(f"Model belum dilatih untuk K={k}.")
        centroids = self.centroids[k]
        features = self.extract_features(image)
        labels = self._assign_clusters(features, centroids)
        return labels.reshape(image.shape[:2]), features

    def visualize(self, image: np.ndarray, labels: np.ndarray, k: int, silhouette_avg: float):
        """Visualisasi gambar asli dan hasil clustering."""
        palette = np.array([
            [139, 69, 19], [34, 139, 34], [107, 142, 35], 
            [0, 255, 127], [210, 105, 30], [85, 107, 47], 
            [128, 128, 128], [255, 215, 0], [47, 79, 79], 
            [139, 0, 0]
        ], dtype=np.uint8)

        segmented_image = palette[labels]

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar Asli", width=300)
        with col2:
            st.image(segmented_image, caption=f"Hasil Clustering dengan K={k}", width=300)

        # Print Silhouette Score
        st.write(f"**Silhouette Score: {silhouette_avg:.4f}**")

def load_training_data(folder_path):
    """Memuat semua gambar dari folder untuk training."""
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(("jpg", "jpeg", "png")):
            img = Image.open(os.path.join(folder_path, filename)).resize((256, 256))
            images.append(np.array(img))
    return images

if "model" not in st.session_state:
    st.session_state.model = KMeansModel()

if "training_done" not in st.session_state:
    st.session_state.training_done = False

DATASET_URL = "https://github.com/adindaaasals/clustering_CitraUdara/archive/refs/heads/main.zip"

if not st.session_state.training_done:
    st.write("Mengunduh dan mengekstrak dataset, harap tunggu...")
    try:
        response = requests.get(DATASET_URL)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall("data_training")
        training_images = load_training_data("data_training/clustering_CitraUdara-main/data_training")

        # Latih model untuk K=2 hingga K=5
        st.session_state.model.fit(training_images)
        st.session_state.training_done = True
        st.success("Training selesai! Anda dapat menguji gambar sekarang.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# Input nilai K untuk pengujian
k_value = st.number_input("Pilih jumlah cluster (K) untuk pengujian", min_value=2, max_value=5, value=3, step=1)

# Upload gambar untuk pengujian
uploaded_files = st.file_uploader("Upload Gambar untuk Pengujian (maks 5 gambar)", 
                                  type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Pengujian dan visualisasi hasil
if st.button("Uji Model"):
    if uploaded_files and st.session_state.training_done:
        for uploaded_file in uploaded_files[:5]:
            test_image = Image.open(uploaded_file).resize((256, 256))
            labels, features = st.session_state.model.predict(np.array(test_image), k_value)

            silhouette_avg = silhouette_score(features, labels.flatten())
            st.session_state.model.visualize(np.array(test_image), labels, k_value, silhouette_avg)
    elif not st.session_state.training_done:
        st.warning("Model belum dilatih. Harap tunggu proses training selesai.")
    else:
        st.warning("Silakan upload gambar untuk pengujian.")
