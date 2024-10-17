import os
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
from skimage.color import rgb2lab
from skimage.filters import sobel
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import silhouette_score
import requests
import zipfile
import io

# Fungsi untuk mengunduh dan mengekstrak dataset dari GitHub
def download_and_extract_dataset(url, extract_to="data_training"):
    """Download and extract dataset ZIP from GitHub."""
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Gagal mengunduh dataset dari {url}")

    # Simpan ZIP file sementara
    zip_path = os.path.join(extract_to, "dataset.zip")
    with open(zip_path, "wb") as f:
        f.write(response.content)

    # Ekstrak ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    os.remove(zip_path)  # Hapus ZIP setelah ekstraksi

# Fungsi untuk mengunduh gambar tunggal dari GitHub
def download_image_from_github(url):
    """Download a single image from GitHub using raw URL."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return np.array(image.resize((256, 256)))
    except UnidentifiedImageError:
        raise Exception(f"File bukan gambar atau format tidak didukung: {url}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Gagal mengunduh gambar dari {url}. Error: {e}")

class KMeansModel:
    def __init__(self):
        self.centroids = None
        self.cluster_size = None

    def extract_features(self, image: np.ndarray):
        """Ekstrak fitur dari gambar."""
        image = cv2.fastNlMeansDenoisingColored(image, None, 30, 30, 7, 21)
        lab_image = rgb2lab(image)
        h, w, _ = image.shape

        # Flatten fitur (Lab channels + koordinat)
        pixels = lab_image.reshape(-1, 3)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        coords = np.column_stack([x.flatten() / w, y.flatten() / h])
        features = np.hstack([pixels, coords])
        return features

    def fit(self, images, cluster_size, max_iters=100):
        """Melatih model K-Means dengan gambar."""
        self.cluster_size = cluster_size
        all_features = np.vstack([self.extract_features(img) for img in images])
        num_samples = all_features.shape[0]

        # Inisialisasi centroid secara acak
        random_indices = np.random.choice(num_samples, cluster_size, replace=False)
        self.centroids = all_features[random_indices]

        # Iterasi untuk update centroid
        for _ in range(max_iters):
            labels = self._assign_clusters(all_features)
            new_centroids = np.array([all_features[labels == i].mean(axis=0) for i in range(cluster_size)])
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids

    def _assign_clusters(self, features):
        distances = np.linalg.norm(features[:, None] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, image: np.ndarray):
        """Lakukan prediksi cluster pada gambar."""
        if self.centroids is None:
            raise ValueError("Model belum dilatih. Harap latih model terlebih dahulu.")
        features = self.extract_features(image)
        labels = self._assign_clusters(features)
        return labels.reshape(image.shape[:2]), features

    def visualize(self, image: np.ndarray, labels: np.ndarray, k: int, silhouette_avg: float):
        """Visualisasi gambar asli dan hasil clustering."""
        h, w = labels.shape
        palette = np.array([
            [139, 69, 19], [34, 139, 34], [107, 142, 35], 
            [0, 255, 127], [210, 105, 30], [85, 107, 47], 
            [128, 128, 128], [255, 215, 0], [47, 79, 79], 
            [139, 0, 0]
        ], dtype=np.uint8)

        assert self.cluster_size <= len(palette), "Palet warna tidak cukup untuk jumlah cluster."
        segmented_image = palette[labels]

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar Asli", use_column_width=True)
        with col2:
            st.image(segmented_image, caption=f"Hasil Clustering dengan K={k}", use_column_width=True)

        st.write(f"Silhouette Score: {silhouette_avg:.4f}")

def load_training_data(folder_path):
    """Memuat semua gambar dari folder untuk training."""
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(("jpg", "jpeg", "png")):
            img = Image.open(os.path.join(folder_path, filename)).resize((256, 256))
            images.append(np.array(img))
    return images

# Inisialisasi Streamlit dan Model
st.title("K-Means Clustering pada Citra Udara")
st.write("Disusun oleh Aliya Rahmania (140810210005), Adinda Salsabila (140810210017), Devi Humaira (140810220015) untuk UTS Data Mining 2024")


if "model" not in st.session_state:
    st.session_state.model = KMeansModel()

if "training_done" not in st.session_state:
    st.session_state.training_done = False

DATASET_URL = "https://github.com/adindaaasals/clustering_CitraUdara/archive/refs/heads/main.zip"

if not st.session_state.training_done:
    st.write("Mengunduh dan mengekstrak dataset, harap tunggu...")
    try:
        download_and_extract_dataset(DATASET_URL)
        training_images = load_training_data("data_training/clustering_CitraUdara-main/data_training")
        k_value = st.number_input("Pilih jumlah cluster (K) untuk training", min_value=2, max_value=10, value=3, step=1)
        st.session_state.model.fit(training_images, k_value)
        st.session_state.training_done = True
        st.success("Training selesai! Anda dapat menguji gambar sekarang.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

uploaded_files = st.file_uploader("Upload Gambar untuk Pengujian (maks 5 gambar)", 
                                  type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.button("Uji Model"):
    if uploaded_files and st.session_state.training_done:
        for uploaded_file in uploaded_files[:5]:
            test_image = Image.open(uploaded_file).resize((256, 256))
            labels, features = st.session_state.model.predict(np.array(test_image))

            silhouette_avg = silhouette_score(features, labels.flatten())
            st.session_state.model.visualize(np.array(test_image), labels, st.session_state.model.cluster_size, silhouette_avg)
    elif not st.session_state.training_done:
        st.warning("Model belum dilatih. Harap tunggu proses training selesai.")
    else:
        st.warning("Silakan upload gambar.")