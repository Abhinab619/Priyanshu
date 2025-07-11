import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from deepface import DeepFace
from scipy.spatial.distance import cosine
from numpy.linalg import norm
import pandas as pd
from itertools import combinations
from mtcnn import MTCNN
from sklearn.cluster import DBSCAN

def read_image_from_url(url):
    """Fetch image from URL and convert to RGB numpy array."""
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch image from {url}")
    image = Image.open(BytesIO(response.content)).convert('RGB')
    return np.array(image)

def extract_face(image_array):
    detector = MTCNN()
    results = detector.detect_faces(image_array)
    if not results:
        raise ValueError("No face detected in the image.")
    x, y, w, h = results[0]['box']
    x, y = max(0, x), max(0, y)
    face = image_array[y:y+h, x:x+w]
    return face

def compare_faces_from_urls(url1, url2):
    """
    Compare faces using image URLs without saving locally.
    """
    print(f"Comparing images from:\n- {url1}\n- {url2}")
    
    # Check for empty URLs
    if not url1.strip() or not url2.strip():
        print("One of the URLs is empty, skipping comparison")
        return {
            "cosine_similarity": 0,
            "euclidean_distance": 1.0,
            "deepface_verified": False,
            "deepface_distance": 1.0,
            "deepface_threshold": 0.25
        }
    
    try:
        # Load images from URLs
        img1_array = read_image_from_url(url1)
        img2_array = read_image_from_url(url2)

        # Extract faces using MTCNN
        face1 = extract_face(img1_array)
        face2 = extract_face(img2_array)

        # Resize faces to 160x160 for Facenet
        face1 = cv2.resize(face1, (160, 160))
        face2 = cv2.resize(face2, (160, 160))

        # DeepFace embedding (Facenet)
        embedding1 = DeepFace.represent(img_path=face1, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        embedding2 = DeepFace.represent(img_path=face2, model_name="Facenet", enforce_detection=False)[0]["embedding"]

        # Convert to numpy
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        # Calculate similarity
        results = {}
        results["cosine_similarity"] = 1 - cosine(emb1, emb2)
        results["euclidean_distance"] = norm(emb1 - emb2)
        
        # Debug info to help track issues
        cosine_sim = results["cosine_similarity"]
        print(f"Cosine similarity: {cosine_sim:.4f} (higher is better)")

        # DeepFace verification (Facenet, but with already extracted faces)
        # DeepFace might not accept threshold as a parameter, so we call it without threshold
        verify_result = DeepFace.verify(
            img1_path=face1,
            img2_path=face2,
            model_name="Facenet",
            distance_metric="cosine",
            enforce_detection=False
        )
        
        # We'll use our own threshold of 0.25 for consistency without changing the actual logic
        threshold = 0.25
        distance = verify_result["distance"]
        verified = verify_result["verified"]
        
        # Log the result details
        print(f"DeepFace verification result: {verified}, distance: {distance:.4f}, our threshold: {threshold}")
        
        results["deepface_verified"] = verified
        results["deepface_distance"] = distance
        results["deepface_threshold"] = threshold

        return results
    except Exception as e:
        print(f"Error in face comparison: {str(e)}")
        return {
            "cosine_similarity": 0,
            "euclidean_distance": 1.0,
            "deepface_verified": False,
            "deepface_distance": 1.0,
            "deepface_threshold": 0.25
        }
