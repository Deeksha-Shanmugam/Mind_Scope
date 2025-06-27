import joblib
import numpy as np
from scipy.sparse import hstack
from app.utils.preprocess import clean_text
import os

# Check if model files exist
model_files = {
    "vectorizer": "models/tfidf_vectorizer.pkl",
    "glove_embeddings": "models/glove_embeddings.pkl", 
    "classifier": "models/final_model.pkl",
    "label_encoder": "models/label_encoder.pkl"
}

# Load models with error handling
try:
    vectorizer = joblib.load(model_files["vectorizer"])
    print("✓ TF-IDF vectorizer loaded")
except Exception as e:
    print(f"✗ Error loading vectorizer: {e}")
    raise

try:
    glove_embeddings = joblib.load(model_files["glove_embeddings"])
    print("✓ GloVe embeddings loaded")
except Exception as e:
    print(f"✗ Error loading GloVe embeddings: {e}")
    raise

try:
    classifier = joblib.load(model_files["classifier"])
    print("✓ Classifier loaded")
except Exception as e:
    print(f"✗ Error loading classifier: {e}")
    raise

try:
    label_encoder = joblib.load(model_files["label_encoder"])
    print("✓ Label encoder loaded")
except Exception as e:
    print(f"✗ Error loading label encoder: {e}")
    raise

embedding_dim = 100  # Set this to your GloVe dimension

def get_average_vector(text, embeddings_index, dim):
    """Get average word vector for text"""
    try:
        words = text.split()
        vectors = [embeddings_index[w] for w in words if w in embeddings_index]
        if len(vectors) == 0:
            return np.zeros(dim)
        return np.mean(vectors, axis=0)
    except Exception as e:
        print(f"Error in get_average_vector: {e}")
        return np.zeros(dim)

def predict_label(text):
    """Predict label for a single text"""
    try:
        print(f"Predicting for text: {text[:50]}...")
        
        # Clean text
        clean = clean_text(text)
        print(f"Cleaned text: {clean[:50]}...")
        
        # Get TF-IDF vector
        tfidf_vec = vectorizer.transform([clean])
        print(f"TF-IDF vector shape: {tfidf_vec.shape}")
        
        # Get GloVe vector
        glove_vec = np.array([get_average_vector(clean, glove_embeddings, embedding_dim)])
        print(f"GloVe vector shape: {glove_vec.shape}")
        
        # Combine vectors
        combined_vec = hstack([tfidf_vec, glove_vec])
        print(f"Combined vector shape: {combined_vec.shape}")
        
        # Make prediction
        pred = classifier.predict(combined_vec)
        print(f"Raw prediction: {pred}")
        
        # Decode label
        label = label_encoder.inverse_transform(pred)[0]
        print(f"Final label: {label}")
        
        return label
        
    except Exception as e:
        print(f"Error in predict_label: {e}")
        import traceback
        print(traceback.format_exc())
        return "Error in prediction"

def predict_labels(texts):
    """Predict labels for multiple texts"""
    try:
        print(f"Predicting for {len(texts)} texts...")
        
        # Clean all texts
        cleaned = [clean_text(text) for text in texts]
        print(f"Cleaned {len(cleaned)} texts")
        
        # Get TF-IDF vectors
        tfidf_vecs = vectorizer.transform(cleaned)
        print(f"TF-IDF vectors shape: {tfidf_vecs.shape}")
        
        # Get GloVe vectors
        glove_vecs = np.array([get_average_vector(text, glove_embeddings, embedding_dim) for text in cleaned])
        print(f"GloVe vectors shape: {glove_vecs.shape}")
        
        # Combine vectors
        combined_vecs = hstack([tfidf_vecs, glove_vecs])
        print(f"Combined vectors shape: {combined_vecs.shape}")
        
        # Make predictions
        preds = classifier.predict(combined_vecs)
        print(f"Raw predictions shape: {preds.shape}")
        
        # Decode labels
        labels = label_encoder.inverse_transform(preds)
        print(f"Decoded {len(labels)} labels")
        
        return labels.tolist()
        
    except Exception as e:
        print(f"Error in predict_labels: {e}")
        import traceback
        print(traceback.format_exc())
        return ["Error in prediction"] * len(texts)