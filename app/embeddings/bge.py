from sentence_transformers import SentenceTransformer

# Load the model globally so it only initializes once when the app starts.
# BAAI/bge-small-en-v1.5 is extremely fast and perfect for local development.
try:
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
except Exception as e:
    print(f"Failed to load embedding model: {e}")
    model = None

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Converts a list of text strings into vector embeddings.
    """
    if not model:
        raise RuntimeError("Embedding model is not loaded.")
        
    # normalize_embeddings=True is recommended for BGE models to use Cosine Similarity
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()