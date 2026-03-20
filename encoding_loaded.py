from pathlib import Path
import pickle, numpy as np
from .config import ENCODINGS_PKL

def load_all_encodings():
    if not ENCODINGS_PKL.exists():
        print(f"❌ Không tìm thấy: {ENCODINGS_PKL}")
        return np.empty((0,128), dtype=np.float32), []
    data = pickle.load(open(ENCODINGS_PKL, 'rb'))
    return np.array(data['embeddings']), data['names']
