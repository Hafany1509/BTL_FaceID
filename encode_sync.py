import pickle, os
from pathlib import Path
import face_recognition

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
OUT_PKL = ROOT / "encodings" / "encodings_dlib20.pkl"

def build():
    encs, names = [], []
    for person in os.listdir(DATASET):
        pdir = DATASET / person
        if not pdir.is_dir(): continue
        for imgfile in pdir.glob("**/*.jpg"):
            img = face_recognition.load_image_file(imgfile)
            boxes = face_recognition.face_locations(img, model='hog')
            enc = face_recognition.face_encodings(img, boxes)
            if enc: encs.append(enc[0]); names.append(person)
    pickle.dump({"embeddings":encs, "names":names}, open(OUT_PKL, "wb"))
    print("✅ Encodings saved:", OUT_PKL)

if __name__ == "__main__":
    build()
