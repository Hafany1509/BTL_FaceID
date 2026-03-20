from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR   = ROOT / "dataset"
ENCODINGS_PKL = ROOT / "encodings" / "encodings_dlib20.pkl"
MODELS_DIR    = ROOT / "models"

MYSQL = {
    "host": "127.0.0.1",   # (ưu tiên 127.0.0.1 thay vì localhost)
    "port": 3306,
    "user": "root",
    "password": "123456",
    "database": "dulieu_app",
}


WORK_START_HOUR = 8  # giờ bắt đầu làm việc
