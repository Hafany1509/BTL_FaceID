# app/capture_faces.py
import cv2
import os
from pathlib import Path
import face_recognition
#cmake --version
from app.config import ROOT

DATASET_DIR = ROOT / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

class FaceCollector:
    """
    Thu ảnh khuôn mặt từ camera.
    - collect(label): chụp n ảnh vào dataset/<label>/... (chỉ lưu khi phát hiện đúng 1 khuôn mặt)
    - collect_one_temp(): chụp 1 ảnh tạm models/__temp_face.jpg
    """
    def __init__(self, camera_index=0, max_images=30):
        self.cam_idx = camera_index
        self.max_images = max_images

    def _open_cam(self):
        cap = cv2.VideoCapture(self.cam_idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass
        return cap

    def collect(self, label: str):
        save_dir = DATASET_DIR / label
        save_dir.mkdir(parents=True, exist_ok=True)

        cap = self._open_cam()
        count = 0
        print("➡ SPACE: chụp (lưu khi thấy ĐÚNG 1 mặt), q: thoát")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog", number_of_times_to_upsample=0)

            # Vẽ hộp nếu có 1 mặt
            disp = frame.copy()
            if len(boxes) == 1:
                t, r, b, l = boxes[0]
                cv2.rectangle(disp, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(disp, "1 face detected - press SPACE to capture",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            elif len(boxes) > 1:
                cv2.putText(disp, "Found >1 faces - move closer/single person",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(disp, "No face - adjust lighting/angle",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow("Capture faces", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == 32:  # SPACE
                if len(boxes) == 1:
                    count += 1
                    fn = save_dir / f"{count:03d}.jpg"
                    cv2.imwrite(str(fn), frame)
                    print(f"💾 {fn}")
                else:
                    print("⚠️ Không lưu: cần đúng 1 khuôn mặt trong khung!")
                if count >= self.max_images:
                    break

        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Đã lưu {count} ảnh vào {save_dir}")

    def collect_one_temp(self) -> str:
        """
        Chụp 1 ảnh và lưu tạm ở models/__temp_face.jpg, trả về đường dẫn ảnh.
        Dùng để encode và kiểm tra trùng khuôn mặt trước khi thêm NV.
        """
        temp_path = ROOT / "models" / "__temp_face.jpg"
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        cap = self._open_cam()
        print("➡ SPACE: chụp 1 ảnh kiểm tra (cần đúng 1 khuôn mặt), q: hủy")

        path = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog", number_of_times_to_upsample=0)
            disp = frame.copy()
            if len(boxes) == 1:
                t, r, b, l = boxes[0]
                cv2.rectangle(disp, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(disp, "1 face detected - SPACE to capture",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            elif len(boxes) > 1:
                cv2.putText(disp, "Found >1 faces", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(disp, "No face", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow("Capture 1 face", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == 32 and len(boxes) == 1:
                cv2.imwrite(str(temp_path), frame)
                print(f"💾 Temp: {temp_path}")
                path = str(temp_path)
                break

        cap.release()
        cv2.destroyAllWindows()
        if path is None:
            raise RuntimeError("Không chụp được ảnh hợp lệ (cần đúng 1 khuôn mặt).")
        return path
