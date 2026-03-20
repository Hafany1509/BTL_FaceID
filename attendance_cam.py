# app/attendance_cam.py
import time
from datetime import datetime, date, time as dtime, timedelta
import cv2
import numpy as np
import face_recognition

from app.db import DB
from app.encoding_loaded import load_all_encodings
from app.config import WORK_START_HOUR

CAP_WIDTH    = 320
CAP_HEIGHT   = 240
FRAME_STRIDE = 4
DOWNSCALE    = 0.40
TOLERANCE    = 0.50
COOLDOWN_S   = 5.0
STATUS_MS    = 4000   # hiển thị "Done" 4s

def _extract_id_from_label(label: str) -> str:
    parts = label.split("_")
    return parts[-1] if len(parts) >= 2 else ""

def _hms_to_seconds(v) -> int:
    """Chuyển giá trị thời gian (str 'HH:MM:SS' | datetime.time | datetime.timedelta) -> giây."""
    if v is None:
        return 0
    if isinstance(v, str):
        hh, mm, ss = [int(x) for x in v.split(":")]
        return hh*3600 + mm*60 + ss
    if isinstance(v, dtime):
        return v.hour*3600 + v.minute*60 + v.second
    if isinstance(v, timedelta):
        # mysql-connector thường trả TIME -> timedelta
        return int(v.total_seconds()) % 86400
    # fallback
    try:
        hh, mm, ss = [int(x) for x in str(v).split(":")]
        return hh*3600 + mm*60 + ss
    except Exception:
        return 0

def _sec_to_hms(s: int) -> str:
    s = int(s) % 86400
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

def _compute_checkin_note(now_hms: str) -> str:
    try:
        hh, mm, ss = [int(x) for x in now_hms.split(":")]
    except Exception:
        return "Đúng giờ"
    start = dtime(hour=WORK_START_HOUR, minute=0, second=0)
    nowt  = dtime(hour=hh, minute=mm, second=ss)
    if nowt <= start:
        return "Đúng giờ"
    late_min = (hh - WORK_START_HOUR) * 60 + mm
    if late_min <= 0:
        return "Đúng giờ"
    if late_min >= 60:
        h = late_min // 60
        p = late_min % 60
        return f"Muộn {h}h{p}p" if p else f"Muộn {h}h"
    return f"Muộn {late_min}p"

def _fetch_emp_info(ma_nv: str):
    if not ma_nv:
        return None
    try:
        return DB().get_employee(ma_nv)
    except Exception:
        return None

def _auto_update_attendance(recogn_name: str, recogn_id: str):
    db = DB()
    today = date.today().strftime("%Y-%m-%d")
    now   = datetime.now().strftime("%H:%M:%S")
    try:
        rows = db.q("SELECT * FROM chamcong WHERE ma_nv=%s AND ngay=%s LIMIT 1",
                    (recogn_id, today))
        if not rows:
            note = _compute_checkin_note(now)
            db.q("INSERT INTO chamcong (ma_nv, ten_nv, ngay, check_in, check_out, total_seconds, note) "
                 "VALUES (%s,%s,%s,%s,%s,NULL,%s)",
                 (recogn_id, recogn_name, today, now, None, note))
            rec = db.q("SELECT ma_nv,ten_nv,ngay,check_in,check_out,total_seconds,note "
                       "FROM chamcong WHERE ma_nv=%s AND ngay=%s LIMIT 1", (recogn_id, today))[0]
            return "checkin", rec

        row = rows[0]
        if not row["check_in"]:
            note = _compute_checkin_note(now)
            db.q("UPDATE chamcong SET check_in=%s, note=%s WHERE id=%s", (now, note, row["id"]))
            row["check_in"] = now
            row["note"] = note
            return "checkin", row

        if not row["check_out"]:
            in_sec  = _hms_to_seconds(row["check_in"])
            now_sec = _hms_to_seconds(now)
            total_seconds = max(0, now_sec - in_sec)
            db.q("UPDATE chamcong SET check_out=%s, total_seconds=%s WHERE id=%s",
                 (now, total_seconds, row["id"]))
            row["check_out"] = now
            row["total_seconds"] = total_seconds
            return "checkout", row

        return "done", row

    except Exception as e:
        print("❌ DB error:", e)
        return "error", None

def run_manual_attendance(camera_index=0, on_event=None):
    try:
        cv2.setUseOptimized(True)
        cv2.setNumThreads(2)
    except Exception:
        pass

    known_encs, labels = load_all_encodings()
    if known_encs is None or len(known_encs) == 0:
        print("❌ Không có dữ liệu encodings.")
        known_encs = np.zeros((0, 128), dtype=np.float32); labels = []

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except: pass
    try:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    except Exception:
        pass

    if not cap.isOpened():
        print("❌ Không mở được camera.")
        return

    print("➡ Auto Attendance — ESC để thoát. Lần 1 trong ngày = Check-in, lần 2 = Check-out.")
    frame_id = 0
    status_text = ""
    status_until = 0.0
    last_success = False
    last_box = None
    cooldown = {}   # {ma_nv: last_time}
    last_emp_info = None
    last_label = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame_id += 1
        if frame.shape[1] != CAP_WIDTH or frame.shape[0] != CAP_HEIGHT:
            frame = cv2.resize(frame, (CAP_WIDTH, CAP_HEIGHT), interpolation=cv2.INTER_LINEAR)

        found_label = None
        found_id    = ""
        found_loc   = None

        if frame_id % FRAME_STRIDE == 0 and len(known_encs) > 0:
            small = cv2.resize(frame, (0,0), fx=DOWNSCALE, fy=DOWNSCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb_small, model="hog", number_of_times_to_upsample=0)
            if locations:
                encs = face_recognition.face_encodings(rgb_small, [locations[0]], num_jitters=1)
                if encs:
                    enc = encs[0]
                    dists = np.linalg.norm(known_encs - enc, axis=1)
                    idx = int(np.argmin(dists))
                    if dists[idx] <= TOLERANCE:
                        found_label = labels[idx]
                        found_id = _extract_id_from_label(found_label)
                        top, right, bottom, left = locations[0]
                        top = int(top / DOWNSCALE); right = int(right / DOWNSCALE)
                        bottom = int(bottom / DOWNSCALE); left = int(left / DOWNSCALE)
                        found_loc = (left, top, right, bottom)
                        last_box = found_loc

        now_ts = time.time()
        if found_id:
            allow = (now_ts - cooldown.get(found_id, 0.0) >= COOLDOWN_S)
            if allow:
                person_name = found_label.rsplit("_", 1)[0] if found_label else "Unknown"
                action, rec = _auto_update_attendance(person_name, found_id)
                cooldown[found_id] = now_ts

                if on_event and isinstance(rec, dict) and action in ("checkin", "checkout"):
                    try: on_event(rec)
                    except Exception: pass

                last_success = action in ("checkin", "checkout")
                status_text = "Done" if last_success else ("Full" if action == "done" else "LỖI")
                status_until = now_ts + (STATUS_MS / 1000.0)

                last_emp_info = _fetch_emp_info(found_id)
                last_label = found_label

        # ===== Overlay =====
        if last_box:
            l, t, r, b = last_box
            color = (0, 255, 0) if (time.time() < status_until and last_success) else (255, 255, 255)
            cv2.rectangle(frame, (l, t), (r, b), color, 2)

        cv2.rectangle(frame, (6, 6), (CAP_WIDTH - 6, 26), (0, 0, 0), -1)
        cv2.putText(frame, "", (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        if last_emp_info and last_label:
            name = last_label.rsplit("_", 1)[0]
            ma   = _extract_id_from_label(last_label)
            dept = last_emp_info.get("phongban") or ""
            role = last_emp_info.get("chucvu") or ""
            info_lines = [f"Ten: {name}", f"Ma:  {ma}", (f"PB:  {dept}" if dept else ""), (f"CV:  {role}" if role else "")]
            y0 = CAP_HEIGHT - 72
            cv2.rectangle(frame, (6, y0-2), (CAP_WIDTH-6, CAP_HEIGHT-40), (0,0,0), -1)
            y = y0
            for line in info_lines:
                if not line: continue
                cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                y += 16

        if time.time() < status_until and status_text:
            cv2.rectangle(frame, (6, CAP_HEIGHT - 36), (CAP_WIDTH - 6, CAP_HEIGHT - 6), (0, 64, 0), -1)
            cv2.putText(frame, status_text, (12, CAP_HEIGHT - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Manual Attendance (Auto)", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
