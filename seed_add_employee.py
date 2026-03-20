# app/seed_add_employee.py
from app.db import DB
import sys

# Cách dùng:
#   python -m app.seed_add_employee <ma_nv> "<ten>" [ngaysinh] [phongban] [chucvu]
# Ví dụ:
#   python -m app.seed_add_employee nv01 "chung" 2004-01-01 "Ky thuat" admin

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m app.seed_add_employee <ma_nv> \"<ten>\" [ngaysinh] [phongban] [chucvu]")
        return
    ma_nv = sys.argv[1]
    ten = sys.argv[2]
    ngaysinh = sys.argv[3] if len(sys.argv) >= 4 else None
    phongban = sys.argv[4] if len(sys.argv) >= 5 else None
    chucvu = sys.argv[5] if len(sys.argv) >= 6 else "admin"

    db = DB()
    db.exec(
        "INSERT INTO nhanvien (ma_nv, ten, ngaysinh, phongban, chucvu) VALUES (%s,%s,%s,%s,%s)",
        (ma_nv, ten, ngaysinh, phongban, chucvu),
    )
    print(f"✅ Đã thêm nhân viên: {ma_nv} - {ten}")

if __name__ == "__main__":
    main()
