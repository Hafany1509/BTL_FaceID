import mysql.connector
from contextlib import contextmanager
from app.config import MYSQL  # {host, port, user, password, database}

class DB:
    def __init__(self):
        self.cn = mysql.connector.connect(**MYSQL)
        self.cn.autocommit = True

    def close(self):
        try:
            self.cn.close()
        except:
            pass

    @contextmanager
    def cur(self):
        c = self.cn.cursor(dictionary=True)
        try:
            yield c
        finally:
            c.close()

    # ---------- Helpers ----------
    def q(self, sql, params=None):
        with self.cur() as c:
            c.execute(sql, params or ())
            if c.with_rows:
                return c.fetchall()
            return []

    def exec(self, sql, params=None):
        with self.cur() as c:
            c.execute(sql, params or ())
            return c.rowcount

    # ---------- Admin accounts ----------
    def get_admin_by_username(self, username: str):
        rows = self.q("SELECT * FROM taikhoan WHERE username=%s LIMIT 1", (username,))
        return rows[0] if rows else None

    def create_admin_account(self, ten_that, ma_nv, username, password_hash):
        sql = ("INSERT INTO taikhoan(ten_that, ma_nv, username, password_hash) "
               "VALUES(%s,%s,%s,%s)")
        return self.exec(sql, (ten_that, ma_nv, username, password_hash))

    # ---------- Employees ----------
    def list_employees(self):
        sql = ("SELECT ma_nv, ten, ngaysinh, phongban, chucvu "
               "FROM nhanvien ORDER BY ma_nv")
        return self.q(sql)

    def get_employee(self, ma_nv: str):
        rows = self.q(
            "SELECT ma_nv, ten, ngaysinh, phongban, chucvu FROM nhanvien WHERE ma_nv=%s LIMIT 1",
            (ma_nv,))
        return rows[0] if rows else None

    def add_employee(self, ma_nv, ten, ngaysinh, phongban, chucvu):
        sql = ("INSERT INTO nhanvien(ma_nv, ten, ngaysinh, phongban, chucvu) "
               "VALUES (%s,%s,%s,%s,%s)")
        return self.exec(sql, (ma_nv, ten, ngaysinh, phongban, chucvu))

    # ---- Auto-ID NV: NV001, NV002, ...
    def get_next_ma_nv(self) -> str:
        """
        Lấy mã kế tiếp dựa trên MAX hiện có. Nếu bảng rỗng -> NV001.
        Sau khi xóa NV lớn nhất, lần sau sẽ dùng lại mã đó.
        """
        rows = self.q(
            "SELECT ma_nv FROM nhanvien "
            "WHERE ma_nv REGEXP '^NV[0-9]+$' "
            "ORDER BY CAST(SUBSTRING(ma_nv,3) AS UNSIGNED) DESC LIMIT 1"
        )
        if not rows:
            return "NV001"
        last = rows[0]["ma_nv"]  # VD: NV027
        try:
            num = int(last[2:]) + 1
        except Exception:
            num = 1
        return f"NV{num:03d}"

    def add_employee_auto(self, ten, ngaysinh, phongban, chucvu) -> str:
        """Tạo nhân viên với mã tự sinh, trả về ma_nv."""
        ma_nv = self.get_next_ma_nv()
        self.add_employee(ma_nv, ten, ngaysinh, phongban, chucvu)
        return ma_nv

    def delete_employee(self, ma_nv: str, ten: str):
        sql = (
            "DELETE FROM nhanvien "
            "WHERE UPPER(TRIM(ma_nv)) = UPPER(TRIM(%s)) "
            "  AND LOWER(TRIM(ten)) = LOWER(TRIM(%s)) "
            "LIMIT 1"
        )
        return self.exec(sql, (ma_nv, ten))
