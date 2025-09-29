# tasks/dbio.py
from __future__ import annotations
from sqlalchemy import create_engine, text
import pathlib, toml

def _load_mysql_from_secrets():
    sec = pathlib.Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"
    cfg = toml.load(sec)
    m = cfg["mysql"]
    return m["host"], int(m["port"]), m["user"], m["password"], m["database"]

def get_engine():
    host, port, user, pw, db = _load_mysql_from_secrets()
    url = f"mysql+pymysql://{user}:{pw}@{host}:{port}/{db}?charset=utf8mb4"
    return create_engine(url, future=True)

def exec_sql(sql: str, params: dict | None = None):
    eng = get_engine()
    with eng.begin() as cx:
        cx.execute(text(sql), params or {})

def exec_sql_script(sql_script: str):
    eng = get_engine()
    def strip_comments(block: str) -> str:
        lines = []
        for ln in block.splitlines():
            if ln.strip().startswith("--"):
                continue
            lines.append(ln)
        return "\n".join(lines).strip()
    cleaned = strip_comments(sql_script)
    statements = [s.strip() for s in cleaned.split(";") if s.strip()]
    with eng.begin() as cx:
        for stmt in statements:
            cx.execute(text(stmt))