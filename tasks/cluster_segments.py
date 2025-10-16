
import sys, pathlib
import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
TASKS = HERE.parent
for p in (str(ROOT), str(TASKS)):
    if p not in sys.path:
        sys.path.insert(0, p)
try:
    from dbconnect import get_engine, exec_sql
except ModuleNotFoundError:
    from tasks.dbconnect import get_engine, exec_sql

DDL_STMTS = (
    """
    CREATE TABLE IF NOT EXISTS mat_store_cluster (
      mct_id VARCHAR(32) PRIMARY KEY,
      ym VARCHAR(8),
      cluster_id INT,
      k INT,
      inertia DOUBLE,
      silhouette DOUBLE,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS mat_cluster_profile (
      cluster_id INT,
      k INT,
      feature VARCHAR(64),
      mean DOUBLE,
      std DOUBLE,
      n INT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (cluster_id, k, feature, created_at)
    );
    """
)

FEATURES = [
    "age_10s","age_30s","age_40s","age_50s","age_60p",
    "aud_new","aud_reu","aud_res","aud_work","aud_flow",
    "cancel_high","delivery_high"
]

def load_df(eng):
    return pd.read_sql("SELECT * FROM vw_store_features", eng).fillna(0.0)

def choose_k(Xz, k_list=(3,4,5,6,7)):
    best = None
    for k in k_list:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(Xz)
        sil = silhouette_score(Xz, labels) if len(set(labels))>1 else -1
        if (best is None) or (sil > best["silhouette"]):
            best = {"k": k, "km": km, "silhouette": sil, "inertia": km.inertia_}
    return best

def save_results(eng, df, labels, k, inertia, silhouette):
    with eng.begin() as cx:
        # 매장별 클러스터
        for mct, ym, lab in zip(df["mct_id"], df["ym"], labels):
            cx.execute(text("""
                INSERT INTO mat_store_cluster (mct_id, ym, cluster_id, k, inertia, silhouette)
                VALUES (:m,:y,:c,:k,:i,:s)
                ON DUPLICATE KEY UPDATE
                  ym=VALUES(ym), cluster_id=VALUES(cluster_id),
                  k=VALUES(k), inertia=VALUES(inertia), silhouette=VALUES(silhouette)
            """), {"m":mct, "y":ym, "c":int(lab), "k":int(k), "i":float(inertia), "s":float(silhouette)})

        # 클러스터 프로파일
        prof = df[FEATURES].copy()
        prof["cluster_id"] = labels
        for cid, g in prof.groupby("cluster_id"):
            n = int(len(g))
            means = g[FEATURES].mean()
            stds  = g[FEATURES].std(ddof=0).fillna(0.0)
            for f in FEATURES:
                cx.execute(text("""
                    INSERT INTO mat_cluster_profile (cluster_id, k, feature, mean, std, n)
                    VALUES (:c,:k,:f,:m,:d,:n)
                """), {"c":int(cid), "k":int(k), "f":f,
                       "m":float(means[f]), "d":float(stds[f]), "n":n})

def run(k=None):
    # DDL 보장
    for stmt in DDL_STMTS:
        exec_sql(stmt)

    eng = get_engine()
    df  = load_df(eng)

    X   = df[FEATURES].values
    Xz  = StandardScaler().fit_transform(X)

    if k is None:
        best = choose_k(Xz)
        km, k, sil, inertia = best["km"], best["k"], best["silhouette"], best["inertia"]
        labels = km.labels_
    else:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(Xz)
        labels = km.labels_
        sil = silhouette_score(Xz, labels) if len(set(labels))>1 else -1
        inertia = km.inertia_

    save_results(eng, df, labels, k, inertia, sil)
    print(f"✅ clustering done: k={k}, silhouette={sil:.3f}, inertia={inertia:.1f}")

if __name__ == "__main__":
    run()
