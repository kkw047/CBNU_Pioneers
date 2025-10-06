
import sys, pathlib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sqlalchemy import text

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

FEATURES = [
    "age_10s","age_30s","age_40s","age_50s","age_60p",
    "aud_new","aud_reu","aud_res","aud_work","aud_flow",
    "cancel_high","delivery_high"
]

DDL_RULES = """
CREATE TABLE IF NOT EXISTS mat_tree_rules (
  rule_id INT AUTO_INCREMENT PRIMARY KEY,
  k INT,
  depth INT,
  cluster_id INT,
  rule_text VARCHAR(1024),
  support INT,
  rule_precision DOUBLE,
  train_acc DOUBLE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

DDL_ASSIGN = """
CREATE TABLE IF NOT EXISTS mat_store_rule_assign (
  mct_id VARCHAR(32),
  k INT,
  rule_id INT,
  cluster_id INT,
  PRIMARY KEY (mct_id, k),
  KEY (rule_id),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def fetch_dataset(eng):
    return pd.read_sql("""
        SELECT c.mct_id,
               f.ym,
               c.cluster_id,
               c.k,
               f.age_10s,f.age_30s,f.age_40s,f.age_50s,f.age_60p,
               f.aud_new,f.aud_reu,f.aud_res,f.aud_work,f.aud_flow,
               f.cancel_high,f.delivery_high
        FROM mat_store_cluster c
        JOIN vw_store_features f ON f.mct_id = c.mct_id
        WHERE c.k = (SELECT k FROM mat_store_cluster ORDER BY created_at DESC LIMIT 1)
    """, eng)

def extract_rules(clf: DecisionTreeClassifier, feature_names, X, y, k, eng):
    tree: _tree.Tree = clf.tree_
    paths = []
    def recurse(node, path):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            f = feature_names[tree.feature[node]]
            thr = tree.threshold[node]
            recurse(tree.children_left[node],  path + [(f, thr, "<=")])
            recurse(tree.children_right[node], path + [(f, thr, ">")])
        else:
            paths.append(path)
    recurse(0, [])

    X_np, y_np = np.asarray(X), np.asarray(y)

    inserted = []
    with eng.begin() as cx:
        for pth in paths:
            if not pth:
                continue
            mask = np.ones(len(X_np), dtype=bool)
            conds=[]
            for (f, thr, op) in pth:
                j = feature_names.index(f)
                mask &= (X_np[:, j] <= thr) if op == "<=" else (X_np[:, j] > thr)
                conds.append(f"{f} {op} {thr:.3f}")
            support = int(mask.sum())
            if support == 0:
                continue
            labels = y_np[mask]
            values, counts = np.unique(labels, return_counts=True)
            cid = int(values[np.argmax(counts)])
            precision = float(np.max(counts) / support)
            rule_text = " AND ".join(conds)
            res = cx.execute(text("""
                INSERT INTO mat_tree_rules(k, depth, cluster_id, rule_text, support, rule_precision, train_acc)
                VALUES(:k, :d, :cid, :txt, :sup, :prec, :acc)
            """), {"k": int(k), "d": len(pth), "cid": cid,
                   "txt": rule_text[:1024], "sup": support,
                   "prec": precision, "acc": None})
            inserted.append((res.lastrowid, pth, cid))
    return inserted

def assign_rules(eng, rules, clf, feature_names, df, k):
    tree: _tree.Tree = clf.tree_
    rule_leaf = {}
    def path_to_leaf(path):
        node = 0
        for (f, thr, op) in path:
            node = tree.children_left[node] if op == "<=" else tree.children_right[node]
        return node
    for rid, pth, cid in rules:
        rule_leaf[path_to_leaf(pth)] = (rid, cid)

    X = df[feature_names].values
    leaves = clf.apply(X)
    with eng.begin() as cx:
        for (mct_id, leaf) in zip(df["mct_id"].values, leaves):
            rid_cid = rule_leaf.get(leaf)
            if not rid_cid:
                continue
            rid, cid = rid_cid
            cx.execute(text("""
                INSERT INTO mat_store_rule_assign (mct_id, k, rule_id, cluster_id)
                VALUES (:m,:k,:r,:c)
                ON DUPLICATE KEY UPDATE rule_id=VALUES(rule_id), cluster_id=VALUES(cluster_id)
            """), {"m": mct_id, "k": int(k), "r": int(rid), "c": int(cid)})

def run(max_depth=3, min_samples_leaf=50):
    eng = get_engine()
    exec_sql(DDL_RULES)
    exec_sql(DDL_ASSIGN)

    df = fetch_dataset(eng)
    k = int(df["k"].iloc[0])

    X = df[FEATURES].values
    y = df["cluster_id"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
    clf.fit(X_tr, y_tr)

    train_acc = accuracy_score(y_tr, clf.predict(X_tr))
    test_acc  = accuracy_score(y_te, clf.predict(X_te))
    print(f"DT k={k} | depth={max_depth}, min_leaf={min_samples_leaf} | train={train_acc:.3f}, test={test_acc:.3f}")

    rules = extract_rules(clf, FEATURES, X_tr, y_tr, k, eng)
    with eng.begin() as cx:
        cx.execute(text("UPDATE mat_tree_rules SET train_acc=:a WHERE k=:k AND train_acc IS NULL"),
                   {"a": float(train_acc), "k": int(k)})
    assign_rules(eng, rules, clf, FEATURES, df[["mct_id"]+FEATURES], k)
    print(f"✅ rules saved: {len(rules)} rules, and assignments done.")

if __name__ == "__main__":
    run()
