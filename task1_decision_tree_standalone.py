from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import math
import numpy as np
import pandas as pd

# ============ Утиліти ============
def _entropy(counts: Dict[int,int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    s = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            s -= p * math.log2(p)
    return s

def _gini(counts: Dict[int,int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    s = 1.0
    for c in counts.values():
        p = c / total
        s -= p * p
    return s

def _criterion_value(counts: Dict[int,int], criterion: str) -> float:
    return _gini(counts) if criterion == "gini" else _entropy(counts)

def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    lab_to_idx = {lab:i for i,lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[lab_to_idx[t], lab_to_idx[p]] += 1
    return cm

def prf(cm: np.ndarray):
    k = cm.shape[0]
    prec = np.zeros(k)
    rec = np.zeros(k)
    f1 = np.zeros(k)
    for i in range(k):
        tp = cm[i,i]
        fp = cm[:,i].sum() - tp
        fn = cm[i,:].sum() - tp
        p = tp / (tp + fp) if (tp+fp)>0 else 0.0
        r = tp / (tp + fn) if (tp+fn)>0 else 0.0
        f = 2*p*r/(p+r) if (p+r)>0 else 0.0
        prec[i], rec[i], f1[i] = p, r, f
    return prec, rec, f1

def train_test_split_df(X: pd.DataFrame, y: Union[pd.Series, list, np.ndarray],
                        test_size: float = 0.2, random_state: Optional[int] = None):
    rng = np.random.default_rng(random_state)
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    ts = max(1, int(round(test_size * n)))
    te_idx = idx[:ts]
    tr_idx = idx[ts:]
    return X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True), \
           pd.Series(np.array(y)[tr_idx]).reset_index(drop=True), pd.Series(np.array(y)[te_idx]).reset_index(drop=True)

# ============ Структури дерева ============
@dataclass
class Node:
    is_leaf: bool = False
    prediction: Optional[int] = None
    class_counts: Dict[int,int] = field(default_factory=dict)
    feature: Optional[str] = None
    threshold: Optional[float] = None  # для числових сплітів
    children: Dict[Any, "Node"] = field(default_factory=dict)  # для категоріальних
    left: Optional["Node"] = None      # <= threshold
    right: Optional["Node"] = None     # > threshold
    depth: int = 0
    n_samples: int = 0

class DecisionTreeID3:
    def __init__(self, criterion: str = "entropy", max_depth: Optional[int] = None,
                 min_gain: float = 1e-5, min_samples_split: int = 2, random_state: Optional[int] = None):
        assert criterion in ("entropy", "gini")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.root_: Optional[Node] = None
        self.class_names_: List[str] = []
        self._label_to_idx: Dict[Any,int] = {}
        self._idx_to_label: Dict[int,Any] = {}

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, list, np.ndarray]):
        X = X.reset_index(drop=True)
        y = pd.Series(np.array(y)).reset_index(drop=True)

        names = sorted(pd.unique(y.astype(str)))
        self.class_names_ = list(names)
        self._label_to_idx = {n:i for i,n in enumerate(self.class_names_)}
        self._idx_to_label = {i:n for n,i in self._label_to_idx.items()}
        y_enc = y.astype(str).map(self._label_to_idx).astype(int).values

        self._is_numeric = {c: pd.api.types.is_numeric_dtype(X[c]) for c in X.columns}
        self.root_ = self._build_tree(X, y_enc, depth=0)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray, list]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            arr = X.reset_index(drop=True)
        else:
            arr = pd.DataFrame(X)
        preds_idx = [self._predict_row(self.root_, arr.iloc[i]) for i in range(len(arr))]
        return np.array(preds_idx, dtype=int)

    def _predict_row(self, node: Node, row: pd.Series) -> int:
        while not node.is_leaf:
            if node.threshold is not None:
                v = float(row[node.feature])
                node = node.left if v <= node.threshold else node.right
            else:
                v = row[node.feature]
                node = node.children.get(v, self._major_leaf(node))
        return node.prediction

    def _major_leaf(self, node: Node) -> Node:
        if node.class_counts:
            pred = max(node.class_counts.items(), key=lambda kv: kv[1])[0]
            return Node(is_leaf=True, prediction=pred, class_counts=node.class_counts)
        return Node(is_leaf=True, prediction=0, class_counts={0: node.n_samples})

    def _build_tree(self, X: pd.DataFrame, y_enc: np.ndarray, depth: int) -> Node:
        node = Node(depth=depth, n_samples=len(y_enc))
        counts = self._class_counts(y_enc)
        node.class_counts = counts
        node.prediction = max(counts.items(), key=lambda kv: kv[1])[0]

        # зупинки
        if len(counts) == 1:
            node.is_leaf = True; return node
        if self.max_depth is not None and depth >= self.max_depth:
            node.is_leaf = True; return node
        if len(y_enc) < self.min_samples_split:
            node.is_leaf = True; return node

        # найкращий спліт
        best = self._best_split(X, y_enc)
        if best is None or best["gain"] < self.min_gain:
            node.is_leaf = True; return node

        feat = best["feature"]; node.feature = feat
        if best["type"] == "numeric":
            thr = best["threshold"]; node.threshold = thr
            mask = X[feat].astype(float) <= thr
            Xl, yl = X[mask], y_enc[mask.values]
            Xr, yr = X[~mask], y_enc[~mask.values]
            node.left = self._build_tree(Xl, yl, depth+1)
            node.right = self._build_tree(Xr, yr, depth+1)
        else:
            node.children = {}
            for val, idxs in best["groups"].items():
                Xi = X.loc[idxs]; yi = y_enc[idxs]
                node.children[val] = self._build_tree(Xi, yi, depth+1)
        return node

    def _class_counts(self, y_enc: np.ndarray) -> Dict[int,int]:
        unique, counts = np.unique(y_enc, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}

    def _best_split(self, X: pd.DataFrame, y_enc: np.ndarray):
        parent_counts = self._class_counts(y_enc)
        parent_imp = _criterion_value(parent_counts, self.criterion)
        best = None

        for feat in X.columns:
            col = X[feat]
            if self._is_numeric[feat]:
                vals = np.unique(col.astype(float))
                if len(vals) <= 1: continue
                thr_list = (vals[:-1] + vals[1:]) / 2.0
                for thr in thr_list:
                    left_mask = col.astype(float) <= thr
                    right_mask = ~left_mask
                    if left_mask.sum() == 0 or right_mask.sum() == 0: continue
                    left_counts = self._class_counts(y_enc[left_mask.values])
                    right_counts = self._class_counts(y_enc[right_mask.values])
                    n = len(y_enc)
                    wl = left_mask.sum()/n; wr = right_mask.sum()/n
                    child_imp = wl*_criterion_value(left_counts, self.criterion) + \
                                wr*_criterion_value(right_counts, self.criterion)
                    gain = parent_imp - child_imp
                    if (best is None) or (gain > best["gain"]):
                        best = {"feature": feat, "type":"numeric", "threshold": float(thr), "gain": float(gain)}
            else:
                groups = {}
                for val in col.unique():
                    idxs = col.index[col == val]
                    if len(idxs) > 0: groups[val] = idxs
                if len(groups) <= 1: continue
                n = len(y_enc); child_imp = 0.0
                for val, idxs in groups.items():
                    counts = self._class_counts(y_enc[idxs])
                    w = len(idxs)/n
                    child_imp += w*_criterion_value(counts, self.criterion)
                gain = parent_imp - child_imp
                if (best is None) or (gain > best["gain"]):
                    best = {"feature": feat, "type":"categorical", "groups": groups, "gain": float(gain)}
        return best

    def print_tree(self) -> str:
        assert self.root_ is not None, "Call fit() first"
        lines: List[str] = []
        def rec(n: Node, indent: str):
            if n.is_leaf:
                dist = " ".join([f"{self._idx_to_label[k]}:{v}" for k,v in sorted(n.class_counts.items())])
                lines.append(f"{indent}Leaf: pred={self._idx_to_label[n.prediction]} | n={n.n_samples} | [{dist}]")
                return
            if n.threshold is not None:
                lines.append(f"{indent}if {n.feature} <= {n.threshold:.6g}:")
                rec(n.left, indent + "  ")
                lines.append(f"{indent}else:  # {n.feature} > {n.threshold:.6g}")
                rec(n.right, indent + "  ")
            else:
                for val, child in n.children.items():
                    lines.append(f"{indent}if {n.feature} == {val!r}:")
                    rec(child, indent + "  ")
        rec(self.root_, "")
        return "\n".join(lines)

# ============ Демо на синтетиці ============
def make_iris_like(n_per_class=50, seed=7):
    rng = np.random.default_rng(seed)
    means = np.array([[5.0,3.5,1.4,0.2],[6.0,2.8,4.5,1.3],[6.5,3.0,5.5,2.0]])
    cov = np.diag([0.2, 0.15, 0.25, 0.1])
    names = ["setosa","versicolor","virginica"]
    X = np.vstack([rng.multivariate_normal(means[i], cov, size=n_per_class) for i in range(3)])
    y = np.array(sum(([names[i]]*n_per_class for i in range(3)), []))
    df = pd.DataFrame(X, columns=["sepal_len","sepal_wid","petal_len","petal_wid"])
    df["species"] = y
    return df, "species"

def accuracy(cm): 
    total = cm.sum()
    return float(np.trace(cm)/total) if total>0 else 0.0

def main():
    # Дані
    df, target = make_iris_like()
    X, y = df.drop(columns=[target]), df[target]

    # Train/Test
    Xtr, Xte, ytr, yte = train_test_split_df(X, y, test_size=0.2, random_state=7)

    # Навчання
    tree = DecisionTreeID3(criterion="entropy", max_depth=6, min_gain=1e-5, random_state=7).fit(Xtr, ytr)

    # Прогноз у НАЗВАХ класів (а не індексах)
    y_pred_idx = tree.predict(Xte)
    y_pred = np.array([tree._idx_to_label[int(i)] for i in y_pred_idx], dtype=object)

    # Метрики
    cm = confusion_matrix(yte.astype(str).values, y_pred)
    prec, rec, f1 = prf(cm); acc = accuracy(cm)

    # Збереження у файли
    Path("tree_demo.txt").write_text(tree.print_tree(), encoding="utf-8")
    lines = [
        "=== Task 1: Decision Tree (ID3) ===",
        "Confusion matrix (rows=true, cols=pred):",
        str(cm),
        f"Accuracy: {acc:.3f}",
        "Precision: " + np.array2string(np.round(prec,3), separator=', '),
        "Recall:    " + np.array2string(np.round(rec,3), separator=', '),
        "F1-score:  " + np.array2string(np.round(f1,3), separator=', '),
    ]
    Path("task1_results.txt").write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()