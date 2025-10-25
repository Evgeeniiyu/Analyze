import sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def f_mm(th, x):
    a, b, c = th
    return a + (b*x) / (c + x + 1e-12)

def lm_fit(func, x, y, theta0, max_iter=400, lam0=1e-2, tol=1e-7):
    th = np.array(theta0, dtype=float); lam = lam0
    def resid(t): return y - func(t, x)
    r = resid(th); sse = float(np.sum(r*r))
    for _ in range(max_iter):
        eps = 1e-6; m = len(th); J = np.zeros((len(x), m))
        for j in range(m):
            step = eps*(1+abs(th[j])); tp = th.copy(); tm = th.copy()
            tp[j] += step; tm[j] -= step
            fp = func(tp, x); fm = func(tm, x)
            J[:, j] = (fp - fm) / (2*step)
        JTJ = J.T @ J; JTr = J.T @ r; A = JTJ + lam*np.eye(m)
        try:
            delta = np.linalg.solve(A, JTr)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(A, JTr, rcond=None)[0]
        new = th + delta
        r_new = resid(new); sse_new = float(np.sum(r_new*r_new))
        if sse_new < sse:
            th = new; r = r_new
            if abs(sse - sse_new) < tol*(1+sse): return th, sse_new
            sse = sse_new; lam *= 0.7
        else:
            lam *= 2.0
        if np.linalg.norm(delta) < tol*(1+np.linalg.norm(th)): return th, sse
    return th, sse

def main(csv_path: str = "glucose_hemo.csv"):
    df = pd.read_csv(csv_path).dropna()
    x = pd.to_numeric(df.iloc[:,0], errors="coerce").values
    y = pd.to_numeric(df.iloc[:,1], errors="coerce").values
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 6:
        raise ValueError("Need at least 6 observations.")
    theta0 = [float(np.min(y)), float(np.ptp(y)), float(np.median(x))]
    th, sse = lm_fit(f_mm, x, y, theta0)
    yhat = f_mm(th, x)
    rmse = float(np.sqrt(np.mean((y - yhat)**2)))
    R2 = 1 - sse/np.sum((y - y.mean())**2)
    print(f"Fitted Michaelis–Menten: a={th[0]:.4f}, b={th[1]:.4f}, c={th[2]:.4f}; RMSE={rmse:.4f}, R2={R2:.4f}")

    plt.figure()
    plt.scatter(x, y, alpha=0.85, label="data")
    xx = np.linspace(x.min(), x.max(), 200)
    plt.plot(xx, f_mm(th, xx), label="MM fit")
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.legend()
    plt.tight_layout()
    plt.savefig("task3_mm_fit.png", dpi=150)
    print("Saved plot: task3_mm_fit.png")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "glucose_hemo.csv")
