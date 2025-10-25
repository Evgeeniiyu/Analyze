import argparse
import numpy as np
import pandas as pd

def f_mm(th, x):
    a, b, c = th
    return a + (b * x) / (c + x + 1e-12)

def lm(func, x, y, th, iters=160, lam=1e-2):
    def resid(t): return y - func(t, x)
    r = resid(th); sse = float(np.sum(r*r))
    for _ in range(iters):
        eps = 1e-6; m = len(th); J = np.zeros((len(x), m))
        for j in range(m):
            step = eps*(1+abs(th[j])); tp = th.copy(); tm = th.copy()
            tp[j] += step; tm[j] -= step
            fp = func(tp, x); fm = func(tm, x)
            J[:, j] = (fp - fm) / (2*step)
        JTJ = J.T @ J; JTr = J.T @ r; A = JTJ + lam*np.eye(m)
        delta = np.linalg.solve(A, JTr)
        new = th + delta
        r_new = resid(new); sse_new = float(np.sum(r_new*r_new))
        if sse_new < sse:
            th = new; r = r_new; sse = sse_new; lam *= 0.7
        else:
            lam *= 2.0
        if np.linalg.norm(delta) < 1e-7*(1+np.linalg.norm(th)): break
    return th, sse

def main():
    ap = argparse.ArgumentParser(description="Print only equation and RMSE.")
    ap.add_argument("csv", nargs="?", default="task4_synth_data.csv",
                    help="Input CSV with columns: stabilized_glucose, hemoglobin")
    ap.add_argument("--out", default=None, help="Optional TXT file to also write the single line")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna().sort_values("stabilized_glucose")
    x = pd.to_numeric(df["stabilized_glucose"], errors="coerce").values
    y = pd.to_numeric(df["hemoglobin"], errors="coerce").values
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    # simple init
    a0 = float(np.min(y)); b0 = float(np.max(y) - a0); c0 = float(np.median(x))
    th0 = np.array([a0, b0, c0], dtype=float)

    th, sse = lm(f_mm, x, y, th0)
    yhat = f_mm(th, x)
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))

    line = f"Equation: y = {th[0]:.4f} + ({th[1]:.4f}*x)/({th[2]:.4f} + x); RMSE={rmse:.4f}"
    print(line)  # only one line to stdout
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(line + "\n")

if __name__ == "__main__":
    main()