import sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(csv_path: str = "lipo_hemo.csv"):
    df = pd.read_csv(csv_path)
    lipocol = next(c for c in df.columns if c.lower().startswith(("lipoprotein","ліпопроте")))
    hemocol = next(c for c in df.columns if c.lower().startswith(("hemoglobin","гемоглоб")))
    x = df[lipocol].astype(float).values
    y = df[hemocol].astype(float).values
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)

    xm, ym = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    r = float(((x-xm)*(y-ym)).sum() / ((n-1)*sx*sy))
    dfree = n - 2
    t = r * math.sqrt(dfree/(1 - r*r)) if abs(r) < 1 and dfree > 0 else float("inf")
    p = 2 * (0.5 * (1 - math.erf(abs(t)/math.sqrt(2))))
    if n > 3 and abs(r) < 1:
        z = 0.5*math.log((1+r)/(1-r)); se = 1/math.sqrt(n-3)
        zlo, zhi = z - 1.96*se, z + 1.96*se
        rlo = (math.exp(2*zlo)-1)/(math.exp(2*zlo)+1)
        rhi = (math.exp(2*zhi)-1)/(math.exp(2*zhi)+1)
    else:
        rlo = rhi = float("nan")

    print(f"n={n}, r={r:.4f}, t={t:.3f}, p≈{p:.4f}, 95%CI=[{rlo:.3f},{rhi:.3f}]")

    plt.figure()
    plt.scatter(x, y, alpha=0.8)
    b1, b0 = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    yy = b1*xx + b0
    plt.plot(xx, yy)
    plt.xlabel(lipocol)
    plt.ylabel(hemocol)
    plt.title("Lipoproteins vs Hemoglobin")
    plt.tight_layout()
    plt.savefig("task2_correlation_plot.png", dpi=150)
    print("Saved plot: task2_correlation_plot.png")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lipo_hemo.csv")
