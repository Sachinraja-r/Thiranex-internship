"""run_all.py — Regenerate data, plots, models, and HTML report."""
import subprocess, sys, os, time

BASE_DIR = os.path.dirname(__file__)
SCRIPTS  = [
    ("scripts/01_generate_data.py",      "Data Generation"),
    ("scripts/02_eda.py",                "EDA & Plots"),
    ("scripts/03_feature_engineering.py","Feature Engineering"),
    ("scripts/04_model_training.py",     "ML Model Training"),
    ("scripts/05_dashboard.py",          "Dashboard"),
    ("build_report.py",                  "HTML Report"),
]

def run(script, label):
    path = os.path.join(BASE_DIR, script)
    print(f"\n>> {label}")
    t = time.time()
    r = subprocess.run([sys.executable, "-X", "utf8", path])
    print(f"   Done in {time.time()-t:.1f}s")
    if r.returncode != 0:
        print(f"FAILED: {script}")
        sys.exit(1)

if __name__ == "__main__":
    for s, l in SCRIPTS:
        run(s, l)
    print("\nAll done! Files ready.")
