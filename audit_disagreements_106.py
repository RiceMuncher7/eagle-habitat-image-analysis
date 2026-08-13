import pandas as pd
from pathlib import Path

COMPARE_DIR = Path("outputs_compare_roi065_106")
MERGED = COMPARE_DIR / "merged_per_image.csv"
OUT = COMPARE_DIR / "audit_top_disagreements.csv"

df = pd.read_csv(MERGED)

# Expect columns like: water_frac_classical, water_frac_dl, vegetation_frac_classical, vegetation_frac_dl
df["water_abs_diff"] = (df["water_frac_classical"] - df["water_frac_dl"]).abs()
df["veg_abs_diff"] = (df["vegetation_frac_classical"] - df["vegetation_frac_dl"]).abs()

topN = 15
top_water = df.sort_values("water_abs_diff", ascending=False).head(topN).copy()
top_water["metric"] = "water_frac"

top_veg = df.sort_values("veg_abs_diff", ascending=False).head(topN).copy()
top_veg["metric"] = "vegetation_frac"

audit = pd.concat([top_water, top_veg], ignore_index=True)

keep = [c for c in [
    "metric", "file_classical", "file_dl", "year",
    "water_frac_classical", "water_frac_dl", "water_abs_diff",
    "vegetation_frac_classical", "vegetation_frac_dl", "veg_abs_diff"
] if c in audit.columns]

audit = audit[keep]
audit.to_csv(OUT, index=False)
print("Wrote:", OUT)
print(audit.head(10).to_string(index=False))