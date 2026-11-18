import pandas as pd, glob
for f in glob.glob("dataset/*/*.csv"):
    if ".camonly" in f:
        continue
    df = pd.read_csv(f)
    df[["fwd", "side", "mpuCal"]] = 0
    df.to_csv(f.replace(".take", ".camonly"), index=False)