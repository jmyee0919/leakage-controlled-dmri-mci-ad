
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os

def apply_filters(df_snap=None, df_delta=None):
    def filter_snapshot(df):
        df["hippo"] = df["st29sv"]
        df["ventricle"] = df["st37sv"]

        mask = (
            (df["hippo"] >= 2200) & (df["hippo"] <= 4200) &
            (df["ventricle"] >= 8000) & (df["ventricle"] <= 75000)
        )
        return df[mask].drop(columns=["hippo", "ventricle"])

    def filter_delta(df):
        df["hippo_delta"] = (df["st29sv_follow"] - df["st29sv_base"]) / df["st29sv_base"] / (df["interval_months"] / 12) * 100
        df["ventricle_delta"] = (df["st37sv_follow"] - df["st37sv_base"]) / df["st37sv_base"] / (df["interval_months"] / 12) * 100

        mask = (
            (df["hippo_delta"] <= 0) &
            (df["ventricle_delta"] >= 0)
        )
        return df[mask].drop(columns=["hippo_delta", "ventricle_delta"])

    results = {}
    if df_snap is not None:
        results["snapshot_filtered"] = filter_snapshot(df_snap)
    if df_delta is not None:
        results["delta_filtered"] = filter_delta(df_delta)
    return results

def load_and_process():
    snap_file = filedialog.askopenfilename(title="Select Snapshot CSV")
    delta_file = filedialog.askopenfilename(title="Select ΔMRI CSV")

    if not snap_file or not delta_file:
        messagebox.showwarning("Missing file", "Both CSV files are required.")
        return

    try:
        df_snap = pd.read_csv(snap_file)
        df_delta = pd.read_csv(delta_file)

        outputs = apply_filters(df_snap, df_delta)

        base_dir = os.path.dirname(snap_file)
        snap_out = os.path.join(base_dir, "filtered_snapshot.csv")
        delta_out = os.path.join(base_dir, "filtered_mridelta.csv")

        outputs["snapshot_filtered"].to_csv(snap_out, index=False)
        outputs["delta_filtered"].to_csv(delta_out, index=False)

        messagebox.showinfo("Done", f"Filtered files saved:\n{snap_out}\n{delta_out}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("ΔMRI Snapshot Filter (Hippocampus + Ventricle Only)")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

label = tk.Label(frame, text="Select snapshot.csv and ΔMRI.csv to filter (entorhinal excluded):")
label.pack(pady=(0, 10))

btn = tk.Button(frame, text="Load & Filter CSVs", command=load_and_process)
btn.pack()

root.mainloop()
