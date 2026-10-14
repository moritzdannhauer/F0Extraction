#!/usr/bin/env python3
#Median F0 per 20% bin of each recording, plus H and L group averages.

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

# Parameters
# THE folder_path NEECS TO ADJUSTED TO THE MALE WAV FILES FOLDER
folder_path = "/path/that/needs/to/be/updated/"
# THE output_excel_path  NEECS TO ADJUSTED TO THE MALE WAV FILES FOLDER
output_excel_path = "/path/that/needs/to/be/updated/F0_Results_Male_New.xlsx"

bin_percentage = 20    # 20% bins
voice_min_freq = 150   # Minimum female voice frequency (Hz)
voice_max_freq = 400   # Maximum female voice frequency (Hz)
frame_length = 2048    # Frame length for pitch analysis (samples)
hop_seconds = 0.01     # 10 ms step, like Praat; converted to samples per file

# Bin labels are the bin start: 0 = 0-20%, 20 = 20-40%, ... 80 = 80-100%
fixed_percentage_bins = np.arange(0, 100, bin_percentage)


def process_audio(file_path):
    """Return a Series of median F0 (Hz) per percentage bin of the recording."""
    print(f"Processing file: {file_path}")
    y, sr = librosa.load(file_path, sr=None)
    hop_length = int(round(hop_seconds * sr))   # 10 ms at this file's sample rate
    total_duration = len(y) / sr

    # Run pYIN once on the whole file (avoids edge effects at bin boundaries).
    # f0 is NaN wherever pYIN judges the frame unvoiced.
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=voice_min_freq, fmax=voice_max_freq, sr=sr,
        frame_length=frame_length, hop_length=hop_length,
    )

    # Position of each frame as a percentage of the recording
    frame_percent = librosa.times_like(f0, sr=sr, hop_length=hop_length) / total_duration * 100

    medians = []
    for start in fixed_percentage_bins:
        end = start + bin_percentage
        if end < 100:
            in_bin = (frame_percent >= start) & (frame_percent < end)
        else:
            in_bin = frame_percent >= start   # last bin runs to the end of the file
        values = f0[in_bin & ~np.isnan(f0)]
        if values.size > 0:
            medians.append(np.median(values))
        else:
            print(f"  No voiced frames in bin {start}%; left empty")
            medians.append(np.nan)

    return pd.Series(medians, index=fixed_percentage_bins)


# Group files by subject. Expected names like P_16_H.wav / P_02_L.wav:
# the condition is a separate "H" or "L" part, the subject ID is the rest.
subject_files = {}
for file in sorted(os.listdir(folder_path)):
    stem, ext = os.path.splitext(file)
    if ext.lower() != ".wav":
        continue
    parts = stem.split("_")
    if "H" in parts:
        condition = "H"
    elif "L" in parts:
        condition = "L"
    else:
        print(f"Skipping (no _H or _L in name): {file}")
        continue
    subject_id = "_".join(p for p in parts if p not in ("H", "L"))
    subject_files.setdefault(subject_id, {})[condition] = file

print(f"Found {len(subject_files)} subjects:")
for subject_id, files in sorted(subject_files.items()):
    print(f"  {subject_id}: H = {files.get('H')}, L = {files.get('L')}")

# Analyze every file
results = {}
for subject_id, files in sorted(subject_files.items()):
    for condition in ("H", "L"):
        if condition in files:
            results[f"{subject_id}_{condition}"] = process_audio(
                os.path.join(folder_path, files[condition]))

final_df = pd.DataFrame(results, index=fixed_percentage_bins)
h_columns = [c for c in results if c.endswith("_H")]
l_columns = [c for c in results if c.endswith("_L")]
final_df["H_Avg"] = final_df[h_columns].mean(axis=1)   # NaN bins are skipped
final_df["L_Avg"] = final_df[l_columns].mean(axis=1)
final_df["H_N"] = final_df[h_columns].notna().sum(axis=1)  # subjects per bin
final_df["L_N"] = final_df[l_columns].notna().sum(axis=1)
final_df.index.name = "Percentage"
final_df = final_df.reset_index()

# Write to Excel (NaN -> empty cell, since Excel can't store NaN)
wb = Workbook()
ws = wb.active
ws.title = "All Data"
excel_df = final_df.astype(object).where(final_df.notna(), None)
for row in dataframe_to_rows(excel_df, index=False, header=True):
    ws.append(row)

# Group chart, in its own figure so earlier runs in the same session can't leak in
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(final_df["Percentage"], final_df["H_Avg"], label="H", marker="o", color="red")
ax.plot(final_df["Percentage"], final_df["L_Avg"], label="L", marker="o", color="blue")
ax.set_xlabel("Bin start (% of recording)")
ax.set_ylabel("Median F0 (Hz)")
ax.set_xticks(fixed_percentage_bins)
ax.legend()
fig.tight_layout()

base_path = os.path.splitext(output_excel_path)[0]

# High-resolution TIFF for publication (LZW compression keeps the file size down)
tiff_path = base_path + "_chart.tif"
fig.savefig(tiff_path, dpi=330, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
print(f"Chart saved to {tiff_path}")

# PNG copy for embedding in Excel (Excel doesn't display TIFF reliably)
chart_path = base_path + "_chart.png"
fig.savefig(chart_path, dpi=100)
plt.close(fig)

# Place the chart two columns right of the data so it doesn't cover it
ws.add_image(Image(chart_path), f"{get_column_letter(final_df.shape[1] + 2)}2")

wb.save(output_excel_path)
print(f"Results saved to {output_excel_path}")
