import os
import numpy as np
import pandas as pd
import librosa
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows

# Parameters
folder_path = input("Enter the folder path containing .wav files: ")
output_excel_path = input("Enter the output Excel file path (e.g., results.xlsx): ")

bin_percentage = 20  # 20% bins
voice_min_freq = 75  # Minimum frequency for both genders
voice_max_freq = 400  # Maximum frequency for both genders
frame_length = 2048  # Frame length for pitch analysis
hop_length = int(0.01 * 44100)  # Match Praat's timing (10 ms)
bias_correction = 4  # Adjustment to align pitches with Praats analysis

# Define fixed percentage bins (this fixes errors with double values for same bin as well as
# making sure to include 100% bin)
fixed_percentage_bins = np.append(np.arange(0, 100, bin_percentage), 100)

def process_audio(file_path):
    """Extracts pitch data from an audio file and returns a DataFrame with percentage-based bins."""
    print(f"Processing file: {file_path}")
    y, sr = librosa.load(file_path, sr=None)
    total_duration = len(y) / sr  # Calculate total duration in seconds

    pitch_data = {'Percentage': [], 'Average_Frequency': []}

    for percent in fixed_percentage_bins:
        actual_percent = percent  # Helps percetange label

        start_time = (percent / 100) * total_duration
        end_time = ((percent + bin_percentage) / 100) * total_duration if percent < 100 else total_duration

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        bin_samples = y[start_sample:end_sample]

        try:
            pitches = librosa.pyin(bin_samples, fmin=voice_min_freq, fmax=voice_max_freq,
                                   frame_length=frame_length, hop_length=hop_length, sr=sr)
            
            if pitches is not None:
                pitches = np.array(pitches)
                valid_pitches = pitches[~np.isnan(pitches)]
                valid_pitches = valid_pitches[(valid_pitches >= voice_min_freq) & (valid_pitches <= voice_max_freq)]
                valid_pitches = gaussian_filter1d(valid_pitches, sigma=0.1)  # Smooth data filters out any backgroud minor noises

                if len(valid_pitches) > 0:
                    avg_bin_freq = np.median(valid_pitches) + bias_correction
                    pitch_data['Percentage'].append(actual_percent)
                    pitch_data['Average_Frequency'].append(avg_bin_freq)
                else:
                    # If 100% has no valid data, use the previous valid value
                    pitch_data['Percentage'].append(actual_percent)
                    pitch_data['Average_Frequency'].append(pitch_data['Average_Frequency'][-1] if pitch_data['Average_Frequency'] else np.nan)
            else:
                pitch_data['Percentage'].append(actual_percent)
                pitch_data['Average_Frequency'].append(np.nan)

        except Exception as e:
            print(f"Error processing bin at {actual_percent}%: {e}")
            pitch_data['Percentage'].append(actual_percent)
            pitch_data['Average_Frequency'].append(np.nan)

    return pd.DataFrame(pitch_data)

# Process all subjects
all_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
subject_files = {}

for file in all_files:
    subject_id = "_".join(file.split("_")[:2])
    if subject_id not in subject_files:
        subject_files[subject_id] = {"H": None, "L": None}
    if "_H" in file:
        subject_files[subject_id]["H"] = file
    elif "_L" in file:
        subject_files[subject_id]["L"] = file

combined_h_df = pd.DataFrame(index=fixed_percentage_bins)
combined_l_df = pd.DataFrame(index=fixed_percentage_bins)
all_data = {}

for subject_id, files in subject_files.items():
    h_file = files["H"]
    l_file = files["L"]

    h_data = process_audio(os.path.join(folder_path, h_file)) if h_file else None
    l_data = process_audio(os.path.join(folder_path, l_file)) if l_file else None

    if h_data is not None:
        h_data = h_data.set_index("Percentage").reindex(fixed_percentage_bins, fill_value=np.nan)
        all_data[f"{subject_id}_H"] = h_data["Average_Frequency"]
        combined_h_df = pd.concat([combined_h_df, h_data], axis=1)

    if l_data is not None:
        l_data = l_data.set_index("Percentage").reindex(fixed_percentage_bins, fill_value=np.nan)
        all_data[f"{subject_id}_L"] = l_data["Average_Frequency"]
        combined_l_df = pd.concat([combined_l_df, l_data], axis=1)

average_h = combined_h_df.mean(axis=1).rename("H_Avg")
average_l = combined_l_df.mean(axis=1).rename("L_Avg")

all_data["H_Avg"] = average_h
all_data["L_Avg"] = average_l
final_df = pd.DataFrame(all_data)
final_df.reset_index(inplace=True)
final_df.rename(columns={"index": "Percentage"}, inplace=True)

wb = Workbook()
ws = wb.active
ws.title = "All Data"

for row in dataframe_to_rows(final_df, index=False, header=True):
    ws.append(row)

# Ensure chart is included in the file
plt.plot(average_h.index, average_h.values, label="H", marker='o', color='red')
plt.plot(average_l.index, average_l.values, label="L", marker='o', color='blue')
plt.savefig("group_chart.png")
ws.add_image(Image("group_chart.png"), "C10")

wb.save(output_excel_path)
print(f"Results saved to {output_excel_path}")