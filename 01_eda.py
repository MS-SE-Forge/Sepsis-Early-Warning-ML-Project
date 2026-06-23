"""
Section: Dataset / EDA

This script is responsible for loading the raw PhysioNet 2019 Sepsis challenge data
and performing foundational Exploratory Data Analysis (EDA). The EDA focuses heavily
on understanding class imbalance, missingness patterns, and the clinical timeline 
of sepsis onset, which are critical constraints for building an early-warning model.
"""
import pandas as pd
import numpy as np
import glob
import os

# Define the paths to the PhysioNet dataset directories
DATA_DIRS = [
      "physionet.org/files/challenge-2019/1.0.0/training/training_setA",
      "physionet.org/files/challenge-2019/1.0.0/training/training_setB",
  ] # adjust paths as needed

def load_all_patients(data_dirs):
    """
    Load every .psv (pipe-separated values) file from the provided directories.
    Each file corresponds to a single patient's ICU stay.
    
    Args:
        data_dirs (list): List of directory paths containing the .psv files.
        
    Returns:
        pd.DataFrame: A single concatenated DataFrame containing all patient hours,
                      tagged with 'patient_id' and 'hospital_source'.
    """
    frames = []
    for d in data_dirs:
        # Determine the hospital source (A or B) based on the directory name
        source = "A" if "setA" in d else "B"
        
        # Find all .psv files in the directory
        files = sorted(glob.glob(os.path.join(d, "*.psv")))
        
        for fp in files:
            # Extract the patient ID from the filename (e.g., 'p000001.psv' -> 'p000001')
            pid = os.path.basename(fp).replace(".psv", "")
            
            # Read the pipe-separated data into a pandas DataFrame
            df = pd.read_csv(fp, sep="|")
            
            # Tag the dataframe with the patient ID and source for later grouping
            df["patient_id"] = pid
            df["hospital_source"] = source
            
            frames.append(df)
            
    # Concatenate all individual patient dataframes into one large dataframe
    return pd.concat(frames, ignore_index=True)

def summarize(df):
    """
    Compute and print summary statistics for the dataset, including size, 
    label balance (at both the patient and row level), and missingness.
    
    Args:
        df (pd.DataFrame): The full loaded dataset.
        
    Returns:
        tuple: (patient_label, miss, los)
               - patient_label (pd.Series): Boolean series indicating if a patient ever got sepsis.
               - miss (pd.Series): Fraction of missing values per column.
               - los (pd.Series): Length of stay (in hours) for each patient.
    """
    print(f"Total rows (patient-hours): {len(df):,}")
    print(f"Total patients: {df['patient_id'].nunique():,}")
    print(f"Hospital sources: {df['hospital_source'].value_counts().to_dict()}")

    # Patient-level label: A patient is considered 'septic' if ANY hour in their stay has SepsisLabel == 1
    patient_label = df.groupby("patient_id")["SepsisLabel"].max()
    print(f"\nPatients who develop sepsis at some point: "
          f"{patient_label.sum():,} / {len(patient_label):,} "
          f"({patient_label.mean()*100:.2f}%)")

    # Row-level label: Shows the extreme imbalance at the hourly level
    print(f"\nRow-level label balance (hourly): "
          f"{df['SepsisLabel'].mean()*100:.2f}% of all hours are labeled septic")

    # Missingness - This is the single most important EDA fact for this clinical dataset.
    # Labs are ordered sparingly, meaning most hours will have NaN for lab values.
    # We drop metadata columns before calculating the missing fraction.
    miss = df.drop(columns=["patient_id", "hospital_source"]).isna().mean().sort_values(ascending=False)
    print("\nTop 15 most-missing columns (fraction missing):")
    print(miss.head(15))

    # Calculate the maximum ICU length of stay (ICULOS) for each patient
    los = df.groupby("patient_id")["ICULOS"].max()
    print(f"\nICU stay length (hours): median={los.median():.0f}, "
          f"mean={los.mean():.1f}, min={los.min()}, max={los.max()}")

    return patient_label, miss, los

def time_to_onset_stats(df):
    """
    Calculate the hour of sepsis onset for patients who develop sepsis.
    
    This is critical for this project: how many hours of pre-onset data exist
    per septic patient? This determines how much "lead time" is even
    possible to predict, and shapes our windowing strategy. If onset happens 
    at hour 2, an early warning system cannot practically give a 6-hour warning.
    
    Args:
        df (pd.DataFrame): The full loaded dataset.
        
    Returns:
        pd.Series: The specific ICULOS (ICU Length of Stay hour) where sepsis first occurs per patient.
    """
    # Isolate the IDs of patients who have at least one septic hour
    septic_ids = df.loc[df["SepsisLabel"] == 1, "patient_id"].unique()
    
    onset_hours = []
    for pid in septic_ids:
        # Filter for the specific patient and ensure the timeline is sorted sequentially
        pdf = df[df["patient_id"] == pid].sort_values("ICULOS")
        
        # Find the very first hour (ICULOS) where the SepsisLabel becomes 1
        onset_row = pdf.loc[pdf["SepsisLabel"] == 1, "ICULOS"].iloc[0]
        onset_hours.append(onset_row)
        
    onset_hours = pd.Series(onset_hours)
    print("\nAmong septic patients, hour of sepsis onset (ICULOS at first label=1):")
    print(onset_hours.describe())
    
    return onset_hours

if __name__ == "__main__":
    # Standard entry point for executing the EDA script standalone
    df = load_all_patients(DATA_DIRS)
    patient_label, miss, los = summarize(df)
    onset_hours = time_to_onset_stats(df)
