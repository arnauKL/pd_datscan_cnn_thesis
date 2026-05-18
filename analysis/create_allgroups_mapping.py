"""
Cohort codes (from rawdata/participants.json):
    1 = HC          (Healthy Control)
    2 = PD          (Parkinson's Disease)
    3 = SWEDD       (Scans Without Evidence of Dopaminergic Deficit)
    4 = Prodromal   (Prodromal PD)

Output CSV columns:
    path    ;; absolute path to the .nii.gz file
    label   ;; binary label used during CNN training (1=PD, 0=HC)
              for SWEDD and Prodromal this is set to -1 (inference only,
              no ground truth label for the classifier)
    cohort  ;; integer cohort code (1/2/3/4)
    group   ;; human-readable group name


Edit BASE_PATH and OUT_CSV below if your paths differ.
"""

import os
import glob
import pandas as pd

BASE_PATH   = "/home/data/PPMI"
RAWDATA     = os.path.join(BASE_PATH, "rawdata")
EXCEL_PATH  = os.path.join(BASE_PATH, "documents/PPMI_Curated_Data_Cut_Public_20240729.xlsx")
OUT_CSV     = "/home/akarel/src_tfg/data/ppmi_raw_allgroups_mapping.csv"
COHORT_NAMES = {
    1: "HC",
    2: "PD",
    3: "SWEDD",
    4: "Prodromal",
}

# Binary label for the CNN, only meaningful for HC (0) and PD (1)
# SWEDD and Prodromal get -1 since we're doing inference, not training
COHORT_LABELS = {
    1:  1,    # HC
    2:  0,    # PD
    3: -1,    # SWEDD  ;; inference only
    4: -1,    # Prodromal ;; inference only
}

# Load cohort labels from PPMI excel
print(f"Loading excel: {EXCEL_PATH}")
df_excel   = pd.read_excel(EXCEL_PATH)
labels_map = (df_excel[['PATNO', 'COHORT']]
              .drop_duplicates(subset='PATNO')
              .set_index('PATNO')['COHORT']
              .to_dict())
print(f"Patients in excel: {len(labels_map)}")
print(f"Cohort distribution in excel:")
cohort_counts = pd.Series(labels_map.values()).value_counts().sort_index()
for code, count in cohort_counts.items():
    print(f"  {code} ({COHORT_NAMES.get(code, 'unknown')}): {count}")

# Find all baseline DaTSCAN images
pattern = os.path.join(RAWDATA, "sub-PPMI*/ses-BL/spect/*_DaTSCAN.nii.gz")
print(f"\nSearching: {pattern}")
all_images = glob.glob(pattern)
print(f"Found {len(all_images)} baseline DaTSCAN images")

# Build mapping
data_list = []
skipped_no_metadata = []
skipped_unknown_cohort = []

for img_path in sorted(all_images):
    # Extract PATNO from path: sub-PPMI100001 => 100001
    parts  = img_path.split('/')
    sub_id = parts[-4]                         # 'sub-PPMI100001'
    patno  = int(sub_id.replace('sub-PPMI', ''))

    if patno not in labels_map:
        skipped_no_metadata.append(patno)
        continue

    cohort = labels_map[patno]

    if cohort not in COHORT_NAMES:
        skipped_unknown_cohort.append((patno, cohort))
        continue

    data_list.append({
        'path':   img_path,
        'label':  COHORT_LABELS[cohort],
        'cohort': cohort,
        'group':  COHORT_NAMES[cohort],
        'PATNO':  patno,
    })

# Summary
result_df = pd.DataFrame(data_list)

print(f"\nMapping result:")
print(f"  Total images mapped: {len(result_df)}")
print(f"  Skipped (no excel metadata): {len(skipped_no_metadata)}")
print(f"  Skipped (unknown cohort): {len(skipped_unknown_cohort)}")

print(f"\nPer-group breakdown:")
for cohort_code, group_name in COHORT_NAMES.items():
    n = (result_df['cohort'] == cohort_code).sum()
    print(f"  {group_name:12s} (cohort {cohort_code}): {n} images")

# Save 
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
result_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved to {OUT_CSV}")
print(f"Columns: {list(result_df.columns)}")
print(f"\nFirst few rows:")
print(result_df.head(8).to_string(index=False))