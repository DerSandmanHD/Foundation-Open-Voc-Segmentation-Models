import os
import pandas as pd
import pydicom

IMAGE_DIR = "stage_2_images"
CSV_PATH = "stage_2_train.csv"

df = pd.read_csv(CSV_PATH)

print("CSV first ImageIds:")
print(df["ImageId"].head(10).to_string(index=False))

first_file = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".dcm")])[0]
path = os.path.join(IMAGE_DIR, first_file)

print("\nFirst DICOM file:")
print(first_file)

ds = pydicom.dcmread(path, stop_before_pixels=True)

print("\nImportant DICOM fields:")
for key in [
    "SOPInstanceUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "PatientID",
    "AccessionNumber",
]:
    value = getattr(ds, key, None)
    print(key, "=", value)

print("\nAll UID-like fields:")
for elem in ds:
    if "UID" in elem.keyword or elem.VR == "UI":
        print(elem.keyword, "=", elem.value)