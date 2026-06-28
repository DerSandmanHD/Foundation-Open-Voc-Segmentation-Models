import pandas as pd

# CSV einlesen
bbox_df = pd.read_csv('BBox_List_2017.csv')

# Zeigt dir die Spalten: Image Index, Finding Label, Bbox x, y, w, h
print(bbox_df.head())

# Such dir die ersten 5 eindeutigen Bildnamen (Image Index) heraus
sample_images = bbox_df['Image Index'].unique()[:5]
print("Diese Bilder lädst du dir nun einzeln für SAM herunter:", sample_images)