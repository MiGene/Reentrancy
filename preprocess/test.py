import glob
import os
import pandas as pd

CSV_FOLDER = './dataset/real-time/test_normal_preprocessed/'
CSV_ATK_FOLDER = './preprocess/temp_db/'

for file_path in glob.glob(os.path.join(CSV_FOLDER, "*.csv")):
    print(file_path)
    filename = os.path.basename(file_path)
    atk_filename = filename.replace('test_','')
    atk_df = pd.read_csv(file_path)


    atk_df.to_csv(f'{CSV_ATK_FOLDER}{atk_filename}',index=False)