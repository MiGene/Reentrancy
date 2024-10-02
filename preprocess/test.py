import glob
import os
import pandas as pd

CSV_FOLDER = './dataset/real-time/test_after_attack/'
CSV_ATK_FOLDER = './dataset/real-time/test_attack/'

for file_path in glob.glob(os.path.join(CSV_FOLDER, "*.csv")):
    filename = os.path.basename(file_path)
    atk_filename = filename.replace('after_attack','realtime')
    atk_path = f'{CSV_ATK_FOLDER}{atk_filename}'
    atk_df = pd.read_csv(atk_path)

    df = pd.read_csv(file_path)
    print(len(df))
    atk_tx = atk_df['transaction_hash'].to_list()
    filtered_df = df[~df['transaction_hash'].isin(atk_tx)]

    filtered_df.to_csv(file_path,index=False)