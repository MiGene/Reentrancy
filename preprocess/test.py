import pandas as pd

df1 = pd.read_csv('./dataset/real-time/fold_upsamped/3.csv')
df2 = pd.read_csv('./dataset/real-time/fill_fold_preprocessed/3.csv')
df3 = pd.read_csv('./dataset/real-time/train_preprocessed/realtime_bacon.csv')

print(len(df1.columns))
print(len(df2.columns))
print(len(df3.columns))