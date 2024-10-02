import pandas as pd
from scipy.stats import zscore

def find_nulls(df):
    null_count = df.isnull().sum()
    null_count = null_count[null_count>0]
    print(null_count)
    print('=================')

def add_z_score(df,cols_to_calculate):
    # display(df.info())
    for column in cols_to_calculate:
        z_score_column_name = f'z_{column}'
        df[z_score_column_name] = zscore(df[column])
        df.loc[df[z_score_column_name].isnull(),z_score_column_name] = 0
    return df

def add_ratio_features(df):
    df['involved_trace_ratio'] = df['trace_involved_amt'] / df['trace_amt']
    df['contract_active_day_ratio'] = df['contract_main_active_days'] / (df['contract_lifetime_days']+1)
    df['tx_count_per_distinct_caller'] = df['contract_tx_count'] / df['distinct_sender_in_contract']
    df['contract_block_ratio'] = df['contract_block_involved'] / (df['contract_lifetime_block']+1)
    df['sender_active_day_ratio'] = df['sender_main_active_days'] / (df['sender_lifetime_days']+1)
    df['sender_tx_count_per_contract'] = df['sender_tx_count'] / df['distinct_contract_sender_called'] 
    df['sender_block_ratio'] = df['sender_block_involved'] / (df['sender_lifetime_block']+1)
    df['tx_sender_call_contract'] = df['sender_tx_count_call_contract'] / df['sender_tx_count']
    df['sender_call_contract_tx_ratio'] = df['sender_tx_count_call_contract'] / df['contract_tx_count']
    df['sender_tx_count_call_contract_per_days'] = df['sender_tx_count_call_contract'] / df['sender_days_call_contract']
    df['sender_block_per_tx'] = df['sender_block_involved']/df['sender_tx_count']
    df['contract_block_per_tx'] = df['contract_block_involved']/df['contract_tx_count']
    df['sender_call_contract_day_ratio'] = df['sender_days_call_contract'] / df['contract_main_active_days']
    df['sender_block_per_tx'] = df['sender_block_involved']/df['sender_tx_count']
    df['contract_block_per_tx'] = df['contract_block_involved']/df['contract_tx_count']
    return df

def merge_files(folder_path,files_list,prefix=''):
    sample_file_path = f'{folder_path}/{prefix}{files_list[0]}.csv'
    sample_df = pd.read_csv(sample_file_path)
    result_df = pd.DataFrame(columns=sample_df.columns)

    # Loop through each CSV file
    for file in files_list:
        file_path = f'{folder_path}/{prefix}{file}.csv'
        # Read the CSV file
        df = pd.read_csv(file_path)
        # Extract values from column A and append them to the result DataFrame
        result_df = pd.concat([result_df,df], ignore_index=True)

    return result_df

def add_cols(base_df,add_df,join_by,cols_to_add):

    add_df = add_df[cols_to_add+[join_by]]

    merged_df = pd.merge(base_df, add_df, on=join_by, how='left')

    return merged_df
