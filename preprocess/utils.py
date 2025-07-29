import pandas as pd
from scipy.stats import zscore

Z_SCORE_COLS =  ['trace_involved_amt','contract_block_involved','contract_tx_count','contract_main_active_days',
                'sender_block_involved','sender_tx_count','sender_main_active_days','contract_interact',
                'sender_tx_count_call_contract','sender_days_call_contract','trace_amt','distinct_sender_in_contract',
                'contract_lifetime_days','contract_lifetime_block','distinct_contract_sender_called',
                'sender_lifetime_days','sender_lifetime_block','contract_involved_amt','max_breadth','depth',
                'distinct_was_called_in_sample','distinct_sender_call_in_sample','gas','gas_price',
                'receipt_cumulative_gas_used','receipt_gas_used','value','nonce']

SUS_FILE = '../dataset/sus_tx.csv'  # Replace with the actual file path

def find_nulls(df):
    null_count = df.isnull().sum()
    null_count = null_count[null_count>0]
    print(null_count)
    print('=================')

def impute_depth_and_max_breadth(df):
    df['max_breadth'] = df['max_breadth']+1
    df['max_breadth'] = df['max_breadth'].fillna(0)
    df['depth'] = df['depth'].fillna(0)
    return df

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

def add_sus_col(df,attack_transactions):

    def set_value_based_on_condition(row):
        if (row['transaction_hash'] in attack_transactions):
            return 1
        else:
            return 0
        
    df['is_sus'] = df.apply(set_value_based_on_condition, axis=1)

    sus_data = df[df['is_sus'] == 1]
    not_sus_data = df[df['is_sus'] == 0]

    print(len(sus_data))
    print(len(not_sus_data))

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
    result_df = result_df.drop_duplicates()
    return result_df

def add_cols(base_df,add_df,join_by,cols_to_add):

    add_df = add_df[cols_to_add+[join_by]]

    merged_df = pd.merge(base_df, add_df, on=join_by, how='left')

    return merged_df

def recalculate_interactions(transaction_info):
    # Ensure block_timestamp is in datetime format for sorting
    transaction_info['block_timestamp'] = pd.to_datetime(transaction_info['block_timestamp'])
    
    # Sort by 'to_address' and 'block_timestamp'
    transaction_info = transaction_info.sort_values(by=['to_address', 'block_timestamp']).reset_index(drop=True)

    # Create a new column 'contract_interact' which mimics ROW_NUMBER() OVER (PARTITION BY to_address ORDER BY block_timestamp)
    transaction_info['contract_interact'] = transaction_info.groupby('to_address').cumcount() + 1

    return transaction_info

def time_slice_df(df, num_rows=50, time_col='block_timestamp', sus_col='is_sus', how='last'):
    sus_df = pd.read_csv(SUS_FILE)
    sus_transactions = sus_df['sus_tx']
    sus_transactions = set(sus_transactions)
    df = add_sus_col(df,sus_transactions)
    # Convert 'time_col' to datetime format
    df[time_col] = pd.to_datetime(df[time_col])
    df = impute_depth_and_max_breadth(df)
    df['value'] = df['value'].astype('float32')
    # Sort by 'time_col' in descending order (from last to first)
    df = df.sort_values(by=time_col, ascending=False).reset_index()

    # Find the index of the first row where 'is_sus' == 1
    sus_index = df[df[sus_col] == 1].index[0] if not df[df[sus_col] == 1].empty else 0
    # print(df.iloc[sus_index])

    # Slice the DataFrame starting from the 'sus_index'
    df_sliced = df.iloc[sus_index:sus_index + num_rows].reset_index(drop=True)

    df_interaction = recalculate_interactions(df_sliced)

    z_score_df = add_z_score(df_interaction,Z_SCORE_COLS)

    ratio_df = add_ratio_features(z_score_df)


    return ratio_df
