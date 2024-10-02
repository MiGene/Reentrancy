import pandas as pd

DATA_PATH = "./dataset/real-time/fold/1.csv"
df = pd.read_csv(DATA_PATH)
columns = df.columns.tolist()
print(len(columns))

def remove_list_preserve_sequence(list,to_remove):
    new_list = [item for item in list 
                if item not in to_remove]
    return new_list

feature_with_sus = remove_list_preserve_sequence(df.columns,[
                        "transaction_hash",
                        "from_address",
                        "to_address",
                        "block_timestamp",
                        "add_feat_hash",
                    ])

feature =remove_list_preserve_sequence(df.columns,[
                "transaction_hash",
                "from_address",
                "to_address",
                "block_timestamp",
                "add_feat_hash",
                "is_sus",
            ])
# print(len(features))

def var():
    variables = {}

    #### feature ###

    variables["feature_with_sus"] = feature_with_sus
    variables["feature"] = feature
    # target feature
    remove_feature = [
        "contract_lifetime_block",
        "contract_lifetime_days",
        "sender_tx_count_call_contract",
        "sender_lifetime_days",
        "sender_lifetime_block",
        "receipt_cumulative_gas_used",
        "z_contract_lifetime_block",
        "z_contract_lifetime_days",
        "z_sender_lifetime_days",
        "z_sender_lifetime_block",
        # "contract_main_active_days",
        "sender_main_active_days",
        "contract_block_involved",
        # "sender_block_involved",
        # "z_distinct_was_called_in_sample",
        # "distinct_was_called_in_sample",
        # "contract_interact",
        # "sender_call_contract_tx_ratio",
        "contract_block_per_tx",
        # "contract_tx_count",
        # "z_receipt_gas_used",
        # "value",
        # "gas",
        # "gas_price",
        # "sender_active_day_ratio",
        # "distinct_sender_in_contract",
        # "distinct_sender_call_in_sample",
        # "z_value",
        # "contract_active_day_ratio",
        # "trace_involved_amt",
        # "sender_block_per_tx",
        # "distinct_sender_in_contract",
    ]
    variables["target_feature"] = remove_list_preserve_sequence(feature,remove_feature)
    variables["z_score_feature"] = [
        member for member in variables["target_feature"] if member.startswith("z_")
    ]

    variables["static_feature"] = remove_list_preserve_sequence(variables["target_feature"],variables["z_score_feature"])

    variables["static_feature_with_sus"] = remove_list_preserve_sequence(variables["feature_with_sus"],variables["z_score_feature"])

    # best feature
    variables["best_feature"] = []

    return variables


# print(len(var()["target_feature"]))