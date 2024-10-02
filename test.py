import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import (
    MinMaxScaler,
)
from sklearn.pipeline import Pipeline
import var_setup


np.random.seed(42)

TRAIN_DATA_PATH = "./dataset/real-time/fold_upsamped_smote/"
TEST_DATA_PATH = "./dataset/real-time/fold/"
PERFORMANCE_PATH = "./result/performance.csv"
FEATURE_IMPORTANCE_PATH = "./result/feature_importance.csv"


def random_forest():
    # return ""

    static_feature = var_setup.var()["static_feature"]
    static_feature = var_setup.remove_list_preserve_sequence(static_feature,['transaction_type', 'rn'])
    z_features = var_setup.var()["z_score_feature"]

    # static_feature = ['trace_involved_amt', 'gas', 'gas_price', 'receipt_cumulative_gas_used', 
    #               'receipt_gas_used', 'nonce', 'contract_tx_count', 'contract_main_active_days', 
    #               'sender_block_involved','sender_block_per_tx','sender_tx_count_call_contract_per_days',
    #               'sender_call_contract_tx_ratio', 'distinct_sender_call_in_sample','contract_involved_amt',
    #               'max_breadth','depth']


    # z_features = ['z_gas_price', 'z_receipt_cumulative_gas_used', 'z_receipt_gas_used',
    #             'z_nonce', 'z_contract_block_involved', 'z_contract_tx_count', 'z_contract_main_active_days',
    #             'z_sender_block_involved', 'z_sender_tx_count', 'z_sender_main_active_days',
    #             'z_contract_interact',]
    features = static_feature + z_features

    # print(features)
    # features = list(set(features) - set(['transaction_type', 'rn']))
    # print(features)

    fold_files = ["1.csv", "2.csv", "3.csv", "4.csv", "5.csv"]
    matching_percentage = []
    f1_percentage = []

    num_pipeline = Pipeline(
        steps=[
            ("scale", MinMaxScaler()),
        ]
    )

    col_trans = ColumnTransformer(
        transformers=[
            ("num_pipeline", num_pipeline, static_feature),
            ("passthrough", "passthrough", z_features),
        ],
        n_jobs=-1,
    )

    feature_importance_df = pd.DataFrame(columns=features)

    performance_df = pd.DataFrame(
        columns=[
            "Fold",
            "Total Values",
            "Matching",
            "Matching Percentage",
            "Actual Attack",
            "Predicted",
            "True Positive",
            "False Positive",
            "False Negative",
            "F1 Score",
        ]
    )

    temp = pd.read_csv(TRAIN_DATA_PATH + "1.csv")

    for i in range(len(fold_files)):
        file_path_train = TRAIN_DATA_PATH
        file_path_test = TEST_DATA_PATH

        train_data = pd.DataFrame(columns=features+['is_sus'])
        # print(train_data.columns)
        print('arranging fold')
        for j in range(len(fold_files)):
            if j == i:continue
            df = pd.read_csv(file_path_train + fold_files[j])
            df = df[features+['is_sus']]
            # print(df.columns)
            train_data = pd.concat([train_data, df], ignore_index=True)
        print('done')
        train_data = train_data.sort_index()
        X = train_data[features]
        # X = X.sort_index()
        y = train_data.is_sus.astype(int)
        # y = y.sort_index()

        rf_model = RandomForestClassifier(
            n_estimators=50,  # 250
            max_depth=7,  # 5
            min_samples_leaf=50,
            random_state=42,
            n_jobs=-1,
        )  # 50
        
        model_pipeline = Pipeline(steps=[("col_trans", col_trans), ("model", rf_model)])
        print('fitting')
        model_pipeline.fit(X, y)
        print('done')
        final_estimator = model_pipeline.steps[-1][1]
        if isinstance(final_estimator, RandomForestClassifier):
            feature_importances = model_pipeline.named_steps[
                "model"
            ].feature_importances_
            feature_importance_df = feature_importance_df._append(
                pd.DataFrame([feature_importances], columns=features),
                ignore_index=True,
            )

        test_data = pd.read_csv(file_path_test + fold_files[i])
        test_data = test_data.sort_index()
        test_X = test_data[features]
        # test_X = test_X.sort_index()
        test_y = test_data.is_sus.astype(int)
        # test_y = test_y.sort_index()
        predict_y = model_pipeline.predict(test_X)

        matching_values = np.sum(np.array(test_y) == np.array(predict_y))
        total_samples = len(test_y)
        percentage_matching = (matching_values / total_samples) * 100
        matching_percentage.append(percentage_matching)
        actualAttack = sum(test_y[test_y == 1])
        predicted = sum(predict_y[predict_y == 1])
        TP = sum((test_y == 1) & (predict_y == 1))
        FP = sum((test_y == 0) & (predict_y == 1))
        FN = sum((test_y == 1) & (predict_y == 0))

        print(f"************** Fold {i+1} **************")
        print("--- Overall Performance ---")
        print("Total Values :", total_samples)
        print("Matching :", matching_values)
        print("Matching percentage :", percentage_matching)
        print("--- Caught Performance ---")
        print("Actual Attack :", actualAttack)
        print("Predicted :", predicted)
        print("True Positive(attack caught) :", TP)
        print("False Positive :", FP)
        print("False Negative :", FN)
        f1Score = f1_score(test_y, predict_y)

        f1_percentage.append(f1Score)
        print("F1 score : ", f1Score)
        print("\n")

        performance_df = performance_df._append(
            {
                "Fold": i,
                "Total Values": total_samples,
                "Matching": matching_values,
                "Matching Percentage": percentage_matching,
                "Actual Attack": actualAttack,
                "Predicted": predicted,
                "True Positive": TP,
                "False Positive": FP,
                "False Negative": FN,
                "F1 Score": f1Score,
            },
            ignore_index=True,
        )

    performance_df.to_csv(PERFORMANCE_PATH, index=False)
    average_result = np.mean(matching_percentage)
    average_result2 = np.mean(f1_percentage)

    print(f"Average Result Across Folds: {average_result}")
    print(f"Average Result Across Folds: {average_result2}")
    feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

random_forest()