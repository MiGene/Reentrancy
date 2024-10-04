import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.pipeline import FunctionTransformer, Pipeline

# Load data
fold_files = ['1.csv','2.csv','3.csv','4.csv','5.csv']
caught_pf = [0,0,0,0]
matching_percentage = []
f1_percentage = []

static_feature = ['trace_involved_amt', 'gas', 'gas_price', 'receipt_cumulative_gas_used', 
                  'receipt_gas_used', 'nonce', 'contract_tx_count', 'contract_main_active_days', 
                  'sender_block_involved','sender_block_per_tx','sender_tx_count_call_contract_per_days',
                  'sender_call_contract_tx_ratio', 'distinct_sender_call_in_sample','contract_involved_amt',
                  'max_breadth','depth']


# z_features = ['z_gas_price',      , 'z_receipt_gas_used',
#               'z_nonce', 'z_c_block_involved', 'z_c_tx_count', 'z_c_main_active_days',
#               'z_s_block_involved', 'z_s_tx_count', 'z_s_block_per_tx', 'z_s_main_active_days',
#               'z_c_interact',]

z_features = ['z_gas_price', 'z_receipt_cumulative_gas_used', 'z_receipt_gas_used',
              'z_nonce', 'z_contract_block_involved', 'z_contract_tx_count', 'z_contract_main_active_days',
              'z_sender_block_involved', 'z_sender_tx_count', 'z_sender_main_active_days',
              'z_contract_interact',]


features = static_feature+z_features
# Defining the custom transformer for applying log transformation
def apply_log(X):
    X = X.astype(np.float32)
    return np.log1p(X)

# num_pipeline = Pipeline(steps=[
#     ('scale', MinMaxScaler())])

num_pipeline = Pipeline(steps=[
    ('scale', MinMaxScaler()),
])

col_trans = ColumnTransformer(transformers=[
    ('num_pipeline', num_pipeline, static_feature)
    ,('passthrough', 'passthrough', z_features)],
    n_jobs=1)

# Initialize a DataFrame to store feature importances
feature_importance_df = pd.DataFrame(columns=features)

# Create a DataFrame to store performance results
performance_df = pd.DataFrame(columns=['Fold', 'Total Values', 'Matching', 'Matching Percentage',
                                       'Actual Attack', 'Predicted', 'True Positive',
                                       'False Positive', 'False Negative', 'F1 Score'])

temp = pd.read_csv('./50_dataset/real-time/fold_upsamped_smote/1.csv')
# Perform 5-fold cross-validation
for i in range(len(fold_files)):
    # Load the data for the current fold
    file_path_train = './50_dataset/real-time/fold_upsamped_smote/' #! Change
    file_path_test = './50_dataset/real-time/fold/' #! Change

    # Creating Test Data
    # Initialize an empty DataFrame to store the combined data
    train_data = pd.DataFrame(columns=temp.columns)

    # Step 2: Loop through each CSV file and append its data to the combined DataFrame
    print('arranging fold')
    for j in range(len(fold_files)):
        if (j==i): continue
        df = pd.read_csv(file_path_train + fold_files[j])
        # print(df.columns)
        train_data = pd.concat([train_data, df], ignore_index=True)
    print('done')
    # Split the fold data into features (X) and labels (y)
    X = train_data[features]
    y = train_data.is_sus.astype(int)

    # rf_model = RandomForestClassifier(random_state=42,
    #                                   n_estimators=10,
    #                                   max_depth=5,
    #                                   min_samples_leaf=50)

    rf_model = RandomForestClassifier(
            n_estimators=9,  # 250
            max_depth=7,  # 5
            min_samples_leaf=25,
            random_state=42,
            n_jobs=1,
        )  # 50
    model_pipeline = Pipeline(steps=[
        ('col_trans', col_trans),
        ('model', rf_model)
    ])
    print('fitting')
    model_pipeline.fit(X, y)
    print('done')
    #?----------------------------------------------------------------------------------
    # Access the final estimator
    final_estimator = model_pipeline.steps[-1][1]

# Extract feature importance
    if isinstance(final_estimator, RandomForestClassifier):
        feature_importances = model_pipeline.named_steps['model'].feature_importances_
        feature_importance_df = feature_importance_df._append(pd.DataFrame([feature_importances], columns=features), ignore_index=True)


    # # Sort feature importances in descending order
    # indices = np.argsort(feature_importances)[::-1]

    # # Print the feature ranking
    # print("Feature ranking:")
    # for f in range(len(features)):
    #     print(f"{features[indices[f]]}: {feature_importances[indices[f]]}")

    # # Plot the feature importances
    # plt.figure(figsize=(10, 6))
    # plt.title("Feature importances")
    # plt.bar(range(len(features)), feature_importances[indices], align="center")
    # plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
    # plt.tight_layout()
    # plt.show()
    #?----------------------------------------------------------------------------------

    test_data = pd.read_csv(file_path_test+fold_files[i])
    test_X = test_data[features]
    test_y = test_data.is_sus.astype(int)
    # test_X = pd.get_dummies(test_X, columns=dummies)

    predict_y = model_pipeline.predict(test_X)
    # predict_y[predict_y>0] = 1

#////////////////////////////////SAVE PREDICTION RESULT///////////////////////////////////////////#
    # Create DataFrame for current fold's data
    fold_data_df = pd.concat([test_data, pd.DataFrame({'True Labels': test_y, 'Predicted Labels': predict_y})], axis=1)

    # Save features and label comparisons to a CSV file for the current fold
    fold_data_df.to_csv(f'./result/{i+1}.csv', index=False)
#////////////////////////////////SAVE PREDICTION RESULT///////////////////////////////////////////#
    # ALL Matching values
        
    # Calculate the number of matching values
    matching_values = np.sum(np.array(test_y) == np.array(predict_y))

    # Calculate the total number of samples
    total_samples = len(test_y)

    # Calculate the percentage of matching values
    percentage_matching = (matching_values / total_samples) * 100

    matching_percentage.append(percentage_matching)
    # Replace this with your model training and evaluation code
    # Train your model on X and y for the current fold
    # Evaluate the model and store the results
    # Example: model.fit(X, y), results.append(model.score(X_test, y_test))
    actualAttack = sum(test_y[test_y==1])
    predicted = sum(predict_y[predict_y==1])
    TP = sum(test_y[predict_y==1])
    FP = sum(predict_y[predict_y==1])-sum(test_y[predict_y==1])
    FN = sum(test_y[test_y==1])-sum(test_y[predict_y==1])
    # Print the results for the current fold
    print(f"************** Fold {i+1} **************")
    print("--- Overall Performance ---")
    print("Total Values :" , total_samples)
    print("Matching :",matching_values)
    print("Matching percentage :" ,percentage_matching)
    print("")
    print("--- Caught Performance ---")
    
    print('Actual Attack :',actualAttack)
    print('Predicted :',predicted)
    print('True Positive(attack caught) :',TP)
    print('False Positive :',FP)
    print('False Negative :',FN)
    f1Score = f1_score(test_y, predict_y)
    
    f1_percentage.append(f1Score)
    print('F1 score : ',f1Score)
    print('\n\n')
    # print('Caught Percentage :', )
    performance_df = performance_df._append({
        'Fold': i,
        'Total Values': total_samples,
        'Matching': matching_values,
        'Matching Percentage': percentage_matching,
        'Actual Attack': actualAttack,
        'Predicted': predicted,
        'True Positive': TP,
        'False Positive': FP,
        'False Negative': FN,
        'F1 Score': f1Score
    }, ignore_index=True)

# Save the DataFrame to a CSV file
performance_df.to_csv('performance_metrics.csv', index=False)
# Calculate and print the average result across all folds
average_result = np.mean(matching_percentage)
average_result2 = np.mean(f1_percentage)

print(f"Average Result Across Folds: {average_result}")
print(f"Average Result Across Folds: {average_result2}")

# Save feature importances to CSV
feature_importance_df.to_csv('feature_importances.csv', index=False)
