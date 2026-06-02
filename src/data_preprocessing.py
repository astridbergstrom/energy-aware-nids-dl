import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

COLUMN_NAMES = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

# Load dataset
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, names=COLUMN_NAMES)
    
    return train_df, test_df

# Binary classification, attack --> 1, normal --> 0
def binarize_labels(df):
    df = df.copy()
    df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)
    return df

# Remove unnecessary column
def drop_unused(df):
    return df.drop(columns=["difficulty"])

# One-hot encoding
def one_hot_encode(train_df, test_df):
    categorical_cols = ["protocol_type", "service", "flag"]
    
    full_df = pd.concat([train_df, test_df], axis=0)
    full_df = pd.get_dummies(full_df, columns=categorical_cols)
    
    # Split back
    train_df = full_df.iloc[:len(train_df), :]
    test_df = full_df.iloc[len(train_df):, :]
    
    return train_df, test_df

# Separate features and labels
def split_features_labels(df):
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y

# Normalization
def normalize(train_X, test_X):
    scaler = StandardScaler()
    
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    
    return train_X, test_X

# The whole pipe-line function
def preprocess_data(train_path, test_path):
    # Load
    train_df, test_df = load_data(train_path, test_path)
    
    # Clean
    train_df = drop_unused(train_df)
    test_df = drop_unused(test_df)
    
    # Labels
    train_df = binarize_labels(train_df)
    test_df = binarize_labels(test_df)
    
    # One-hot
    train_df, test_df = one_hot_encode(train_df, test_df)
    
    # Split
    train_X, train_y = split_features_labels(train_df)
    test_X, test_y = split_features_labels(test_df)
    
    # Normalize
    train_X, test_X = normalize(train_X, test_X)
    
    return train_X, train_y.values, test_X, test_y.values