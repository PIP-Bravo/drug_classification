import pandas as pd
import os

def check_data_types():
    """Check data types dalam CSV file"""
    data_path = r"C:\drug-pipeline\data\drug200.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ File tidak ditemukan: {data_path}")
        return
    
    # Load data
    df = pd.read_csv(data_path)
    
    print("📊 Data Overview:")
    print(f"Shape: {df.shape}")
    print("\nColumns and Data Types:")
    print(df.dtypes)
    
    print("\nSample Data:")
    print(df.head())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nUnique Values per Column:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")
        if df[col].nunique() < 20:  # Print values untuk categorical columns
            print(f"   Values: {sorted(df[col].unique())}")

if __name__ == "__main__":
    check_data_types()