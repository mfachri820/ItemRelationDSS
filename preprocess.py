import pandas as pd
import ast
import os
import pickle
from mlxtend.preprocessing import TransactionEncoder

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def parse_product_list(value):
    # Optimize: Check if it's already a list first
    if isinstance(value, list):
        return value
    if pd.isna(value) or value == '':
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Handle cases where strings might not have quotes
        return [x.strip() for x in str(value).split(',') if x.strip()]

def clean_data(df):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Product"] = df["Product"].apply(parse_product_list)
    
    # Vectorized boolean conversion
    df["Discount_Applied"] = df["Discount_Applied"].astype(str).str.lower().isin(["true", "1", "yes"])
    
    for col in ["Total_Items", "Total_Cost"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    cat_cols = ["Customer_Name", "Payment_Method", "City", "Store_Type", "Customer_Category", "Season", "Promotion"]
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    
    # Drop rows with empty baskets
    df = df[df["Product"].map(len) > 0]
    return df

def create_date_features(df):
    df["Year"] = df["Date"].dt.year
    df["Month_Name"] = df["Date"].dt.month_name()
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.day_name()
    return df

def create_basket_dataset(df):
    te = TransactionEncoder()
    basket = df["Product"].tolist()
    # Sparse=True saves RAM on large datasets, but mlxtend fpgrowth usually needs dense pandas df
    # We stick to dense for compatibility, but watch RAM usage.
    encoded = te.fit(basket).transform(basket)
    return pd.DataFrame(encoded, columns=te.columns_)

def create_basket_dataset(df):
    """
    OPTIMIZED: Uses Sparse Matrix to save memory and speed up processing.
    """
    
    te = TransactionEncoder()
    basket = df["Product"].tolist()
    
    # 1. Create Sparse Matrix (boolean) instead of dense array
    # sparse=True is crucial for performance on large datasets
    te_ary = te.fit(basket).transform(basket, sparse=True)
    
    # 2. Convert to Pandas Sparse DataFrame
    basket_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    
    return basket_df

def full_preprocessing(path, force_reload=False):
    cache_path = os.path.join(CACHE_DIR, "processed_data_sparse.pkl")
    
    if not force_reload and os.path.exists(cache_path):
        # print("⚡ Loading optimized data from cache...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # print("⚙️  Processing raw data (this happens once)...")
    df = load_dataset(path)
    df = clean_data(df)
    df = create_date_features(df)
    basket_df = create_basket_dataset(df)
    
    with open(cache_path, "wb") as f:
        pickle.dump((df, basket_df), f)
        
    return df, basket_df