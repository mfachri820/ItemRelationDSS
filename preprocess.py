import pandas as pd
import os
import pickle

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def load_dataset(path):
    """
    Load dataset with specific handling for semicolon delimiters
    and potential encoding issues.
    """
    try:
        # Try default UTF-8
        df = pd.read_csv(path, sep=';', encoding='utf-8')
    except UnicodeDecodeError:
        # Fallback for legacy datasets (often Latin-1)
        df = pd.read_csv(path, sep=';', encoding='latin-1')
        
    df.columns = df.columns.str.strip()
    return df

def clean_data(df):
    """
    Clean the specific format of the Online Retail dataset.
    """
    # 1. Parse Dates (Input format: 01.12.2010 08:26)
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y %H:%M", errors="coerce")
    
    # 2. Handle Price: Convert "2,55" to 2.55
    if df["Price"].dtype == object:
        df["Price"] = df["Price"].astype(str).str.replace(',', '.').astype(float)
        
    # 3. Clean Item Names
    df["Itemname"] = df["Itemname"].astype(str).str.strip()
    # Remove empty or 'nan' items
    df = df[~df["Itemname"].str.lower().isin(['nan', 'none', ''])]
    
    # 4. Filter Quantities
    # Remove returns (negative quantity)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df = df[df["Quantity"] > 0]
    
    return df

def create_basket_dataset(df):
    """
    Transforms Long-Format Data (one row per item) -> Sparse Basket Matrix.
    """
    from mlxtend.preprocessing import TransactionEncoder
    
    # GROUPING: Consolidate items by BillNo
    # Result: [['White Heart', 'Lantern'], ['Cream Cup', ...]]
    transactions = df.groupby('BillNo')['Itemname'].apply(list).tolist()
    
    te = TransactionEncoder()
    
    # Create Sparse Matrix (Memory Efficient)
    te_ary = te.fit(transactions).transform(transactions, sparse=True)
    
    return pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

def full_preprocessing(path, force_reload=True):
    # Changed cache name to avoid conflict with previous dataset
    cache_path = os.path.join(CACHE_DIR, "processed_retail_v2.pkl")
    
    if not force_reload and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    df = load_dataset(path)
    df = clean_data(df)
    
    # Note: df is now the cleaned "Long" format
    basket_df = create_basket_dataset(df)
    
    with open(cache_path, "wb") as f:
        pickle.dump((df, basket_df), f)
        
    return df, basket_df