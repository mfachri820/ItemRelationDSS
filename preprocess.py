import pandas as pd
import ast

def load_dataset(path):
    """
    Load transaction dataset.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def parse_product_list(value):
    """
    Convert string representation ['A','B','C'] into a python list.
    """
    if isinstance(value, list):
        return value

    try:
        return ast.literal_eval(value)
    except:
        # fallback
        return [v.strip() for v in str(value).split(',')]


def clean_data(df):
    """
    Fix datatypes, booleans, numerics, lists.
    """

    # Date → datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Product list normalization
    df["Product"] = df["Product"].apply(parse_product_list)

    # Convert TRUE/FALSE to boolean
    df["Discount_Applied"] = df["Discount_Applied"].map(
        lambda x: True if str(x).strip().lower() in ["true", "1", "yes"] else False
    )

    # Ensure numeric types
    df["Total_Items"] = pd.to_numeric(df["Total_Items"], errors="coerce")
    df["Total_Cost"]  = pd.to_numeric(df["Total_Cost"], errors="coerce")

    # Fill categorical missing values
    for col in [
        "Customer_Name", "Payment_Method", "City", "Store_Type",
        "Customer_Category", "Season", "Promotion"
    ]:
        df[col] = df[col].fillna("Unknown")

    # Remove empty product rows
    df = df[df["Product"].apply(lambda x: len(x) > 0)]

    return df


def create_date_features(df):
    """
    Add features useful for seasonality analysis.
    """
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday
    df["Hour"] = df["Date"].dt.hour
    return df


def create_basket_dataset(df):
    """
    Convert to one-hot transaction basket for FP-Growth/Apriori.
    """
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()

    basket = df["Product"].tolist()
    encoded = te.fit(basket).transform(basket)

    return pd.DataFrame(encoded, columns=te.columns_)


def create_customer_aggregates(df):
    """
    For customer segmentation.
    """
    agg = df.groupby("Customer_Name").agg({
        "Total_Cost": ["sum", "mean"],
        "Total_Items": ["sum", "mean"],
        "Discount_Applied": "mean",
        "Transaction_ID": "count"
    })

    agg.columns = [
        "Total_Spend",
        "Avg_Spend",
        "Total_Items",
        "Avg_Items",
        "Discount_Rate",
        "Transaction_Count"
    ]

    return agg.reset_index()


def full_preprocessing(path):
    """
    Full pipeline: load → clean → enrich → prepare outputs.
    """
    df = load_dataset(path)
    df = clean_data(df)
    df = create_date_features(df)

    basket_df = create_basket_dataset(df)
    customer_df = create_customer_aggregates(df)

    return df, basket_df, customer_df
