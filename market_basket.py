import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules


# ============================================================
# 1. FP-GROWTH Frequent Itemset Mining
# ============================================================
def generate_frequent_itemsets(basket_df, min_support=0.01):
    """
    Generate frequent itemsets using FP-Growth.
    
    Parameters:
        basket_df (DataFrame): One-hot encoded product basket.
        min_support (float): Minimum support threshold.

    Returns:
        DataFrame of frequent itemsets.
    """
    itemsets = fpgrowth(
        basket_df,
        min_support=min_support,
        use_colnames=True
    )
    
    # Add length of itemset
    itemsets["itemset_length"] = itemsets["itemsets"].apply(len)
    
    return itemsets


# ============================================================
# 2. ASSOCIATION RULE GENERATOR
# ============================================================
def generate_rules(itemsets, metric="lift", min_threshold=1.0):
    """
    Generate association rules from frequent itemsets.
    
    Parameters:
        itemsets (DataFrame): Frequent itemsets from FP-growth.
        metric (str): "confidence", "lift", "support", etc.
        min_threshold (float): Minimum threshold for metric.

    Returns:
        Association rules DataFrame.
    """
    rules = association_rules(
        itemsets,
        metric=metric,
        min_threshold=min_threshold
    )
    
    # Sort by lift descending (stronger rules first)
    rules = rules.sort_values("lift", ascending=False)
    
    return rules


# ============================================================
# 3. CONTEXT-AWARE RULE MINING
# ============================================================
def generate_context_rules(df, context_filters, min_support=0.01, min_lift=1.0):
    """
    Generate rules based on conditional context.
    
    Example usage:
        filters = {"Season": "Winter", "Store_Type": "Warehouse Club"}
    
    Parameters:
        df (DataFrame): Cleaned transactions (NOT basket_df).
        context_filters (dict): e.g. {"Season": "Winter"}
        min_support (float)
        min_lift (float)

    Returns:
        Rules DataFrame.
    """

    filtered_df = df.copy()

    # Apply filters
    for col, value in context_filters.items():
        filtered_df = filtered_df[filtered_df[col] == value]

    # Edge case: not enough data
    if filtered_df.shape[0] < 2:
        return pd.DataFrame()

    # Build one-hot basket for filtered data
    basket = filtered_df["Product"].tolist()

    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    encoded = te.fit(basket).transform(basket)
    basket_df = pd.DataFrame(encoded, columns=te.columns_)

    # Generate itemsets + rules
    itemsets = generate_frequent_itemsets(basket_df, min_support=min_support)
    if itemsets.empty:
        return pd.DataFrame()

    rules = generate_rules(itemsets, metric="lift", min_threshold=min_lift)
    return rules


# ============================================================
# 4. STREAMLIT QUERY FUNCTION
# ============================================================
def get_rules_for_product(rules, product_name):
    """
    Return all rules where `product_name` appears in antecedents or consequents.
    
    Output is sorted by lift descending.
    """
    mask = (
        rules["antecedents"].apply(lambda x: product_name in list(x)) |
        rules["consequents"].apply(lambda x: product_name in list(x))
    )

    return rules[mask].sort_values("lift", ascending=False)


def search_rules(rules, keyword):
    """
    Fuzzy search rules by keyword (product or category).
    """
    keyword = keyword.lower()

    mask = (
        rules["antecedents"].astype(str).str.lower().str.contains(keyword) |
        rules["consequents"].astype(str).str.lower().str.contains(keyword)
    )

    return rules[mask].sort_values("lift", ascending=False)
