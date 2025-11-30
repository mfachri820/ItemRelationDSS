import pandas as pd
# Import your two custom modules
import preprocess
import market_basket

# 1. Setup Path to your dataset
DATA_PATH = "Dataset/Retail_Transactions_Dataset.csv" 

print("--- Step 1: Loading and Cleaning Data ---")
try:
    df, basket_df, customer_df = preprocess.full_preprocessing(DATA_PATH)
    print(f"Data loaded. Rows: {len(df)}")
    print(f"Basket Matrix Shape: {basket_df.shape}")
except FileNotFoundError:
    print("❌ Error: File not found. Please check DATA_PATH.")
    exit()

print("\n--- Step 2: Generating Frequent Itemsets ---")
# FIX: Lowered min_support from 0.02 (2%) to 0.001 (0.1%)
# In large datasets (1M rows), patterns are often sparse.
current_support = 0.001 
itemsets = market_basket.generate_frequent_itemsets(basket_df, min_support=current_support)
print(f"Found {len(itemsets)} frequent itemsets using support={current_support}.")

# DEBUG: Check if we actually found any pairs (length > 1)
if not itemsets.empty and 'itemset_length' in itemsets.columns:
    print("Itemset Length Distribution:")
    print(itemsets['itemset_length'].value_counts().sort_index())

print("\n--- Step 3: Generating Association Rules ---")
if not itemsets.empty:
    # We use a standard lift threshold. 
    # If you still get 0 rules, try changing metric="confidence" and min_threshold=0.1
    rules = market_basket.generate_rules(itemsets, metric="lift", min_threshold=1.0)
    print(f"Found {len(rules)} rules.")
    
    if not rules.empty:
        # Show top 5 rules
        print("\nTop 5 Rules by Lift:")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))
        
        # Test the search function
        print("\n--- Step 4: Testing Search ---")
        keyword = "Honey" 
        results = market_basket.search_rules(rules, keyword)
        if not results.empty:
            print(f"Rules involving '{keyword}':")
            print(results[['antecedents', 'consequents', 'lift']].head())
        else:
            print(f"No rules found for '{keyword}'")
    else:
        print("Itemsets found, but no rules met the lift > 1.0 criteria.")
else:
    print("No itemsets found. Try lowering min_support even further.")