import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth, association_rules

# ============================================================
# 1. FP-GROWTH Frequent Itemset Mining (Optimized)
# ============================================================
def generate_frequent_itemsets(basket_df, min_support=0.01, max_len=3):
    """
    Generate frequent itemsets using FP-Growth.
    
    Parameters:
        basket_df (DataFrame): Boolean dataframe of transactions.
        min_support (float): Minimum support threshold.
        max_len (int): Optimization - Limit itemset size (e.g., 3 means max 3 items together).
                       Lower numbers = Much Faster.

    Returns:
        DataFrame of frequent itemsets.
    """
    # max_len limits the depth of the search tree, drastically improving speed
    itemsets = fpgrowth(
        basket_df,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len 
    )
    
    if not itemsets.empty:
        itemsets["itemset_length"] = itemsets["itemsets"].apply(len)
    
    return itemsets


# ============================================================
# 2. ASSOCIATION RULE GENERATOR
# ============================================================
def generate_rules(itemsets, metric="lift", min_threshold=1.0, min_confidence=None):
    """
    Generate association rules from frequent itemsets.
    
    Parameters:
        itemsets (DataFrame): Frequent itemsets.
        metric (str): Primary metric for filtering (default "lift").
        min_threshold (float): Threshold for the primary metric.
        min_confidence (float): Optional secondary threshold for confidence (e.g., 0.1).
                                Helps remove weak rules even if they meet the lift criteria.
    """
    if itemsets.empty:
        return pd.DataFrame()
    
    # num_itemsets parameter helps prevent OOM on massive rule sets
    rules = association_rules(
        itemsets,
        metric=metric,
        min_threshold=min_threshold
    )
    
    if rules.empty:
        return pd.DataFrame()

    # Apply Secondary Filter (Confidence)
    # This is crucial for DSS to avoid showing rules with 1% chance of happening
    if min_confidence is not None:
        rules = rules[rules["confidence"] >= min_confidence]
        
    if rules.empty:
        return pd.DataFrame()

    # Pre-calculate string representations for the UI to use later
    # We use a simple lambda. For massive datasets, this part can be skipped until display.
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ', '.join(list(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ', '.join(list(x)))
    
    # Sort by lift descending (stronger rules first)
    rules = rules.sort_values("lift", ascending=False)
    
    return rules


# ============================================================
# 3. HELPER: FILTERING
# ============================================================
def filter_rules_by_product(rules, product_name):
    if rules.empty: return rules
    # Case insensitive string match
    mask = (
        rules["antecedents_str"].str.contains(product_name, case=False, na=False) |
        rules["consequents_str"].str.contains(product_name, case=False, na=False)
    )
    return rules[mask]


# ============================================================
# 4. VISUALIZATION
# ============================================================
def plot_network_graph(rules, max_rules=20):
    """
    Visualizes the association rules as a network graph.
    """
    if rules.empty: return None

    G = nx.DiGraph()
    
    # Limit to top N rules to avoid clutter and lag
    top_rules = rules.head(max_rules)

    for _, row in top_rules.iterrows():
        # Add Node A -> Node B with weight = Lift
        G.add_edge(row['antecedents_str'], row['consequents_str'], weight=row['lift'])

    # Layout calculation
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.9, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowstyle='-|>', arrowsize=20, ax=ax)
    
    # Draw Labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
    
    ax.set_title(f"Top {max_rules} Association Rules (Network Graph)")
    ax.axis('off')
    return fig