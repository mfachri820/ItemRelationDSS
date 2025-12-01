import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def get_styled_dataframe(rules_df):
    """
    Takes the raw rules DataFrame and returns a styled version 
    for the Streamlit dashboard.
    """
    # 1. Create a clean copy for display
    # We select specific columns to avoid clutter
    display_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
    
    # Check if 'conviction' exists (it's useful for DSS)
    if 'conviction' in rules_df.columns:
        display_cols.append('conviction')

    # Filter columns that exist in the dataframe
    cols_to_use = [c for c in display_cols if c in rules_df.columns]
    
    df_view = rules_df[cols_to_use].copy()

    # 2. Rename columns for business users
    df_view.rename(columns={
        'antecedents_str': 'IF (Antecedent)',
        'consequents_str': 'THEN (Consequent)',
        'support': 'Support',
        'confidence': 'Confidence (%)',
        'lift': 'Lift Strength',
        'conviction': 'Conviction'
    }, inplace=True)

    # 3. Apply Pandas Styling (Colors and Formats)
    # - Gradient background for Lift (Darker Green = Stronger Rule)
    # - Percentage format for Confidence
    # - 3 decimal places for others
    styler = df_view.style.background_gradient(subset=['Lift Strength'], cmap='Greens') \
        .format({
            'Support': '{:.4f}',
            'Confidence (%)': '{:.1%}',  # Converts 0.85 -> 85.0%
            'Lift Strength': '{:.2f}',
            'Conviction': '{:.2f}'
        })
        
    return styler

def plot_rules_heatmap(rules_df, metric='lift', top_n=10):
    """
    Generates a Heatmap of Antecedents vs Consequents.
    Useful to see which items drive the sales of others.
    """
    # Sort and take top N rules to prevent the chart from being too crowded
    top_rules = rules_df.sort_values(metric, ascending=False).head(top_n)

    # Pivot data for heatmap format
    # Index = IF, Columns = THEN, Values = Metric (Lift/Confidence)
    try:
        pivot = top_rules.pivot(index='antecedents_str', 
                                columns='consequents_str', 
                                values=metric)
    except ValueError:
        # Fallback if duplicate keys exist or pivot fails
        return None

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    
    ax.set_title(f"Top {top_n} Rules by {metric.capitalize()}")
    ax.set_ylabel("IF (Customer Buys)")
    ax.set_xlabel("THEN (Likely to Buy)")
    
    return fig

def render_analysis_ui(rules, search_term=None):
    """
    Master function to render the entire analysis section in Streamlit.
    """
    # 1. Filter Logic
    display_rules = rules.copy()
    if search_term:
        mask = (
            display_rules["antecedents_str"].str.contains(search_term, case=False, na=False) |
            display_rules["consequents_str"].str.contains(search_term, case=False, na=False)
        )
        display_rules = display_rules[mask]
    
    if display_rules.empty:
        st.warning("No rules match your search criteria.")
        return

    # 2. Key Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Filtered Rules", len(display_rules))
    col2.metric("Max Lift", f"{display_rules['lift'].max():.2f}")
    col3.metric("Avg Confidence", f"{display_rules['confidence'].mean():.1%}")

    # 3. Display Styled Table
    st.subheader("📋 Detailed Rules Table")
    st.dataframe(
        get_styled_dataframe(display_rules),
        use_container_width=True
    )

    # 4. Display Heatmap
    st.subheader("🔥 Association Heatmap")
    st.write(f"Visualizing the strongest relationships within the current selection.")
    
    fig_heatmap = plot_rules_heatmap(display_rules, metric='lift', top_n=15)
    if fig_heatmap:
        st.pyplot(fig_heatmap)
    else:
        st.info("Not enough unique item pairs to generate a heatmap (need at least 2 distinct pairs).")