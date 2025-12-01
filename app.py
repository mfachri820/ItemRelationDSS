import streamlit as st
import pandas as pd
import preprocess
import market_basket
import rule_display 
import time

# Page Config
st.set_page_config(page_title="Retail DSS System", layout="wide")
st.title("🛒 Retail Decision Support System")

# Sidebar: Controls
st.sidebar.header("Configuration")
# UPDATE: Pointing to the new dataset structure
DATA_PATH = "Dataset/Assignment-1_Data.csv" 

# 1. Load Data (Cached)
@st.cache_data 
def get_data():
    try:
        return preprocess.full_preprocessing(DATA_PATH)
    except FileNotFoundError:
        return None, None

df_full, basket_df_full = get_data()

if df_full is None:
    st.error(f"Dataset not found at {DATA_PATH}. Please ensure the file exists.")
    st.stop()

# --- PERFORMANCE TUNING ---
st.sidebar.subheader("🚀 Speed & Performance")

# CHANGE 1: Sample Size with unique key to force reset
sample_fraction = st.sidebar.slider(
    "Data Sample Size (%)", 
    min_value=1, 
    max_value=100, 
    value=10, 
    step=5,
    key="sample_slider_v3", 
    help="Start small (10%). Only increase if you need more data."
)

# Apply Sampling
if sample_fraction < 100:
    basket_df = basket_df_full.sample(frac=sample_fraction/100, random_state=42)
    current_txns = len(basket_df)
else:
    basket_df = basket_df_full
    current_txns = len(basket_df)

# CHANGE 2: The "Magic Switch" for speed
max_len = st.sidebar.slider(
    "Max Combination Length", 
    min_value=2, 
    max_value=4, 
    value=2,
    key="maxlen_slider",
    help="2 = A->B (Fastest). 3 = A+B->C (Slower)."
)

st.sidebar.markdown("---")

# --- PARAMETERS ---
st.sidebar.subheader("Rule Filters")

# CHANGE 3: Stricter Default Support (2%)
min_support = st.sidebar.slider("Minimum Support", 0.001, 0.2, 0.02, 0.001, format="%.4f")

# CHANGE 4: Default Lift set to 1.5
min_threshold = st.sidebar.slider("Minimum Lift Threshold", 0.1, 10.0, 1.5, 0.1)

# Confidence
min_confidence = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.2, 0.05)

# --- Dashboard Layout ---

with st.expander("📊 Dataset Overview"):
    col1, col2, col3 = st.columns(3)
    # UPDATE: Logic for new dataset structure
    # df_full is now 'Long Format' (Line Items), so unique BillNo = Transactions
    col1.metric("Unique Invoices", df_full['BillNo'].nunique())
    col2.metric("Analyzed Sample", f"{current_txns} ({sample_fraction}%)")
    col3.metric("Total Products", basket_df.shape[1])

# Section 2: Mining Engine
st.subheader("🔍 Market Basket Analysis")

try:
    with st.status("Running Analysis...", expanded=True) as status:
        
        # Step 1: Preparation
        st.write(f"Step 1/3: Preparing {current_txns} transactions...")
        basket_df_bool = basket_df.astype(bool)

        # Step 2: FP-Growth
        st.write(f"Step 2/3: Mining frequent pairs (Max Length: {max_len})...")
        
        # CRITICAL: Passing max_len here limits the complexity
        itemsets = market_basket.generate_frequent_itemsets(
            basket_df_bool, 
            min_support, 
            max_len=max_len
        )
        
        # Step 3: Association Rules
        if itemsets.empty:
            status.update(label="Finished: No itemsets found.", state="complete", expanded=False)
            rules = pd.DataFrame()
        else:
            st.write(f"Found {len(itemsets)} itemsets. Step 3/3: Generating Rules...")
            
            rules = market_basket.generate_rules(
                itemsets, 
                metric="lift", 
                min_threshold=min_threshold, 
                min_confidence=min_confidence
            )
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)

except Exception as e:
    st.error(f"Analysis Failed: {e}")
    st.stop()

col1, col2 = st.columns(2)
col1.info(f"Frequent Itemsets Found: {len(itemsets)}")

if rules.empty:
    col2.warning("Association Rules Found: 0")
    st.warning("No rules found. Try lowering **Support** slightly.")
    st.stop()
else:
    col2.success(f"Association Rules Found: {len(rules)}")
    
    search_term = st.text_input("🔎 Search for a Product (e.g., 'Milk')")
    rule_display.render_analysis_ui(rules, search_term)


# Section 4: Recommendation Simulator
st.divider()
st.subheader("💡 'If-Then' Simulator")

all_products = sorted(basket_df.columns.tolist())
selected_item = st.selectbox("Customer buys...", all_products)

if selected_item:
    recs = rules[rules['antecedents_str'] == selected_item]
    
    if not recs.empty:
        recs = recs.sort_values('lift', ascending=False)
        top_rec = recs.iloc[0]
        
        st.success(f"**Recommendation:** Suggest buying **{top_rec['consequents_str']}**")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Lift", f"{top_rec['lift']:.2f}")
        c2.metric("Confidence", f"{top_rec['confidence']:.1%}")
        c3.metric("Support", f"{top_rec['support']:.4f}")
        
        if len(recs) > 1:
            with st.expander("See other alternatives"):
                st.dataframe(recs[['consequents_str', 'lift', 'confidence']])
    else:
        st.info(f"No specific rules found starting with '{selected_item}' at current thresholds.")