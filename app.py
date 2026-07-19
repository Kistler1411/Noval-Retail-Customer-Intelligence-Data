import streamlit as st
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(
    page_title="NovaRetail Customer Intelligence",
    page_icon="📊",
    layout="wide"
)

# Brand palette (matching your notebook specs)
SEG_COLOR = {
    'Promising': '#1F8A70',
    'Growth': '#2E6FA7',
    'Stable': '#C08A2E',
    'Decline': '#B23A48',
}
SEGMENTS = ['Promising', 'Growth', 'Stable', 'Decline']

# Load data with caching for performance
@st.cache_data
def load_data():
    # Make sure 'NR_dataset.xlsx' is in your GitHub repository
    df = pd.read_excel('NR_dataset.xlsx')
    return df

# Main Header
st.title("NovaRetail Customer Intelligence Dashboard")
st.caption("**Prepared for:** Sophia Martinez, Director of Customer Intelligence")

try:
    df = load_data()
    
    # -------------------------------------------------------------
    # SIDEBAR FILTERS (Replacing ipywidgets)
    # -------------------------------------------------------------
    st.sidebar.header("Filter Dashboard")

    # Segment filter
    available_segments = df['Segment'].dropna().unique().tolist() if 'Segment' in df.columns else SEGMENTS
    selected_segments = st.sidebar.multiselect(
        "Customer Segment",
        options=available_segments,
        default=available_segments
    )

    # Apply filters
    filtered_df = df[df['Segment'].isin(selected_segments)] if 'Segment' in df.columns else df

    # -------------------------------------------------------------
    # METRICS DISPLAY
    # -------------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{len(filtered_df):,}")
    
    if 'Customer_ID' in filtered_df.columns:
        col2.metric("Unique Customers", f"{filtered_df['Customer_ID'].nunique():,}")
    
    if 'Revenue' in filtered_df.columns:
        col3.metric("Total Revenue", f"${filtered_df['Revenue'].sum():,.2f}")

    st.markdown("---")

    # -------------------------------------------------------------
    # CHARTS (Plotly)
    # -------------------------------------------------------------
    st.subheader("Segment Breakdown")

    if 'Segment' in filtered_df.columns:
        seg_counts = filtered_df['Segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']

        fig = px.bar(
            seg_counts,
            x='Segment',
            y='Count',
            color='Segment',
            color_discrete_map=SEG_COLOR,
            title="Transactions by Customer Segment"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Raw Data Preview
    with st.expander("View Filtered Raw Data"):
        st.dataframe(filtered_df)

except FileNotFoundError:
    st.error("⚠️ `NR_dataset.xlsx` was not found. Please upload it to your GitHub repository.")
