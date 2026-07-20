"""
NovaRetail — Customer Intelligence Dashboard (Streamlit)
=========================================================

Prepared for Sophia Martinez, Director of Customer Intelligence.

Answers three questions, interactively:
  1. Which customers generate the most revenue?
  2. Which segments are at risk?
  3. Where should NovaRetail focus investment to maximize growth and retention?

Deployment:
    Keep NR_dataset.csv in the repo root (same folder as this file), or use
    the uploader that appears if no file is found.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless backend, required for Streamlit Cloud
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NovaRetail — Customer Intelligence",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEGMENT_COLORS = {
    "Promising": "#F2A93B",
    "Growth": "#2FB27D",
    "Stable": "#4472C4",
    "Decline": "#E0556F",
    "Unclassified": "#8A93A6",
}
SEGMENT_ORDER = ["Promising", "Growth", "Stable", "Decline", "Unclassified"]
AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55-64", "55+"]

CATEGORY_MAP = {
    "Electronics": "Electronics",
    "Home Appliances": "Home & Living", "Furniture": "Home & Living",
    "Home Improvement": "Home & Living", "Gardening Tools": "Home & Living",
    "Home Decor": "Home & Living", "Home & Garden": "Home & Living",
    "Furniture & Decor": "Home & Living",
    "Clothing": "Clothing & Fashion", "Fashion": "Clothing & Fashion",
    "Fashion & Apparel": "Clothing & Fashion", "Fashion Accessories": "Clothing & Fashion",
    "Sportswear": "Clothing & Fashion", "Children's Clothing": "Clothing & Fashion",
    "Groceries": "Groceries", "Grocery": "Groceries", "Grocery Items": "Groceries",
    "Food & Beverages": "Groceries",
    "Books": "Books & Media", "Books & Magazines": "Books & Media",
    "Toys": "Toys & Gaming", "Toys & Games": "Toys & Gaming", "Gaming": "Toys & Gaming",
    "Health & Wellness": "Health & Beauty", "Beauty Products": "Health & Beauty",
    "Health & Beauty": "Health & Beauty", "Beauty & Personal Care": "Health & Beauty",
    "Cosmetics": "Health & Beauty", "Health Supplements": "Health & Beauty",
    "Sporting Goods": "Sports & Outdoors", "Sports & Outdoors": "Sports & Outdoors",
    "Sports Equipment": "Sports & Outdoors", "Outdoor Equipment": "Sports & Outdoors",
    "Office Supplies": "Other", "Automotive": "Other",
}

DEFAULT_DATA_PATH = "NR_dataset.csv"


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(file) -> pd.DataFrame:
    """Load the raw transaction export and standardize it for analysis.

    - Consolidates the 35 overlapping raw ProductCategory labels into 9 clean categories.
    - Fills the one transaction missing a behavioral segment label as 'Unclassified'
      rather than dropping it, so it stays visible in totals.
    """
    df_raw = pd.read_csv(file)

    df = df_raw.copy()
    df["Category"] = df["ProductCategory"].map(CATEGORY_MAP).fillna("Other")
    df["Segment"] = df["label"].fillna("Unclassified")
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df["CustomerAgeGroup"] = pd.Categorical(
        df["CustomerAgeGroup"], categories=AGE_ORDER, ordered=True
    )

    df = df.rename(columns={
        "PurchaseAmount": "Amount",
        "CustomerRegion": "Region",
        "RetailChannel": "Channel",
        "CustomerSatisfaction": "Satisfaction",
        "CustomerGender": "Gender",
        "CustomerAgeGroup": "AgeGroup",
        "CustomerID": "CustomerId",
    })

    return df[[
        "TransactionID", "CustomerId", "Segment", "TransactionDate", "Category",
        "ProductCategory", "Amount", "AgeGroup", "Gender", "Region", "Satisfaction", "Channel",
    ]]


def apply_filters(
    df: pd.DataFrame,
    segments: list[str] | None = None,
    region: str | None = None,
    channel: str | None = None,
    category: str | None = None,
    age_group: str | None = None,
    gender: str | None = None,
) -> pd.DataFrame:
    """Return the subset of df matching the given filters. None/empty means 'no filter'."""
    data = df
    if segments:
        data = data[data["Segment"].isin(segments)]
    if region:
        data = data[data["Region"] == region]
    if channel:
        data = data[data["Channel"] == channel]
    if category:
        data = data[data["Category"] == category]
    if age_group:
        data = data[data["AgeGroup"] == age_group]
    if gender:
        data = data[data["Gender"] == gender]
    return data


# ---------------------------------------------------------------------------
# KPIs & tables
# ---------------------------------------------------------------------------

def kpi_summary(data: pd.DataFrame) -> dict:
    total_revenue = data["Amount"].sum()
    total_customers = data["CustomerId"].nunique()
    avg_satisfaction = data["Satisfaction"].mean() if len(data) else float("nan")
    decline_revenue = data.loc[data["Segment"] == "Decline", "Amount"].sum()
    decline_share = (decline_revenue / total_revenue * 100) if total_revenue else 0.0
    return {
        "Total Revenue": f"${total_revenue:,.2f}",
        "Transactions": f"{len(data):,}",
        "Active Customers": f"{total_customers:,}",
        "Avg Satisfaction (1-5)": f"{avg_satisfaction:.2f}",
        "Revenue at Risk (Decline)": f"${decline_revenue:,.2f} ({decline_share:.1f}%)",
    }


def top_customers_table(data: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    top = (
        data.groupby(["CustomerId", "Segment"])
        .agg(
            Revenue=("Amount", "sum"),
            Orders=("Amount", "size"),
            AvgSatisfaction=("Satisfaction", "mean"),
            Region=("Region", lambda s: s.mode().iat[0]),
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
        .head(n)
    )
    top["AvgSatisfaction"] = top["AvgSatisfaction"].round(1)
    return top


def decline_detail_table(data: pd.DataFrame) -> pd.DataFrame:
    decline = data[data["Segment"] == "Decline"]
    if decline.empty:
        return pd.DataFrame()
    detail = (
        decline.groupby(["Region", "Category"])
        .agg(
            Revenue=("Amount", "sum"),
            Transactions=("Amount", "size"),
            AvgSatisfaction=("Satisfaction", "mean"),
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )
    detail["AvgSatisfaction"] = detail["AvgSatisfaction"].round(1)
    return detail


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_revenue_by_segment(data: pd.DataFrame, ax=None):
    """Which customers generate the most revenue — by behavioral segment."""
    seg_rev = (
        data.groupby("Segment")["Amount"].sum()
        .reindex(SEGMENT_ORDER).dropna().sort_values(ascending=True)
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    if seg_rev.empty:
        ax.axis("off")
        return ax
    colors = [SEGMENT_COLORS[s] for s in seg_rev.index]
    bars = ax.barh(seg_rev.index, seg_rev.values, color=colors)
    ax.set_title("Revenue by Segment", fontsize=13, fontweight="bold", loc="left")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for b, v in zip(bars, seg_rev.values):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  ${v:,.0f}", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_decline_by_region(data: pd.DataFrame, ax=None):
    """Early-warning view: Decline-segment revenue & satisfaction by region."""
    decline = data[data["Segment"] == "Decline"]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    if decline.empty:
        ax.axis("off")
        return ax
    g = decline.groupby("Region").agg(Revenue=("Amount", "sum"), AvgSat=("Satisfaction", "mean"))
    g = g.sort_values("Revenue", ascending=False)
    bars = ax.bar(g.index, g["Revenue"], color=SEGMENT_COLORS["Decline"])
    ax.set_title("Decline Segment — Revenue & Satisfaction by Region", fontsize=13, fontweight="bold", loc="left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for b, (rev, sat) in zip(bars, zip(g["Revenue"], g["AvgSat"])):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{sat:.1f}\u2605",
                ha="center", va="bottom", fontsize=9, color=SEGMENT_COLORS["Decline"])
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_growth_by_category(data: pd.DataFrame, ax=None):
    """Investment candidates: top categories among Growth-segment customers."""
    growth = data[data["Segment"] == "Growth"]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.8))
    if growth.empty:
        ax.axis("off")
        return ax
    g = growth.groupby("Category")["Amount"].sum().sort_values(ascending=True).tail(7)
    bars = ax.barh(g.index, g.values, color=SEGMENT_COLORS["Growth"])
    ax.set_title("Growth Opportunity by Category", fontsize=13, fontweight="bold", loc="left")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for b, v in zip(bars, g.values):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  ${v:,.0f}", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_channel_by_category(data: pd.DataFrame, ax=None):
    """Online vs. physical-store revenue across product categories."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    if data.empty:
        ax.axis("off")
        return ax
    pivot = (
        data.pivot_table(index="Category", columns="Channel", values="Amount", aggfunc="sum", fill_value=0)
        .assign(total=lambda d: d.sum(axis=1))
        .sort_values("total", ascending=False)
        .drop(columns="total")
    )
    colors = ["#4472C4", "#2A3350"] if "Online" in pivot.columns else None
    pivot.plot(kind="bar", stacked=True, ax=ax, color=colors)
    ax.set_title("Channel Performance by Category", fontsize=13, fontweight="bold", loc="left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="")
    return ax


def render_charts(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    plot_revenue_by_segment(data, ax=axes[0, 0])
    plot_decline_by_region(data, ax=axes[0, 1])
    plot_growth_by_category(data, ax=axes[1, 0])
    plot_channel_by_category(data, ax=axes[1, 1])
    fig.suptitle("NovaRetail — Customer Intelligence Dashboard", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def get_data_source():
    """Use the bundled CSV if it's in the repo; otherwise ask for an upload."""
    import os
    if os.path.exists(DEFAULT_DATA_PATH):
        return DEFAULT_DATA_PATH
    st.warning(f"`{DEFAULT_DATA_PATH}` not found in the repo.")
    uploaded = st.file_uploader("Upload the transaction CSV", type="csv")
    if uploaded is None:
        st.stop()
    return uploaded


def main() -> None:
    st.title("📊 NovaRetail — Customer Intelligence Dashboard")
    st.caption("Prepared for Sophia Martinez, Director of Customer Intelligence.")

    source = get_data_source()
    df = load_data(source)

    st.sidebar.header("Filters")
    segments = st.sidebar.multiselect("Segment", SEGMENT_ORDER)
    region = st.sidebar.selectbox("Region", [None] + sorted(df["Region"].dropna().unique().tolist()))
    channel = st.sidebar.selectbox("Channel", [None] + sorted(df["Channel"].dropna().unique().tolist()))
    category = st.sidebar.selectbox("Category", [None] + sorted(df["Category"].dropna().unique().tolist()))
    age_group = st.sidebar.selectbox("Age Group", [None] + AGE_ORDER)
    gender = st.sidebar.selectbox("Gender", [None] + sorted(df["Gender"].dropna().unique().tolist()))

    data = apply_filters(
        df,
        segments=segments,
        region=region,
        channel=channel,
        category=category,
        age_group=age_group,
        gender=gender,
    )

    st.caption(f"Showing {len(data):,} of {len(df):,} transactions")

    # KPIs
    kpis = kpi_summary(data)
    cols = st.columns(len(kpis))
    for col, (label, value) in zip(cols, kpis.items()):
        col.metric(label, value)

    if data.empty:
        st.info("No transactions match the current filters.")
        return

    st.divider()
    render_charts(data)

    st.divider()
    st.subheader("Top Customers by Revenue")
    top = top_customers_table(data)
    if top.empty:
        st.write("No customers match the current filters.")
    else:
        st.dataframe(top, use_container_width=True, hide_index=True)

    st.subheader("Decline Segment Detail (early warning, by region & category)")
    detail = decline_detail_table(data)
    if detail.empty:
        st.write("No Decline-segment activity in this view.")
    else:
        st.dataframe(detail, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
