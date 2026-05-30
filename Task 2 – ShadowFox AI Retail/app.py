from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ai_engine.ai_insights import generate_business_insights
from ai_engine.chatbot import answer_business_question
from ai_engine.recommendation_engine import build_recommendations
from anomaly_detection.detector import detect_anomalies
from auth.login import render_login, render_user_menu
from config.settings import APP_NAME, APP_SUBTITLE, LOGO_PATH
from dashboard.kpi_engine import calculate_kpis
from forecasting.forecast_model import forecast_metric
from reports.report_generator import build_excel_report, build_pdf_report
from utils.preprocess import generate_sample_data, prepare_retail_data, read_retail_csv
from visualization.charts import category_treemap, discount_margin_scatter, forecast_chart, region_profit_bar, sales_profit_trend


logo_path = Path(LOGO_PATH)
st.set_page_config(page_title=APP_NAME, page_icon=str(logo_path) if logo_path.exists() else "SF", layout="wide")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root { --panel: rgba(17, 24, 39, .72); --line: rgba(148, 163, 184, .22); }
        .stApp {
            background:
                radial-gradient(circle at 15% 10%, rgba(124, 58, 237, .24), transparent 26%),
                radial-gradient(circle at 85% 5%, rgba(6, 182, 212, .16), transparent 25%),
                linear-gradient(135deg, #050816 0%, #090b19 45%, #12051f 100%);
            color: #eef2ff;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(8, 10, 24, .98), rgba(26, 10, 46, .95));
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            margin-bottom: .35rem;
        }
        [data-testid="stSidebar"] .stButton>button {
            min-height: 38px;
            padding: .35rem .7rem;
        }
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 0 14px;
            border-bottom: 1px solid rgba(148, 163, 184, .18);
            margin-bottom: 12px;
        }
        .sidebar-brand img {
            width: 56px;
            height: 56px;
            object-fit: cover;
            border-radius: 8px;
            border: 1px solid rgba(148, 163, 184, .22);
        }
        .sidebar-brand-title {
            font-size: 1.02rem;
            font-weight: 800;
            line-height: 1.1;
        }
        .sidebar-brand-subtitle {
            color: #94a3b8;
            font-size: .78rem;
            margin-top: 3px;
        }
        .sidebar-account {
            border: 1px solid rgba(148, 163, 184, .18);
            background: rgba(15, 23, 42, .58);
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 10px;
        }
        .sidebar-account-label {
            color: #94a3b8;
            font-size: .72rem;
            text-transform: uppercase;
        }
        .sidebar-account-email {
            color: #e2e8f0;
            font-size: .82rem;
            overflow-wrap: anywhere;
            margin-top: 3px;
        }
        .sidebar-role {
            display: inline-block;
            margin-top: 8px;
            padding: 2px 8px;
            border-radius: 999px;
            color: #a7f3d0;
            background: rgba(16, 185, 129, .14);
            border: 1px solid rgba(16, 185, 129, .28);
            font-size: .78rem;
        }
        [data-testid="stSidebar"] details {
            border: 1px solid rgba(148, 163, 184, .16);
            border-radius: 8px;
            background: rgba(15, 23, 42, .42);
            padding: 2px 8px;
            margin: 10px 0;
        }
        [data-testid="stSidebar"] details summary {
            font-weight: 700;
        }
        .block-container { padding-top: 1.4rem; }
        .hero {
            border: 1px solid var(--line);
            background: linear-gradient(135deg, rgba(15,23,42,.82), rgba(88,28,135,.34));
            padding: 24px 28px;
            border-radius: 8px;
            box-shadow: 0 20px 70px rgba(0,0,0,.38);
        }
        .hero h1 { margin: 0; font-size: 2.2rem; letter-spacing: 0; }
        .hero p { color: #cbd5e1; margin: 8px 0 0; }
        .metric-card {
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 8px;
            padding: 18px;
            min-height: 124px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 14px 40px rgba(0,0,0,.24);
        }
        .metric-label { color: #94a3b8; font-size: .82rem; text-transform: uppercase; }
        .metric-value { font-size: 1.65rem; font-weight: 750; margin-top: 8px; }
        .metric-trend { color: #22c55e; margin-top: 8px; font-size: .88rem; }
        .glass {
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 8px;
            padding: 18px;
        }
        .alert-high { border-left: 4px solid #e11d48; }
        .alert-medium { border-left: 4px solid #facc15; }
        .stButton>button, .stDownloadButton>button {
            border-radius: 8px;
            border: 1px solid rgba(124,58,237,.55);
            background: linear-gradient(135deg, #6d28d9, #0891b2);
            color: white;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    return generate_sample_data()


def currency(value: float) -> str:
    return f"INR {value:,.0f}"


def metric_card(label: str, value: str, trend: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-trend">{trend}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        with st.expander("Business Filters", expanded=False):
            st.caption("Leave selections empty to include all data.")
            all_regions = sorted(df["region"].unique())
            all_categories = sorted(df["category"].unique())
            regions = st.multiselect("Regions", all_regions, default=[], placeholder="All regions")
            categories = st.multiselect("Categories", all_categories, default=[], placeholder="All categories")
            min_date, max_date = df["order_date"].min().date(), df["order_date"].max().date()
            dates = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    selected_regions = regions or all_regions
    selected_categories = categories or all_categories
    filtered = df[df["region"].isin(selected_regions) & df["category"].isin(selected_categories)]
    if isinstance(dates, tuple) and len(dates) == 2:
        start, end = pd.to_datetime(dates[0]), pd.to_datetime(dates[1])
        filtered = filtered[(filtered["order_date"] >= start) & (filtered["order_date"] <= end)]
    return filtered


def render_header() -> None:
    logo = Path(LOGO_PATH)
    if logo.exists():
        logo_col, title_col = st.columns([0.16, 0.84], vertical_alignment="center")
        with logo_col:
            st.image(str(logo), width=120)
        with title_col:
            st.markdown(
                f"""
                <div class="hero">
                    <h1>{APP_NAME}</h1>
                    <p>{APP_SUBTITLE}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_NAME}</h1>
            <p>{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(df: pd.DataFrame) -> None:
    kpis = calculate_kpis(df)
    cols = st.columns(4)
    with cols[0]:
        metric_card("Total Sales", currency(kpis["total_sales"]), f"{kpis['revenue_growth']:.1%} latest growth")
    with cols[1]:
        metric_card("Total Profit", currency(kpis["total_profit"]), f"{kpis['profit_margin']:.1%} margin")
    with cols[2]:
        metric_card("Customers", f"{kpis['customers']:,}", f"{kpis['orders']:,} orders")
    with cols[3]:
        metric_card("Top Region", kpis["top_region"], kpis["top_product"][:28])

    left, right = st.columns([1.5, 1])
    with left:
        st.plotly_chart(sales_profit_trend(df), width="stretch")
    with right:
        st.markdown("#### AI Executive Summary")
        for insight in generate_business_insights(df):
            st.markdown(f"<div class='glass'>{insight}</div>", unsafe_allow_html=True)

    st.markdown("#### Smart Recommendations")
    rec_cols = st.columns(2)
    for idx, item in enumerate(build_recommendations(df)):
        klass = "alert-high" if item["priority"] == "High" else "alert-medium"
        with rec_cols[idx % 2]:
            st.markdown(
                f"<div class='glass {klass}'><b>{item['area']}</b><br>{item['recommendation']}</div>",
                unsafe_allow_html=True,
            )


def render_analytics(df: pd.DataFrame) -> None:
    tab_sales, tab_profit, tab_customer, tab_product = st.tabs(["Sales", "Profit", "Customers", "Products"])
    with tab_sales:
        st.plotly_chart(sales_profit_trend(df), width="stretch")
        st.dataframe(df.groupby(["year", "quarter"], as_index=False).agg(sales=("sales", "sum"), profit=("profit", "sum")), width="stretch")
    with tab_profit:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(region_profit_bar(df), width="stretch")
        with c2:
            st.plotly_chart(discount_margin_scatter(df), width="stretch")
    with tab_customer:
        customer = df.groupby("customer", as_index=False).agg(sales=("sales", "sum"), profit=("profit", "sum"), orders=("order_id", "count")).sort_values("profit", ascending=False)
        st.dataframe(customer.head(25), width="stretch")
    with tab_product:
        st.plotly_chart(category_treemap(df), width="stretch")
        product = df.groupby(["category", "subcategory", "product"], as_index=False).agg(sales=("sales", "sum"), profit=("profit", "sum"), discount=("discount", "mean")).sort_values("profit")
        st.dataframe(product.head(30), width="stretch")


def render_forecasting(df: pd.DataFrame) -> None:
    metric = st.segmented_control("Forecast Metric", ["sales", "profit"], default="sales")
    forecast = forecast_metric(df, metric=metric, periods=8)
    st.plotly_chart(forecast_chart(forecast, f"{metric.title()} Forecast"), width="stretch")
    st.dataframe(forecast.tail(12), width="stretch")


def render_anomalies(df: pd.DataFrame) -> None:
    anomalies = detect_anomalies(df)
    st.dataframe(anomalies.sort_values("month", ascending=False), width="stretch")
    critical = anomalies[anomalies["anomaly"]]
    st.markdown("#### Smart Alerts")
    if critical.empty:
        st.info("No major anomalies detected in the selected period.")
    for _, row in critical.iterrows():
        st.warning(f"{row['severity']}: unusual sales/profit behavior in {row['month'].strftime('%b %Y')}.")


def render_ai_chat(df: pd.DataFrame) -> None:
    question = st.text_input("Ask a business question", placeholder="Show top 10 profitable products")
    if question:
        st.code(answer_business_question(question, df), language="text")
    else:
        st.info("Try: Why did profits decrease? What is the discount impact? Sales by region.")


def render_reports(df: pd.DataFrame) -> None:
    kpis = calculate_kpis(df)
    insights = generate_business_insights(df)
    recommendations = build_recommendations(df)
    st.markdown("#### Executive Report Generator")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Export Excel Report", data=build_excel_report(df, kpis, recommendations), file_name="shadowfox_retail_report.xlsx")
    with c2:
        st.download_button("Export PDF Report", data=build_pdf_report(kpis, insights, recommendations), file_name="shadowfox_executive_report.pdf")

    for insight in insights:
        st.markdown(f"<div class='glass'>{insight}</div>", unsafe_allow_html=True)


def main() -> None:
    inject_css()
    if not render_login():
        return

    render_user_menu()
    render_header()

    pages = ["Dashboard", "Analytics Center", "Forecasting", "Anomaly Detection", "AI Chat Assistant", "Report Generator", "Admin Panel"]
    with st.sidebar:
        st.markdown("### Workspace")
        page = st.selectbox("Page", pages, label_visibility="collapsed")
        with st.expander("Data Source", expanded=False):
            uploaded = st.file_uploader("Upload Retail CSV", type=["csv"], label_visibility="collapsed")
            st.caption("Upload a CSV or use the built-in demo retail dataset.")

    if uploaded is not None:
        try:
            df = prepare_retail_data(read_retail_csv(uploaded))
        except ValueError as error:
            st.error(str(error))
            return
    else:
        df = load_default_data()

    df = filter_data(df)
    if df.empty:
        st.error("No records match the selected filters.")
        return

    if page == "Dashboard":
        render_dashboard(df)
    elif page == "Analytics Center":
        render_analytics(df)
    elif page == "Forecasting":
        render_forecasting(df)
    elif page == "Anomaly Detection":
        render_anomalies(df)
    elif page == "AI Chat Assistant":
        render_ai_chat(df)
    elif page == "Report Generator":
        render_reports(df)
    else:
        st.markdown("#### Admin Panel")
        st.dataframe(pd.DataFrame({"Role": ["Admin", "Analyst", "Viewer"], "Access": ["Upload, Analyze, Export, Configure", "Analyze, Export", "Read-only analytics"]}), width="stretch")
        st.info("OpenAI/Gemini keys can be added later through environment variables for live LLM responses.")


if __name__ == "__main__":
    main()
