
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(
    page_title="Benford Analysis Dashboard",
    layout="wide"
)

st.title("Benford Analysis Dashboard")
st.write("Upload, filter, visualize, and analyze financial data using Benford's Law.")

# Upload CSV or Excel file
uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True),
        errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

# Benford analysis function
def calculate_benford_statistics(values, digits=1):
    values = values.abs()
    values = values[values > 0].dropna()

    leading = values.apply(
        lambda x: int(
            format(x, ".15e")
            .split("e")[0]
            .replace(".", "")[:digits]
        )
    )

    if digits == 1:
        digit_range = np.arange(1, 10)
    else:
        digit_range = np.arange(10, 100)

    counts = leading.value_counts().reindex(
        digit_range, fill_value=0
    )

    results = pd.DataFrame({
        "digit": digit_range,
        "count": counts.values
    })

    results["percent"] = (
        results["count"] / results["count"].sum()
    )

    results["benford"] = np.log10(
        1 + 1 / results["digit"]
    )

    results["difference"] = (
        results["percent"] - results["benford"]
    )

    return results

# KS Statistic Using Numpy
def calculate_ks(results):
    cum_diff = (
        np.cumsum(results["percent"].values)
        - np.cumsum(results["benford"].values)
    )

    max_diff = np.max(np.abs(cum_diff))

    critical_value = (
        1.36 / np.sqrt(results["count"].sum())
    )

    return max_diff, critical_value

if uploaded_file is not None:

    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

    except Exception as e:
        st.error(f"Unable to read file: {e}")
        st.stop()

    if df.empty:
        st.warning("The uploaded file contains no data.")
        st.stop()

    st.success("File uploaded successfully!")

    st.subheader("Original Data Preview")
    st.dataframe(df.head(20))

    numeric_columns = [
        col for col in df.columns
        if clean_numeric(df[col]).notna().any()
    ]

    if not numeric_columns:
        st.warning("No numeric columns were found.")
        st.stop()

    st.sidebar.header("Data Filters")

    amount_column = st.sidebar.selectbox(
        "Select column for Benford analysis",
        numeric_columns
    )

    filtered_df = df.copy()

    filtered_df[amount_column] = clean_numeric(
        filtered_df[amount_column]
    )

    # Filter 1: Minimum amount
    min_amount = st.sidebar.number_input(
        "Minimum amount",
        value=float(filtered_df[amount_column].min())
    )

    # Filter 2: Maximum amount
    max_amount = st.sidebar.number_input(
        "Maximum amount",
        value=float(filtered_df[amount_column].max())
    )

    filtered_df = filtered_df[
        filtered_df[amount_column].between(
            min_amount, max_amount
        )
    ]

    # Filter 3: Category
    category_column = st.sidebar.selectbox(
        "Category filter column",
        [None] + list(df.columns)
    )

    if category_column is not None:
        category_values = (
            filtered_df[category_column]
            .dropna().unique().tolist()
        )

        selected_categories = st.sidebar.multiselect(
            "Select categories",
            category_values
        )

        if selected_categories:
            filtered_df = filtered_df[
                filtered_df[category_column].isin(
                    selected_categories
                )
            ]

    # Filter 4: Positive or negative amounts
    sign_filter = st.sidebar.selectbox(
        "Amount type",
        ["All nonzero amounts",
         "Positive amounts only",
         "Negative amounts only"]
    )

    if sign_filter == "Positive amounts only":
        filtered_df = filtered_df[
            filtered_df[amount_column] > 0
        ]

    elif sign_filter == "Negative amounts only":
        filtered_df = filtered_df[
            filtered_df[amount_column] < 0
        ]

    filtered_df = filtered_df[
        filtered_df[amount_column].notna()
        & (filtered_df[amount_column] != 0)
    ]

    if filtered_df.empty:
        st.warning("No valid data remains after filtering.")
        st.stop()

    values = filtered_df[amount_column]

    # Summary statistics
    st.header("Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Original Records", len(df))
    col2.metric("Analyzed Records", len(filtered_df))
    col3.metric("Total Amount", f"{values.sum():,.2f}")
    col4.metric("Average Amount", f"{values.mean():,.2f}")

    summary = values.describe().to_frame(
        name="Value"
    )

    st.dataframe(summary)

    st.download_button(
        "Download Summary Statistics",
        summary.to_csv().encode("utf-8"),
        file_name="summary_statistics.csv",
        mime="text/csv"
    )

    st.subheader("Filtered Data")
    st.dataframe(filtered_df)

    st.download_button(
        "Download Filtered Data",
        filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_data.csv",
        mime="text/csv"
    )

    statistical_results = []

    # First digit and first two digit analysis
    for digits in [1, 2]:

        if digits == 1:
            title = "First Digit Analysis"
        else:
            title = "First Two Digits Analysis"

        st.header(title)

        results = calculate_benford_statistics(
            values, digits
        )

        ks_statistic, critical_value = calculate_ks(
            results
        )

        display_results = results.copy()

        for column in ["percent", "benford", "difference"]:
            display_results[column] *= 100

        display_results.columns = [
            "Digit",
            "Count",
            "Actual %",
            "Benford %",
            "Difference (percentage points)"
        ]

        st.dataframe(display_results.round(4))

        # Visualization
        fig, ax = plt.subplots(figsize=(14, 5))

        x = np.arange(len(results))
        width = 0.4

        ax.bar(
            x - width / 2,
            results["percent"] * 100,
            width,
            label="Actual",
            color="steelblue"
        )

        ax.bar(
            x + width / 2,
            results["benford"] * 100,
            width,
            label="Benford",
            color="orange"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            results["digit"],
            rotation=90 if digits == 2 else 0,
            fontsize=7 if digits == 2 else 10
        )

        ax.set_xlabel("Leading Digits")
        ax.set_ylabel("Percentage")
        ax.set_title(title + ": Actual vs. Benford")
        ax.legend()

        fig.tight_layout()

        st.pyplot(fig)

        # Download visualization
        image_buffer = io.BytesIO()

        fig.savefig(
            image_buffer,
            format="png",
            dpi=200,
            bbox_inches="tight"
        )

        st.download_button(
            "Download Visualization",
            image_buffer.getvalue(),
            file_name=f"benford_{digits}_digit_chart.png",
            mime="image/png",
            key=f"chart_{digits}"
        )

        plt.close(fig)

        # Statistical analysis
        st.subheader("KS Statistical Test")

        st.write("Significance level: 5% (alpha = 0.05)")

        st.write(
            "Null hypothesis: The actual distribution "
            "follows Benford's distribution."
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "KS Statistic",
            f"{ks_statistic:.6f}"
        )

        c2.metric(
            "Critical Value",
            f"{critical_value:.6f}"
        )

        if ks_statistic > critical_value:
            conclusion = (
                "Reject the null hypothesis. "
                "The actual distribution is statistically "
                "different from Benford's distribution "
                "using the course's KS decision rule."
            )
            st.warning(conclusion)

        else:
            conclusion = (
                "Fail to reject the null hypothesis. "
                "There is insufficient evidence that the "
                "actual distribution is statistically "
                "different from Benford's distribution."
            )
            st.success(conclusion)

        statistical_results.append({
            "Analysis": title,
            "Sample Size": len(values),
            "KS Statistic": ks_statistic,
            "Critical Value": critical_value,
            "Significance Level": 0.05,
            "Conclusion": conclusion
        })

        # Digits to investigate
        st.subheader("Digits with Largest Differences")

        investigation = display_results.copy()

        investigation["Absolute Difference"] = (
            investigation[
                "Difference (percentage points)"
            ].abs()
        )

        investigation = investigation.sort_values(
            "Absolute Difference",
            ascending=False
        ).head(5)

        st.dataframe(investigation.round(4))

        st.download_button(
            "Download Benford Analysis",
            display_results.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=f"benford_{digits}_digit_analysis.csv",
            mime="text/csv",
            key=f"analysis_{digits}"
        )

    # Download statistical conclusions
    st.header("Statistical Results Summary")

    statistical_df = pd.DataFrame(statistical_results)

    st.dataframe(statistical_df)

    st.download_button(
        "Download Statistical Results",
        statistical_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="ks_statistical_results.csv",
        mime="text/csv"
    )

    st.info(
        "Benford deviations identify potential areas "
        "for further investigation. They do not "
        "automatically indicate fraud. The KS critical "
        "value follows the course method and is "
        "conservative for discrete digit distributions."
    )

else:
    st.info("Upload a CSV or Excel file to begin.")
