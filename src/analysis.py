import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# =========================================================
# LOAD DATA
# =========================================================
def load_data():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_path, "data", "retail_sales.csv")

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["Quantity"] = pd.to_numeric(df["Quantity"])
    df["Price"] = pd.to_numeric(df["Price"])
    df["Total_Sales"] = pd.to_numeric(df["Total_Sales"])

    return df


# =========================================================
# DESCRIPTIVE ANALYSIS
# =========================================================
def descriptive_analysis(df):

    print("\n========== DESCRIPTIVE ANALYSIS ==========\n")

    total_revenue = df["Total_Sales"].sum()
    total_orders = df["Invoice_ID"].nunique()
    unique_customers = df["Customer_ID"].nunique()
    aov = total_revenue / total_orders

    print("Total Revenue:", round(total_revenue, 2))
    print("Total Orders:", total_orders)
    print("Unique Customers:", unique_customers)
    print("Average Order Value:", round(aov, 2))

    if aov > df["Total_Sales"].median():
        print("Interpretation: Customers tend to make relatively high-value purchases.")
    else:
        print("Interpretation: Revenue may rely more on transaction volume than order size.")

    # Revenue by category
    category_revenue = df.groupby("Category")["Total_Sales"].sum().sort_values(ascending=False)
    print("\nRevenue by Category:\n", category_revenue)

    category_revenue.plot(kind="bar", title="Revenue by Category")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.show()

    # Pareto Analysis
    customer_revenue = df.groupby("Customer_ID")["Total_Sales"].sum().sort_values(ascending=False)
    cumulative = customer_revenue.cumsum() / customer_revenue.sum()

    top_20_cutoff = int(len(customer_revenue) * 0.2)
    top_20_share = customer_revenue.head(top_20_cutoff).sum() / total_revenue * 100

    print("\nTop 20% customers contribute:", round(top_20_share, 2), "% of revenue")

    if top_20_share > 60:
        print("Interpretation: High revenue concentration risk.")
    else:
        print("Interpretation: Revenue is relatively diversified.")

    plt.plot(cumulative.values)
    plt.axhline(0.8, linestyle="--")
    plt.title("Pareto Analysis (Customer Contribution)")
    plt.ylabel("Cumulative Revenue %")
    plt.tight_layout()
    plt.show()

    # Monthly revenue trend
    df["Month"] = df["Date"].dt.to_period("M")
    monthly_revenue = df.groupby("Month")["Total_Sales"].sum()
    monthly_revenue.index = monthly_revenue.index.to_timestamp()

    monthly_revenue.plot(marker="o", title="Monthly Revenue Trend")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.show()

    monthly_growth = monthly_revenue.pct_change() * 100
    avg_growth = monthly_growth.mean()

    print("\nAverage Monthly Growth Rate (%):", round(avg_growth, 2))

    if avg_growth > 0:
        print("Interpretation: Overall revenue trend is upward.")
    else:
        print("Interpretation: Revenue growth is stagnating or declining.")


# =========================================================
# REGRESSION MODEL
# =========================================================
def regression_model(df):

    print("\n========== REGRESSION MODEL ==========\n")

    X = df[["Quantity", "Price"]]
    y = df["Total_Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print("R2 Score:", round(r2, 4))
    print("MAE:", round(mae, 2))

    if r2 > 0.9:
        print("Interpretation: Sales are strongly explained by price and quantity.")
    elif r2 > 0.5:
        print("Interpretation: Moderate predictive strength.")
    else:
        print("Interpretation: Additional features may be needed for better prediction.")


# =========================================================
# TIME SERIES FORECAST
# =========================================================
def time_series_forecast(df):

    print("\n========== TIME SERIES FORECAST ==========\n")

    df["Month"] = df["Date"].dt.to_period("M")
    monthly_sales = df.groupby("Month")["Total_Sales"].sum().reset_index()
    monthly_sales["Month"] = monthly_sales["Month"].dt.to_timestamp()

    monthly_sales["Time_Index"] = np.arange(len(monthly_sales))

    X = monthly_sales[["Time_Index"]]
    y = monthly_sales["Total_Sales"]

    model = LinearRegression()
    model.fit(X, y)

    future_months = 6
    future_index = np.arange(len(monthly_sales), len(monthly_sales) + future_months)
    future_predictions = model.predict(future_index.reshape(-1, 1))

    for i, value in enumerate(future_predictions, 1):
        print(f"Forecast Month {i}:", round(value, 2))

    if future_predictions[-1] > y.iloc[-1]:
        print("Interpretation: Forecast indicates continued growth.")
    else:
        print("Interpretation: Forecast suggests potential slowdown.")

    plt.plot(monthly_sales["Month"], y, marker="o", label="Actual")

    future_dates = pd.date_range(
        monthly_sales["Month"].iloc[-1],
        periods=future_months + 1,
        freq="ME"
    )[1:]

    plt.plot(future_dates, future_predictions, marker="o", linestyle="--", label="Forecast")

    plt.title("Monthly Revenue Forecast")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# MAIN
# =========================================================
def main():
    df = load_data()
    descriptive_analysis(df)
    regression_model(df)
    time_series_forecast(df)


if __name__ == "__main__":
    main()
