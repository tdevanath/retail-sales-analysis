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

    print(f"Total Revenue: {total_revenue:,.2f}")
    print(f"Total Orders: {total_orders}")
    print(f"Unique Customers: {unique_customers}")
    print(f"Average Order Value: {aov:,.2f}")

    # Category Revenue
    category_revenue = df.groupby("Category")["Total_Sales"].sum().sort_values(ascending=False)

    top_category = category_revenue.idxmax()
    bottom_category = category_revenue.idxmin()

    top_share = (category_revenue.max() / total_revenue) * 100
    bottom_share = (category_revenue.min() / total_revenue) * 100

    print("\nRevenue by Category:\n", category_revenue)

    print("\nInsight:")
    print(f"- {top_category} is the primary revenue engine contributing {top_share:.2f}% of total revenue.")
    print(f"- {bottom_category} contributes only {bottom_share:.2f}% and represents relative underperformance.")

    category_revenue.plot(kind="bar", title="Revenue by Category")
    plt.tight_layout()
    plt.show()

    # Monthly Revenue
    df["Month"] = df["Date"].dt.to_period("M")
    monthly_revenue = df.groupby("Month")["Total_Sales"].sum()
    monthly_revenue.index = monthly_revenue.index.to_timestamp()

    best_month = monthly_revenue.idxmax()
    worst_month = monthly_revenue.idxmin()

    best_month_name = best_month.strftime("%B %Y")
    worst_month_name = worst_month.strftime("%B %Y")

    print("\nMonthly Revenue Breakdown:\n", monthly_revenue)

    print(f"\nHighest Revenue Month: {best_month_name} ({monthly_revenue.max():,.2f})")
    print(f"Lowest Revenue Month: {worst_month_name} ({monthly_revenue.min():,.2f})")

    monthly_revenue.plot(marker="o", title="Monthly Revenue Trend")
    plt.tight_layout()
    plt.show()

    growth_rate = monthly_revenue.pct_change() * 100
    avg_growth = growth_rate.mean()

    print(f"\nAverage Monthly Growth Rate: {avg_growth:.2f}%")

    print("\nInsight:")
    print(f"- Revenue peaked in {best_month_name}, indicating seasonal or campaign-driven demand.")
    print(f"- Weakest performance occurred in {worst_month_name}.")
    print("- Trend analysis supports seasonal inventory planning.")

    return best_month, total_revenue


# =========================================================
# PARETO ANALYSIS
# =========================================================
def pareto_analysis(df, total_revenue):

    print("\n========== PARETO ANALYSIS ==========\n")

    customer_sales = df.groupby("Customer_ID")["Total_Sales"].sum().sort_values(ascending=False)

    top_20 = int(len(customer_sales) * 0.2)
    contribution = customer_sales.head(top_20).sum() / total_revenue * 100

    print(f"Top 20% customers contribute {contribution:.2f}% of total revenue.")

    if contribution > 60:
        print(f"- Revenue concentration risk exists (heavy dependency on top customers).")
    else:
        print(f"- Revenue diversified across customer base, structural stability observed.")

    cumulative = customer_sales.cumsum() / customer_sales.sum()
    plt.plot(cumulative.values)
    plt.axhline(0.8, linestyle="--")
    plt.title("Pareto Curve")
    plt.tight_layout()
    plt.show()


# =========================================================
# DIAGNOSTIC ANALYSIS
# =========================================================
def diagnostic_analysis(df, best_month):

    print("\n========== DIAGNOSTIC ANALYSIS ==========\n")

    correlation = df[["Quantity", "Price", "Total_Sales"]].corr()

    qty_corr = correlation.loc["Quantity", "Total_Sales"]
    price_corr = correlation.loc["Price", "Total_Sales"]

    print("Correlation Matrix:\n", correlation)

    print("\nInsight:")
    print(f"- Price correlation with revenue: {price_corr:.2f}")
    print(f"- Quantity correlation with revenue: {qty_corr:.2f}")

    if price_corr > qty_corr:
        print("- Pricing has stronger influence on revenue than volume.")
    else:
        print("- Sales volume drives revenue more strongly than pricing.")

    peak_data = df[df["Date"].dt.to_period("M") == best_month.to_period("M")]
    peak_category = peak_data.groupby("Category")["Total_Sales"].sum()

    print("\nPeak Month Category Breakdown:\n", peak_category)

    dominant_peak_category = peak_category.idxmax()

    print(f"\nInsight:")
    print(f"- {dominant_peak_category} drove the revenue spike in {best_month.strftime('%B %Y')}.")
    print("- Indicates category-specific seasonal acceleration.")


# =========================================================
# PREDICTIVE ANALYSIS – REGRESSION
# =========================================================
def regression_model(df):

    print("\n========== PREDICTIVE ANALYSIS (REGRESSION) ==========\n")

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

    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")

    print("\nInsight:")
    print(f"- {r2*100:.2f}% of revenue variation is explained by price and quantity.")
    print(f"- Average prediction error is {mae:.2f} currency units.")


# =========================================================
# FORECASTING
# =========================================================
def forecasting(df):

    print("\n========== PREDICTIVE ANALYSIS (FORECAST) ==========\n")

    df["Month"] = df["Date"].dt.to_period("M")
    monthly_sales = df.groupby("Month")["Total_Sales"].sum().reset_index()
    monthly_sales["Month"] = monthly_sales["Month"].dt.to_timestamp()
    monthly_sales["Time_Index"] = np.arange(len(monthly_sales))

    X = monthly_sales[["Time_Index"]]
    y = monthly_sales["Total_Sales"]

    model = LinearRegression()
    model.fit(X, y)

    future_index = np.arange(len(monthly_sales), len(monthly_sales) + 6)
    future_df = pd.DataFrame({"Time_Index": future_index})
    forecast = model.predict(future_df)

    for i, value in enumerate(forecast, 1):
        print(f"Forecast Month {i}: {value:,.2f}")

    projected_growth = ((forecast[-1] - y.iloc[-1]) / y.iloc[-1]) * 100

    print(f"\nProjected 6-Month Revenue Growth: {projected_growth:.2f}%")

    if projected_growth > 0:
        print("- Revenue expected to continue upward trajectory.")
    else:
        print("- Potential stagnation or decline projected.")

    future_dates = pd.date_range(monthly_sales["Month"].iloc[-1], periods=7, freq="ME")[1:]
    plt.plot(monthly_sales["Month"], y, label="Actual")
    plt.plot(future_dates, forecast, linestyle="--", label="Forecast")
    plt.legend()
    plt.title("Revenue Forecast")
    plt.tight_layout()
    plt.show()


# =========================================================
# PRESCRIPTIVE ANALYSIS
# =========================================================
def prescriptive_analysis():

    print("\n========== PRESCRIPTIVE STRATEGIC RECOMMENDATIONS ==========\n")

    print("1. Increase marketing allocation toward dominant revenue category.")
    print("2. Replicate conditions from peak month performance.")
    print("3. Optimize pricing strategy based on strong revenue-price correlation.")
    print("4. Scale inventory in anticipation of projected growth.")
    print("5. Maintain diversified customer acquisition to preserve stability.")
    print("6. Monitor weaker months proactively for corrective campaigns.")


# =========================================================
# MAIN
# =========================================================
def main():
    df = load_data()
    best_month, total_revenue = descriptive_analysis(df)
    pareto_analysis(df, total_revenue)
    diagnostic_analysis(df, best_month)
    regression_model(df)
    forecasting(df)
    prescriptive_analysis()


if __name__ == "__main__":
    main()
