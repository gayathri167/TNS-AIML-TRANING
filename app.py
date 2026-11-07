
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

from backend import get_sales_data, predict_sales

st.set_page_config(
    page_title="Walmart Sales Forecasting",
    layout="wide"
)

# Currency conversion
rate = 83

# Load data
data = get_sales_data()

data["Sales_INR"] = data["Weekly_Sales"] * rate

# Title
st.title("Walmart Sales Forecasting")
st.write("Sales Analysis and Prediction Using Machine Learning")

# Sidebar menu
st.sidebar.title("Menu")

choice = st.sidebar.radio(
    "Select Option",
    ["Dashboard", "Sales Analysis", "Prediction"]
)

# Dashboard
if choice == "Dashboard":

    st.subheader("Sales Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Stores", data["Store"].nunique())
    col2.metric("Departments", data["Dept"].nunique())

    col3.metric(
        "Total Sales",
        "₹{:,.0f}".format(data["Sales_INR"].sum())
    )

    col4.metric(
        "Average Sales",
        "₹{:,.0f}".format(data["Sales_INR"].mean())
    )

    st.divider()

    st.subheader("Weekly Sales Trend")

    sales = data.groupby("Date")["Sales_INR"].sum()

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(sales.index, sales.values)

    ax.set_xlabel("Date")
    ax.set_ylabel("Sales in Rupees")
    ax.set_title("Weekly Sales")

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    st.divider()

    col1, col2 = st.columns(2)

    # Store pie chart
    with col1:

        st.subheader("Store-wise Sales")

        store_sales = data.groupby("Store")["Sales_INR"].sum()

        top_stores = store_sales.nlargest(5)

        fig, ax = plt.subplots()

        ax.pie(
            top_stores,
            labels=top_stores.index,
            autopct="%1.1f%%"
        )

        st.pyplot(fig)

    # Department pie chart
    with col2:

        st.subheader("Department-wise Sales")

        dept_sales = data.groupby("Dept")["Sales_INR"].sum()

        top_depts = dept_sales.nlargest(5)

        fig, ax = plt.subplots()

        ax.pie(
            top_depts,
            labels=top_depts.index,
            autopct="%1.1f%%"
        )

        st.pyplot(fig)


# Sales Analysis
elif choice == "Sales Analysis":

    st.subheader("Sales Analysis")

    store = st.selectbox(
        "Select Store",
        sorted(data["Store"].unique())
    )

    store_data = data[data["Store"] == store]

    st.metric(
        "Total Store Sales",
        "₹{:,.2f}".format(store_data["Sales_INR"].sum())
    )

    st.subheader("Department Sales")

    dept_sales = store_data.groupby("Dept")["Sales_INR"].sum()

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(dept_sales.index.astype(str), dept_sales.values)

    ax.set_xlabel("Department")
    ax.set_ylabel("Sales in Rupees")
    ax.set_title("Department-wise Sales")

    plt.tight_layout()

    st.pyplot(fig)

    if st.button("Show Sales Data"):

        st.dataframe(
            store_data[["Store", "Dept", "Date", "Sales_INR"]],
            use_container_width=True
        )


# Sales Prediction
elif choice == "Prediction":

    st.subheader("Weekly Sales Prediction")

    col1, col2 = st.columns(2)

    with col1:

        store = st.selectbox(
            "Select Store",
            sorted(data["Store"].unique())
        )

        dept = st.selectbox(
            "Select Department",
            sorted(data["Dept"].unique())
        )

    with col2:

        prediction_date = st.date_input(
            "Select Date",
            value=date(2013, 1, 1)
        )

        holiday = st.selectbox(
            "Holiday Week",
            ["No", "Yes"]
        )

    if st.button("Predict Sales"):

        try:

            result = predict_sales(
                int(store),
                int(dept),
                prediction_date,
                holiday == "Yes"
            )

            result_inr = result * rate

            st.success("Prediction completed")

            st.metric(
                "Predicted Weekly Sales",
                "₹{:,.2f}".format(result_inr)
            )

        except Exception as e:

            st.error(str(e))


st.divider()

st.write("Walmart Sales Forecasting using Linear Regression")
