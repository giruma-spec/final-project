import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="INFO450 Final Project Dashboard",
    layout="wide"
)

st.title("INFO450 Final Project: Income and Work Hours Dashboard")

st.write("""
This dashboard analyzes income, work hours, education, and gender patterns using IPUMS data.
The app includes cleaned data, feature engineering, visual analysis, statistical comparison, and a simple predictive model.
""")

# Load data
df = pd.read_csv("IPUMS.csv")
df_occ = pd.read_csv("Occupation Codes.csv")

df["OCC"] = df["OCC"].astype(str)

merged_df = pd.merge(df, df_occ, left_on="OCC", right_on="OCC Code", how="left")

cleaned_df = merged_df.copy()

cleaned_df["Occupation Title"] = cleaned_df["Occupation Title"].fillna("Unspecified Occupation")
cleaned_df["OCC Code"] = cleaned_df["OCC Code"].fillna("0000")

cleaned_df["UHRSWORKT"] = cleaned_df["UHRSWORKT"].replace(999, np.nan)
cleaned_df["EDUC"] = cleaned_df["EDUC"].replace(999, np.nan)

columns_for_nan_check = ["SEX", "UHRSWORKT", "EDUC", "INCWAGE", "WKSWORK1"]
cleaned_df = cleaned_df.dropna(subset=columns_for_nan_check)

cleaned_df = cleaned_df.rename(columns={"INCWAGE": "AnnualIncome"})

cleaned_df["HourlyWage"] = cleaned_df["AnnualIncome"] / (
    cleaned_df["UHRSWORKT"] * cleaned_df["WKSWORK1"]
)

cleaned_df["HourlyWage"] = cleaned_df["HourlyWage"].replace(
    [float("inf"), -float("inf")], np.nan
)

cleaned_df["HourlyWage"] = cleaned_df["HourlyWage"].fillna(0)


final_columns = [
    "SEX",
    "UHRSWORKT",
    "EDUC",
    "AnnualIncome",
    "Occupation Title",
    "OCC Code",
    "HourlyWage",
]

df_final = cleaned_df[final_columns].copy()

df_final = df_final[
    (df_final["UHRSWORKT"] < 60) &
    (df_final["UHRSWORKT"] > 30)
]

df_final = df_final[
    (df_final["AnnualIncome"] < 250000) &
    (df_final["AnnualIncome"] > 10000)
]

df_final["Gender"] = df_final["SEX"].map({1: "Male", 2: "Female"})

# Sidebar filters
st.sidebar.header("Filters")

gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=df_final["Gender"].dropna().unique(),
    default=df_final["Gender"].dropna().unique()
)

income_range = st.sidebar.slider(
    "Annual Income Range",
    int(df_final["AnnualIncome"].min()),
    int(df_final["AnnualIncome"].max()),
    (
        int(df_final["AnnualIncome"].min()),
        int(df_final["AnnualIncome"].max())
    )
)

filtered_df = df_final[
    (df_final["Gender"].isin(gender_filter)) &
    (df_final["AnnualIncome"].between(income_range[0], income_range[1]))
]

header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", f"{len(filtered_df):,}")
col2.metric("Average Income", f"${filtered_df['AnnualIncome'].mean():,.0f}")
col3.metric("Average Hours Worked", f"{filtered_df['UHRSWORKT'].mean():.1f}")
col4.metric("Average Hourly Wage", f"${filtered_df['HourlyWage'].mean():.2f}")

st.write("""
Use the sidebar filters to adjust the data shown in the dashboard.
The charts below update based on the selected gender and income range.
""")

# Graphs
header("Visual Analysis")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("### Distribution of Hours Worked")
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    filtered_df["UHRSWORKT"].plot(kind="hist", bins=20, edgecolor="black", ax=ax1)
    ax1.set_xlabel("Hours Worked Per Week")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Weekly Work Hours")
    st.pyplot(fig1)

with chart_col2:
    st.markdown("### Distribution of Annual Income")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    filtered_df["AnnualIncome"].plot(kind="box", ax=ax2)
    ax2.set_ylabel("Annual Income")
    ax2.set_title("Annual Income Distribution")
    st.pyplot(fig2)

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.markdown("### Annual Income by Gender")
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    filtered_df.boxplot(column="AnnualIncome", by="Gender", ax=ax3)
    ax3.set_xlabel("Gender")
    ax3.set_ylabel("Annual Income")
    ax3.set_title("Annual Income by Gender")
    plt.suptitle("")
    st.pyplot(fig3)

with chart_col4:
    st.markdown("### Average Income by Education Level")
    avg_income_by_educ = filtered_df.groupby("EDUC")["AnnualIncome"].mean().sort_index()
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    avg_income_by_educ.plot(kind="bar", ax=ax4)
    ax4.set_xlabel("Education Level Code")
    ax4.set_ylabel("Average Annual Income")
    ax4.set_title("Average Annual Income by Education Level")
    plt.xticks(rotation=45)
    st.pyplot(fig4)

# Statistical Analysis
st.subheader("Statistical Analysis")

st.subheader("Statistical Analysis")

# 1. 95% Confidence Interval for Mean Annual Income
mean_income = df_final["AnnualIncome"].mean()
sem_income = df_final["AnnualIncome"].sem()
n_total = len(df_final)

ci_low, ci_high = stats.t.interval(
    0.95,
    df=n_total - 1,
    loc=mean_income,
    scale=sem_income
)

st.markdown("### 95% Confidence Interval for Mean Annual Income")

col1, col2, col3 = st.columns(3)

col1.metric("Mean Annual Income", f"${mean_income:,.2f}")
col2.metric("95% CI Lower Bound", f"${ci_low:,.2f}")
col3.metric("95% CI Upper Bound", f"${ci_high:,.2f}")

st.write(f"Sample size: **{n_total:,}**")

st.write("""
This confidence interval estimates the likely range for the true mean annual income in the population
based on the cleaned sample used in this project.
""")

# 2. Two-Sample T-Test: Male vs. Female Income
male_income = df_final[df_final["SEX"] == 1]["AnnualIncome"]
female_income = df_final[df_final["SEX"] == 2]["AnnualIncome"]

t_stat, p_value = stats.ttest_ind(
    male_income,
    female_income,
    equal_var=False
)

mean_male = male_income.mean()
mean_female = female_income.mean()
mean_difference = mean_male - mean_female

st.markdown("### Two-Sample T-Test: Male vs. Female Annual Income")

col4, col5, col6 = st.columns(3)

col4.metric("Male Mean Income", f"${mean_male:,.2f}")
col5.metric("Female Mean Income", f"${mean_female:,.2f}")
col6.metric("Difference in Means", f"${mean_difference:,.2f}")

col7, col8 = st.columns(2)

col7.metric("T-Statistic", f"{t_stat:.3f}")
col8.metric("P-Value", f"{p_value:.6f}")

if p_value < 0.05:
    st.success(
        "Result: The p-value is below 0.05, so we reject the null hypothesis. "
        "There is a statistically significant difference in mean annual income between males and females."
    )
else:
    st.info(
        "Result: The p-value is above 0.05, so we fail to reject the null hypothesis. "
        "There is no statistically significant difference in mean annual income between males and females."
    )

# Predictive Model
st.subheader("Predictive Model")

income_threshold = df_final["AnnualIncome"].quantile(0.75)
df_final["HighEarner"] = (df_final["AnnualIncome"] > income_threshold).astype(int)

features = ["SEX", "UHRSWORKT", "EDUC"]
target = "HighEarner"

X = df_final[features]
y = df_final[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.metric("Decision Tree Model Accuracy", f"{accuracy:.2%}")

feature_importances = pd.Series(
    dtree.feature_importances_,
    index=features
).sort_values(ascending=False)

st.write("### Feature Importance")
st.dataframe(feature_importances.rename("Importance"))

st.write("""
The predictive model estimates whether a person is a high earner based on gender, weekly hours worked,
education level. This makes the app more robust because it goes beyond
basic charts and includes a simple machine learning component.
""")
