"""
===============================================================================
Task 6: Advanced Data Visualization for Data Analysts
Dataset: Titanic (via Seaborn)
Libraries: Matplotlib, Seaborn, Plotly Express, Pandas, NumPy
===============================================================================
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

# Set global styles for Matplotlib and Seaborn
sns.set_theme(style="whitegrid")
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9

# Load Dataset
titanic = sns.load_dataset("titanic")

# ===============================================================================
# PART 1: Matplotlib & Right Chart Selection
# ===============================================================================
print("=== PART 1: Matplotlib Visualizations ===")

# 1. Histogram of Passenger Ages
plt.figure(figsize=(8, 5))
plt.hist(titanic["age"].dropna(), bins=30, color="skyblue", edgecolor="black")
plt.title("Distribution of Passenger Ages (Titanic)", fontsize=14, fontweight="bold")
plt.xlabel("Age (Years)")
plt.ylabel("Frequency (Passenger Count)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# 2. Bar Chart of Passengers by Pclass
plt.figure(figsize=(7, 5))
class_counts = titanic["pclass"].value_counts().sort_index()
plt.bar(
    class_counts.index.astype(str),
    class_counts.values,
    color=["#2b5c8f", "#4682b4", "#6baed6"],
    edgecolor="black",
)
plt.title("Passenger Distribution Across Ticket Classes", fontsize=14, fontweight="bold")
plt.xlabel("Passenger Class (Pclass)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()

# 3. Scatter Plot between Age and Fare
plt.figure(figsize=(8, 5))
plt.scatter(
    titanic["age"], titanic["fare"], alpha=0.5, color="purple", edgecolors="none"
)
plt.title("Scatter Plot of Age vs. Ticket Fare", fontsize=14, fontweight="bold")
plt.xlabel("Age (Years)")
plt.ylabel("Fare ($)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ===============================================================================
# PART 2: Advanced Seaborn Visualizations
# ===============================================================================
print("\n=== PART 2: Advanced Seaborn Visualizations ===")

# 1. Correlation Matrix Heatmap
plt.figure(figsize=(8, 6))
numeric_cols = titanic.select_dtypes(include=[np.number])
corr_matrix = numeric_cols.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap of Numeric Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# 2. Boxplot of Age divided by Pclass
plt.figure(figsize=(8, 5))
sns.boxplot(x="pclass", y="age", data=titanic, palette="Blues")
plt.title("Age Distribution across Passenger Classes", fontsize=14, fontweight="bold")
plt.xlabel("Passenger Class (Pclass)")
plt.ylabel("Age (Years)")
plt.tight_layout()
plt.show()

# 3. Countplot of Survival grouped by Sex
plt.figure(figsize=(7, 5))
sns.countplot(x="survived", hue="sex", data=titanic, palette="Set2")
plt.title("Survival Count Grouped by Gender", fontsize=14, fontweight="bold")
plt.xlabel("Survival Status (0 = Did Not Survive, 1 = Survived)")
plt.ylabel("Passenger Count")
plt.xticks(ticks=[0, 1], labels=["Deceased (0)", "Survived (1)"])
plt.tight_layout()
plt.show()

# 4. Pairplot of Key Numerical Columns
pairplot_fig = sns.pairplot(
    titanic[["age", "fare", "pclass"]].dropna(),
    hue="pclass",
    palette="viridis",
    diag_kind="kde",
)
pairplot_fig.fig.suptitle(
    "Pairplot of Age, Fare, and Pclass", y=1.02, fontsize=14, fontweight="bold"
)
plt.show()

# ===============================================================================
# PART 3: Interactive Visualizations with Plotly
# ===============================================================================
print("\n=== PART 3: Plotly Interactive Visualizations ===")

# Prepare aggregated data for Plotly Bar Chart
pclass_summary = (
    titanic["pclass"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "pclass", "pclass": "count"})
)

# 1. Interactive Bar Chart of Passenger Counts by Pclass
fig1 = px.bar(
    pclass_summary,
    x="pclass",
    y="count",
    labels={"pclass": "Passenger Class", "count": "Passenger Count"},
    title="<b>Interactive Passenger Count by Ticket Class</b>",
    color="pclass",
    color_continuous_scale="Blues",
)
fig1.update_layout(template="plotly_white")
fig1.show()

# 2. Interactive Scatter Plot of Age vs Fare colored by Survival Status
titanic_plotly = titanic.copy()
titanic_plotly["survived_label"] = titanic_plotly["survived"].map(
    {0: "Died", 1: "Survived"}
)

fig2 = px.scatter(
    titanic_plotly,
    x="age",
    y="fare",
    color="survived_label",
    hover_data=["pclass", "sex"],
    title="<b>Interactive Scatter Plot: Age vs Fare (by Survival Status)</b>",
    labels={
        "age": "Age (Years)",
        "fare": "Fare ($)",
        "survived_label": "Survival Status",
    },
    color_discrete_map={"Died": "#EF553B", "Survived": "#00CC96"},
)
fig2.update_layout(template="plotly_white")
fig2.show()

# 3. Interactive Histogram of Age Distribution divided by Sex
fig3 = px.histogram(
    titanic,
    x="age",
    color="sex",
    barmode="overlay",
    opacity=0.6,
    title="<b>Interactive Age Distribution Grouped by Sex</b>",
    labels={"age": "Age (Years)", "sex": "Gender"},
)
fig3.update_layout(template="plotly_white")
fig3.show()

# 4. Exporting Interactive Chart to Standalone HTML File
output_filename = "chart.html"
fig2.write_html(output_filename)
print(f"Successfully exported interactive scatter plot to '{output_filename}'.")

# ===============================================================================
# PART 4: Selecting the Right Chart for Real Business Questions
# ===============================================================================
print("\n=== PART 4: Real Business Questions Solutions ===")

# --- Business Question 1 ---
plt.figure(figsize=(8, 5))
sns.boxplot(
    x="pclass", y="fare", data=titanic, showfliers=False, palette="Set3"
)
plt.title(
    "Fare Distribution across Ticket Classes (Outliers Omitted for Scale)",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Passenger Class")
plt.ylabel("Fare ($)")
plt.tight_layout()
plt.show()

# --- Business Question 2 ---
survived_counts = titanic["survived"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(
    survived_counts,
    labels=["Deceased", "Survived"],
    autopct="%1.1f%%",
    startangle=90,
    colors=["#e74c3c", "#2ecc71"],
    explode=(0.05, 0),
)
plt.title("Overall Proportion of Passenger Survival", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# --- Business Question 3 ---
titanic["family_size"] = titanic["sibsp"] + titanic["parch"] + 1

plt.figure(figsize=(8, 5))
sns.barplot(
    x="family_size", y="survived", data=titanic, ci=None, palette="viridis"
)
plt.title("Survival Rate by Family Size Onboard", fontsize=13, fontweight="bold")
plt.xlabel("Family Size (Self + SibSp + Parch)")
plt.ylabel("Survival Probability")
plt.tight_layout()
plt.show()

# --- Business Question 4 ---
plt.figure(figsize=(7, 5))
sns.barplot(
    x="embarked",
    y="fare",
    data=titanic,
    ci=None,
    palette="magma",
    order=["C", "Q", "S"],
)
plt.title("Average Ticket Fare by Embarkation Port", fontsize=13, fontweight="bold")
plt.xlabel("Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)")
plt.ylabel("Mean Fare ($)")
plt.tight_layout()
plt.show()