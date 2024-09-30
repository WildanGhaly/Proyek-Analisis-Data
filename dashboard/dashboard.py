import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load Data
hour_df = pd.read_csv('./data/hour.csv')
day_df = pd.read_csv('./data/day.csv')

# Convert date columns to datetime
day_df['dteday'] = pd.to_datetime(day_df['dteday'])
hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])

# Function to assess data
def assess_data(input_df):
    total_null = input_df.isnull().sum()
    total_duplicate = input_df.duplicated().sum()
    total_na = input_df.isna().sum()
    return total_null, total_duplicate, total_na

# Data Assessment
hour_df_null, hour_df_duplicate, hour_df_na = assess_data(hour_df)

# Dashboard Layout
st.title("Bike Sharing Data Analysis Dashboard")

# Total Records & Columns
st.header("Dataset Overview")
st.subheader("Hourly Data")
st.write("Total Records:", hour_df.shape[0])
st.write("Total Columns:", hour_df.shape[1])
st.write("Missing Values:", hour_df_null[hour_df_null > 0])

st.subheader("Daily Data")
st.write("Total Records:", day_df.shape[0])
st.write("Total Columns:", day_df.shape[1])
st.write("Missing Values:", day_df.isnull().sum().sum())

# Distribution Plots
st.header("Distribution of Bike Rentals")
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(hour_df['cnt'], bins=30, kde=True, ax=ax[0])
ax[0].set_title('Distribusi Peminjaman Sepeda per Jam')
ax[0].set_xlabel('Jumlah Peminjaman')
ax[0].set_ylabel('Frekuensi')

sns.histplot(day_df['cnt'], bins=30, kde=True, ax=ax[1])
ax[1].set_title('Distribusi Peminjaman Sepeda per Hari')
ax[1].set_xlabel('Jumlah Peminjaman')
ax[1].set_ylabel('Frekuensi')

st.pyplot(fig)

# Seasonal Analysis
seasonal_rentals = day_df.groupby('season')['cnt'].sum().reset_index()
fig_seasonal = plt.figure(figsize=(8, 5))
sns.barplot(x='season', y='cnt', data=seasonal_rentals)
plt.title('Total Peminjaman Sepeda berdasarkan Musim')
plt.xlabel('Musim (1:spring, 2:summer, 3:fall, 4:winter)')
plt.ylabel('Total Peminjaman')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Spring', 'Summer', 'Fall', 'Winter'])
st.pyplot(fig_seasonal)

# Hourly Rentals on Weekdays vs Weekends
hour_df['is_weekend'] = hour_df['weekday'].apply(lambda x: 1 if x in [5, 6] else 0)
hourly_rentals = hour_df.groupby(['hr', 'is_weekend'])['cnt'].sum().unstack()

fig_hourly = plt.figure(figsize=(12, 6))
hourly_rentals.plot(kind='bar', ax=plt.gca())
plt.title('Total Peminjaman Sepeda berdasarkan Jam (Hari Kerja vs Akhir Pekan)')
plt.xlabel('Jam di satu hari')
plt.ylabel('Total Peminjaman')
plt.xticks(rotation=0)
plt.legend(['Hari Kerja', 'Akhir Pekan'])
st.pyplot(fig_hourly)

# Conclusions
st.header("Insights & Conclusions")
st.write("1. Total peminjaman sepeda terbanyak berada pada musim gugur (fall).")
st.write("2. Total peminjaman sepeda paling sedikit berada pada musim semi (spring).")
st.write("3. Peminjaman sepeda pada hari kerja memiliki grafik yang mirip dengan akhir pekan namun dengan peningkatan signifikan pada jam 17.")
st.write("4. Semakin mendekati angka 1, semakin kuat korelasi positif antara dua variabel.")
st.write("5. Kebanyakan sepeda yang dipinjam ada di waktu siang hari.")
st.write("6. Musim mempengaruhi jumlah peminjaman sepeda dengan tren yang konsisten.")

