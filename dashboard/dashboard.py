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

# Filter by date
st.sidebar.header("Filter by Date")
selected_date = st.sidebar.date_input("Select a date:", value=pd.to_datetime("2011-01-01"), min_value=day_df['dteday'].min(), max_value=day_df['dteday'].max())

# Filter data based on selected date
filtered_hour_df = hour_df[hour_df['dteday'] == pd.to_datetime(selected_date)]
filtered_day_df = day_df[day_df['dteday'] == pd.to_datetime(selected_date)]

# Display filtered data
if not filtered_hour_df.empty:
    st.subheader(f"Filtered Hourly Data for {selected_date}")
    st.write(filtered_hour_df)

# Filter by season
st.sidebar.header("Filter by Season")
selected_season = st.sidebar.selectbox("Select a season:", options=['All', 'Spring', 'Summer', 'Fall', 'Winter'])

if selected_season != 'All':
    season_mapping = {'Spring': 1, 'Summer': 2, 'Fall': 3, 'Winter': 4}
    filtered_day_df = day_df[day_df['season'] == season_mapping[selected_season]]

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

# Average hourly rentals
average_hourly_rentals = hourly_rentals / day_df.groupby('weekday').count()['dteday'].max()

fig_hourly = plt.figure(figsize=(12, 6))
average_hourly_rentals.plot(kind='bar', ax=plt.gca())
plt.title('Rata-rata Peminjaman Sepeda berdasarkan Jam (Hari Kerja vs Akhir Pekan)')
plt.xlabel('Jam di satu hari')
plt.ylabel('Rata-rata Peminjaman')
plt.xticks(rotation=0)
plt.legend(['Hari Kerja', 'Akhir Pekan'])
st.pyplot(fig_hourly)

# Average rentals by weather
weather_rentals = day_df.groupby('weathersit')['cnt'].mean().reset_index()
fig_weather = plt.figure(figsize=(8, 5))
sns.barplot(x='weathersit', y='cnt', data=weather_rentals)
plt.title('Rata-rata Peminjaman Sepeda berdasarkan Kondisi Cuaca')
plt.xlabel('Kondisi Cuaca (1: Clear, 2: Misty, 3: Light Rain, 4: Heavy Rain)')
plt.ylabel('Rata-rata Peminjaman')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Jernih', 'Kabut', 'Hujan Ringan', 'Hujan Berat'])
st.pyplot(fig_weather)

# Yearly Trends
st.header("Perbandingan Peminjaman Berdasarkan Tahun")
yearly_rentals = day_df.groupby('yr')['cnt'].sum().reset_index()
fig_yearly = plt.figure(figsize=(8, 5))
sns.barplot(x='yr', y='cnt', data=yearly_rentals)
plt.title('Total Peminjaman Sepeda Berdasarkan Tahun')
plt.xlabel('Tahun')
plt.ylabel('Total Peminjaman')
st.pyplot(fig_yearly)

# Monthly Trends
st.header("Tren Peminjaman Berdasarkan Bulan")
monthly_rentals = day_df.groupby('mnth')['cnt'].sum().reset_index()
fig_monthly = plt.figure(figsize=(10, 5))
sns.barplot(x='mnth', y='cnt', data=monthly_rentals)
plt.title('Total Peminjaman Sepeda Berdasarkan Bulan')
plt.xlabel('Bulan (1: Jan, 2: Feb, ..., 12: Des)')
plt.ylabel('Total Peminjaman')
plt.xticks(ticks=np.arange(0, 12, 1), labels=['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'])
st.pyplot(fig_monthly)

# Correlation Analysis
correlation_matrix = day_df.corr()
fig_corr = plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Matriks Korelasi')
st.pyplot(fig_corr)

# Conclusions
st.header("Insights & Conclusions")
st.write("1. Total peminjaman sepeda terbanyak berada pada musim gugur (fall).")
st.write("2. Total peminjaman sepeda paling sedikit berada pada musim semi (spring).")
st.write("3. Rata-rata peminjaman sepeda pada akhir pekan menunjukkan peningkatan signifikan pada jam 17, yang merupakan waktu pulang kerja.")
st.write("4. Terdapat hubungan yang signifikan antara temperatur dan jumlah peminjaman; semakin tinggi temperatur, semakin banyak orang yang meminjam sepeda.")
st.write("5. Tren peminjaman terlihat meningkat dari tahun ke tahun, menunjukkan popularitas layanan peminjaman sepeda yang terus berkembang.")
st.write("6. Peminjaman bulanan menunjukkan pola musiman, dengan bulan-bulan hangat (Mei - Sep) menunjukkan rata-rata peminjaman yang lebih tinggi.")
st.write("7. Cuaca berpengaruh besar terhadap peminjaman sepeda, dengan kondisi cuaca yang cerah (1) menghasilkan rata-rata peminjaman jauh lebih tinggi dibandingkan dengan kondisi hujan.")
st.write("8. Pemahaman yang lebih baik tentang pola, waktu, dan kondisi yang mempengaruhi peminjaman sepeda dapat membantu pengelola layanan peminjaman sepeda dalam merencanakan strategi dan meningkatkan layanan.")
# Halo kak, terima kasih feedbacknya
