import pandas as pd

df = pd.read_excel(r"C:\Users\Asus\Downloads\telcom_data (2).xlsx")
print(df.head())
print(df.columns)
print("Missing values before cleaning:")
print(df.isnull().sum())

df = df.fillna(df.mean(numeric_only=True))

print("Missing values after cleaning:")
print(df.isnull().sum())

top_10_handsets = df['Handset Type'].value_counts().head(10)
print("Top 10 Handsets:")
print(top_10_handsets)

top_3_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
print("Top 3 Manufacturers:")
print(top_3_manufacturers)

top_3_names = top_3_manufacturers.index.tolist()

top_5_per_manufacturer = (
    df[df['Handset Manufacturer'].isin(top_3_names)]
    .groupby(['Handset Manufacturer', 'Handset Type'])
    .size()
    .sort_values(ascending=False)
)

print("Top 5 Handsets per Top 3 Manufacturers:")
print(top_5_per_manufacturer.groupby(level=0).head(5))

user_agg = df.groupby('MSISDN').agg(
    sessions_count=('Bearer Id', 'count'),
    total_duration=('Dur. (ms)', 'sum'),
    total_dl=('Total DL (Bytes)', 'sum'),
    total_ul=('Total UL (Bytes)', 'sum'),
    social_media=('Social Media DL (Bytes)', 'sum'),
    google=('Google DL (Bytes)', 'sum'),
    email=('Email DL (Bytes)', 'sum'),
    youtube=('Youtube DL (Bytes)', 'sum'),
    netflix=('Netflix DL (Bytes)', 'sum'),
    gaming=('Gaming DL (Bytes)', 'sum'),
    other=('Other DL (Bytes)', 'sum')
)

user_agg['total_data_volume'] = user_agg['total_dl'] + user_agg['total_ul']

print(user_agg.head())
print(user_agg.describe())
user_agg['duration_decile'] = pd.qcut(user_agg['total_duration'], 5, labels=False)

decile_summary = user_agg.groupby('duration_decile')['total_data_volume'].sum()
print(decile_summary)
apps = ['social_media','google','email','youtube','netflix','gaming','other']
print(user_agg[apps].corr())

import matplotlib.pyplot as plt
import seaborn as sns

# Histogram: Total data volume per user
plt.figure(figsize=(8,5))
sns.histplot(user_agg['total_data_volume'], bins=50)
plt.title("Distribution of Total Data Volume per User")
plt.xlabel("Total Data Volume")
plt.ylabel("Number of Users")
plt.show()

# Boxplot: Outliers in total data usage
plt.figure(figsize=(8,5))
sns.boxplot(x=user_agg['total_data_volume'])
plt.title("Outliers in Total Data Volume")
plt.show()

for app in apps:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=user_agg[app], y=user_agg['total_data_volume'])
    plt.title(f"{app} vs Total Data Usage")
    plt.xlabel(app)
    plt.ylabel("Total Data Volume")
    plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = user_agg[apps]

# Standardize data
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

# ================================
# TASK 2: USER ENGAGEMENT ANALYSIS
# ================================

from sklearn.cluster import KMeans

# Top 10 users by engagement metrics
print("Top 10 users by number of sessions:")
print(user_agg.sort_values('sessions_count', ascending=False).head(10))

print("\nTop 10 users by total session duration:")
print(user_agg.sort_values('total_duration', ascending=False).head(10))

print("\nTop 10 users by total data usage:")
print(user_agg.sort_values('total_data_volume', ascending=False).head(10))

# Normalize engagement metrics
engagement_metrics = user_agg[
    ['sessions_count', 'total_duration', 'total_data_volume']
]

scaler = StandardScaler()
engagement_scaled = scaler.fit_transform(engagement_metrics)

# K-Means (k = 3)
kmeans_3 = KMeans(n_clusters=3, random_state=42)
user_agg['engagement_cluster'] = kmeans_3.fit_predict(engagement_scaled)

print("\nUsers per engagement cluster:")
print(user_agg['engagement_cluster'].value_counts())

# Cluster statistics
cluster_summary = user_agg.groupby('engagement_cluster')[
    ['sessions_count', 'total_duration', 'total_data_volume']
].agg(['min', 'max', 'mean', 'sum'])

print("\nEngagement cluster summary:")
print(cluster_summary)

# Top 10 users per application
for app in apps:
    print(f"\nTop 10 users for {app}:")
    print(user_agg.sort_values(app, ascending=False).head(10))

# Top 3 most used applications
total_app_usage = user_agg[apps].sum().sort_values(ascending=False)

print("\nTotal usage per application:")
print(total_app_usage)

top_3_apps = total_app_usage.head(3)

plt.figure(figsize=(8,5))
top_3_apps.plot(kind='bar')
plt.title("Top 3 Most Used Applications")
plt.xlabel("Application")
plt.ylabel("Total Data Usage")
plt.show()

# Elbow Method
inertia = []

for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(engagement_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 10), inertia, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()



# ================================
# TASK 3: EXPERIENCE ANALYTICS
# ================================

from sklearn.cluster import KMeans

# --------------------------------
# STEP 1: AGGREGATE EXPERIENCE METRICS PER USER
# --------------------------------

experience_agg = df.groupby('MSISDN').agg(
    avg_tcp_retrans=('TCP DL Retrans. Vol (Bytes)', 'mean'),
    avg_rtt=('Avg RTT DL (ms)', 'mean'),
    avg_throughput=('Avg Bearer TP DL (kbps)', 'mean'),
    handset_type=('Handset Type', 'first')
)

# Handle missing values again (mean for numeric)
experience_agg[['avg_tcp_retrans','avg_rtt','avg_throughput']] = (
    experience_agg[['avg_tcp_retrans','avg_rtt','avg_throughput']]
    .fillna(experience_agg[['avg_tcp_retrans','avg_rtt','avg_throughput']].mean())
)

print(experience_agg.head())


# --------------------------------
# STEP 2: TOP, BOTTOM & MOST FREQUENT VALUES
# --------------------------------

print("\nTop 10 TCP Retransmission values:")
print(df['TCP DL Retrans. Vol (Bytes)'].sort_values(ascending=False).head(10))

print("\nBottom 10 TCP Retransmission values:")
print(df['TCP DL Retrans. Vol (Bytes)'].sort_values().head(10))

print("\nMost frequent TCP Retransmission values:")
print(df['TCP DL Retrans. Vol (Bytes)'].value_counts().head(10))


print("\nTop 10 RTT values:")
print(df['Avg RTT DL (ms)'].sort_values(ascending=False).head(10))

print("\nBottom 10 RTT values:")
print(df['Avg RTT DL (ms)'].sort_values().head(10))

print("\nMost frequent RTT values:")
print(df['Avg RTT DL (ms)'].value_counts().head(10))


print("\nTop 10 Throughput values:")
print(df['Avg Bearer TP DL (kbps)'].sort_values(ascending=False).head(10))

print("\nBottom 10 Throughput values:")
print(df['Avg Bearer TP DL (kbps)'].sort_values().head(10))

print("\nMost frequent Throughput values:")
print(df['Avg Bearer TP DL (kbps)'].value_counts().head(10))


# --------------------------------
# STEP 3: EXPERIENCE BY HANDSET TYPE
# --------------------------------

# Throughput distribution per handset type
plt.figure(figsize=(10,5))
sns.boxplot(x='handset_type', y='avg_throughput', data=experience_agg)
plt.xticks(rotation=90)
plt.title("Average Throughput per Handset Type")
plt.show()

# TCP retransmission per handset type
plt.figure(figsize=(10,5))
sns.boxplot(x='handset_type', y='avg_tcp_retrans', data=experience_agg)
plt.xticks(rotation=90)
plt.title("Average TCP Retransmission per Handset Type")
plt.show()


# --------------------------------
# STEP 4: K-MEANS CLUSTERING (k = 3)
# --------------------------------

experience_metrics = experience_agg[
    ['avg_tcp_retrans', 'avg_rtt', 'avg_throughput']
]

scaler_exp = StandardScaler()
experience_scaled = scaler_exp.fit_transform(experience_metrics)

kmeans_exp = KMeans(n_clusters=3, random_state=42)
experience_agg['experience_cluster'] = kmeans_exp.fit_predict(experience_scaled)

print("\nUsers per experience cluster:")
print(experience_agg['experience_cluster'].value_counts())


# --------------------------------
# STEP 5: CLUSTER INTERPRETATION DATA
# --------------------------------

experience_cluster_summary = experience_agg.groupby('experience_cluster')[
    ['avg_tcp_retrans','avg_rtt','avg_throughput']
].mean()

print("\nExperience cluster summary (mean values):")
print(experience_cluster_summary)


# ================================
# TASK 4: SATISFACTION ANALYSIS
# ================================

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --------------------------------
# STEP 4.1: ENGAGEMENT SCORE
# --------------------------------
# Engagement score = distance from LEAST engaged cluster

# Find least engaged cluster (based on total data usage)
least_engaged_cluster = (
    user_agg.groupby('engagement_cluster')['total_data_volume']
    .mean()
    .idxmin()
)

# Centroid of least engaged cluster
eng_centroid = kmeans_3.cluster_centers_[least_engaged_cluster]

# Engagement score (Euclidean distance)
user_agg['engagement_score'] = np.linalg.norm(
    engagement_scaled - eng_centroid,
    axis=1
)

print("Engagement score calculated")


# --------------------------------
# STEP 4.1 (b): EXPERIENCE SCORE
# --------------------------------
# Experience score = distance from WORST experience cluster

# Find worst experience cluster (low throughput, high RTT)
worst_experience_cluster = (
    experience_agg.groupby('experience_cluster')['avg_throughput']
    .mean()
    .idxmin()
)

# Centroid of worst experience cluster
exp_centroid = kmeans_exp.cluster_centers_[worst_experience_cluster]

# Experience score (Euclidean distance)
experience_agg['experience_score'] = np.linalg.norm(
    experience_scaled - exp_centroid,
    axis=1
)

print("Experience score calculated")


# --------------------------------
# STEP 4.2: SATISFACTION SCORE
# --------------------------------
# Merge engagement + experience scores

final_scores = user_agg.merge(
    experience_agg[['experience_score']],
    left_index=True,
    right_index=True
)

# Satisfaction score = average of engagement & experience score
final_scores['satisfaction_score'] = (
    final_scores['engagement_score'] +
    final_scores['experience_score']
) / 2

print("Top 10 satisfied customers:")
print(
    final_scores.sort_values(
        'satisfaction_score',
        ascending=False
    ).head(10)
)


# --------------------------------
# STEP 4.3: REGRESSION MODEL
# --------------------------------
# Predict satisfaction score

X = final_scores[['engagement_score', 'experience_score']]
y = final_scores['satisfaction_score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

y_pred = reg_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Regression RMSE:", rmse)


# --------------------------------
# STEP 4.4: K-MEANS (k = 2)
# --------------------------------

kmeans_sat = KMeans(n_clusters=2, random_state=42)
final_scores['satisfaction_cluster'] = kmeans_sat.fit_predict(
    final_scores[['engagement_score', 'experience_score']]
)

print("Users per satisfaction cluster:")
print(final_scores['satisfaction_cluster'].value_counts())


# --------------------------------
# STEP 4.5: AGGREGATE SCORES PER CLUSTER
# --------------------------------

cluster_score_summary = final_scores.groupby('satisfaction_cluster')[
    ['engagement_score', 'experience_score', 'satisfaction_score']
].mean()

print("Cluster-wise average scores:")
print(cluster_score_summary)


# --------------------------------
# STEP 4.6: EXPORT TO MYSQL
# --------------------------------
# ⚠️ Update username, password, DB name if needed

from sqlalchemy import create_engine

engine = create_engine(
    "mysql+pymysql://root:password@localhost/tellco_db"
)

final_scores.reset_index().to_sql(
    "user_satisfaction_scores",
    engine,
    if_exists="replace",
    index=False
)

print("Data exported to MySQL successfully")


# --------------------------------
# STEP 4.7: MODEL TRACKING (BASIC)
# --------------------------------

import datetime

tracking_log = {
    "model": "LinearRegression",
    "rmse": rmse,
    "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "features": list(X.columns)
}

tracking_df = pd.DataFrame([tracking_log])
tracking_df.to_csv("model_tracking_log.csv", index=False)

print("Model tracking log saved")
