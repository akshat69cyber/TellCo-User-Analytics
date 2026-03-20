import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="TellCo User Analytics Dashboard",
    layout="wide"
)


df = pd.read_excel(r"C:\Users\Asus\Downloads\telcom_data (2).xlsx")

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))


st.title("TellCo User Analytics Dashboard")

st.markdown("""
This dashboard presents insights from the **Telecommunication User Analytics Project**.
It covers:
- User Overview
- User Engagement
- User Experience
- User Satisfaction
""")


st.header("User Overview Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Handsets Used")
    top_10_handsets = df['Handset Type'].value_counts().head(10)
    st.bar_chart(top_10_handsets)

with col2:
    st.subheader("Top 3 Handset Manufacturers")
    top_3_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
    st.bar_chart(top_3_manufacturers)


st.header("User Engagement Analysis")

user_agg = df.groupby('MSISDN').agg(
    sessions=('Bearer Id', 'count'),
    duration=('Dur. (ms)', 'sum'),
    total_dl=('Total DL (Bytes)', 'sum'),
    total_ul=('Total UL (Bytes)', 'sum')
)

user_agg['total_data'] = user_agg['total_dl'] + user_agg['total_ul']

st.subheader("Top 10 Users by Total Data Usage")
st.dataframe(
    user_agg.sort_values('total_data', ascending=False).head(10)
)


st.header("User Experience Analysis")

experience_agg = df.groupby('MSISDN').agg(
    avg_rtt=('Avg RTT DL (ms)', 'mean'),
    avg_throughput=('Avg Bearer TP DL (kbps)', 'mean'),
    avg_tcp=('TCP DL Retrans. Vol (Bytes)', 'mean')
)

st.subheader("Sample Network Experience Metrics")
st.dataframe(experience_agg.head(10))

# Plot Throughput Distribution
st.subheader("Distribution of Average Throughput")

fig, ax = plt.subplots()
sns.histplot(experience_agg['avg_throughput'], bins=50, ax=ax)
ax.set_xlabel("Average Throughput (kbps)")
ax.set_ylabel("Number of Users")
st.pyplot(fig)


st.header("User Satisfaction Summary")

st.markdown("""
User satisfaction is influenced by:
- High engagement (more sessions & data usage)
- Good network experience (low RTT, high throughput)

Users who are both **highly engaged** and have **good network quality**
are the most satisfied customers.
""")


st.header("Final Business Insight")

st.success("""
**Recommendation:** BUY TellCo  

**Reason:**  
- Strong user engagement  
- High demand for data-intensive applications  
- Clear opportunities for network and service optimization  
""")
