import streamlit as st
import pandas as pd
import plotly.express as px

# Load the CSV data
df = pd.read_csv("sentiments_full2.csv")

# Title of the app
st.title("Analysis of Sentiments")

# Top 5 speakers by activity and their prevalent sentiment
st.subheader("Top 5 Speakers by Activity and Their Sentiment Breakdown")

# Calculate sentiment breakdown for the top 5 speakers
top5_speakers = df['ГОВОРИТЕЛ'].value_counts().head(5).index
top5_speaker_data = df[df['ГОВОРИТЕЛ'].isin(top5_speakers)]
speaker_sentiment_breakdown = top5_speaker_data.groupby(['ГОВОРИТЕЛ', 'SENTIMENT']).size().unstack().fillna(0)

# Remove 'neutral' sentiment and ensure all sentiment categories are present
for sentiment in ['negative', 'positive']:
    if sentiment not in speaker_sentiment_breakdown.columns:
        speaker_sentiment_breakdown[sentiment] = 0

# Convert counts to percentages
speaker_sentiment_percent = speaker_sentiment_breakdown[['negative', 'positive']].divide(speaker_sentiment_breakdown[['negative', 'positive']].sum(axis=1), axis=0) * 100

# Plotly stacked bar chart for sentiment breakdown of top 5 speakers
fig1 = px.bar(speaker_sentiment_percent.reset_index(), 
              x='ГОВОРИТЕЛ', y=['negative', 'positive'], 
              title="Sentiment Breakdown for Top 5 Speakers",
              labels={'value': 'Percentage', 'variable': 'Sentiment', 'ГОВОРИТЕЛ': 'Speaker'},
              height=400)
fig1.update_layout(barmode='stack')
st.plotly_chart(fig1)

# Analysis of sentiment based on parliamentary group
st.subheader("Analysis of Sentiment Based on Parliamentary Group")

# Calculate sentiment breakdown for each parliamentary group
group_sentiment_breakdown = df.groupby(['ПАРЛ_ГРУПА', 'SENTIMENT']).size().unstack().fillna(0)

# Remove 'neutral' sentiment and ensure all sentiment categories are present
for sentiment in ['negative', 'positive']:
    if sentiment not in group_sentiment_breakdown.columns:
        group_sentiment_breakdown[sentiment] = 0

# Convert counts to percentages
group_sentiment_percent = group_sentiment_breakdown[['negative', 'positive']].divide(group_sentiment_breakdown[['negative', 'positive']].sum(axis=1), axis=0) * 100

# Plotly stacked bar chart for sentiment breakdown of parliamentary groups
fig2 = px.bar(group_sentiment_percent.reset_index(), 
              x='ПАРЛ_ГРУПА', y=['negative', 'positive'], 
              title="Sentiment Breakdown for Parliamentary Groups",
              labels={'value': 'Percentage', 'variable': 'Sentiment', 'ПАРЛ_ГРУПА': 'Parliamentary Group'},
              height=500)
fig2.update_layout(barmode='stack')
st.plotly_chart(fig2)

# Speakers with only negative entries
st.subheader("Speakers with Only Negative Entries")

# Filter speakers who have only negative sentiments
all_negative_speakers = df.groupby('ГОВОРИТЕЛ').filter(lambda x: all(x['SENTIMENT'] == 'negative'))
all_negative_speaker_counts = all_negative_speakers['ГОВОРИТЕЛ'].value_counts()
fig3 = px.bar(all_negative_speaker_counts, title="Speakers with Only Negative Entries", labels={'value': 'Number of Negative Entries', 'index': 'Speaker'}, height=500)
st.plotly_chart(fig3)
