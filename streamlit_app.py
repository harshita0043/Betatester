import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure the Gemini API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("Ever AI: Product Improvement & Beta Version Creator")
st.write("Use generative AI, data scraping, and automation to improve your product and create beta suggestions.")

# Default URL to analyze if no input is given
default_url = "https://example.com"  # Replace this with a real URL or a placeholder for automatic scraping

# Feature 1: Scrape Website Data (Text)
def scrape_website(url):
    try:
        st.write("Starting to scrape the website...")
        response = requests.get(url, timeout=10)  # Timeout after 10 seconds
        if response.status_code != 200:
            st.error(f"Error: Unable to fetch the page. Status code: {response.status_code}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Scrape all text content
        text_data = soup.get_text()

        return text_data
    except requests.exceptions.Timeout:
        st.error("The request timed out. Please try again later.")
    except Exception as e:
        st.error(f"Error scraping website: {e}")
    return None

# Feature 2: Visualize Basic Data Insights (Word Frequency Analysis)
def analyze_word_frequency(text_data):
    if not text_data:
        st.warning("No text data available to visualize.")
        return

    words = text_data.split()
    word_counts = Counter(words)
    common_words = word_counts.most_common(10)

    # Create bar plot for word frequency
    words, counts = zip(*common_words)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(words), y=list(counts))
    plt.title("Most Common Words in Scraped Text")
    st.pyplot()

# Feature 3: Sentiment Analysis
def analyze_sentiment(text_data):
    sentiment = TextBlob(text_data).sentiment
    return sentiment

# Feature 4: Keyword Extraction (TF-IDF)
def extract_keywords(text_data):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text_data])
    feature_names = vectorizer.get_feature_names_out()
    dense = X.todense()
    keyword_scores = dense.tolist()[0]
    keywords = [(feature_names[i], keyword_scores[i]) for i in range(len(feature_names))]
    keywords = sorted(keywords, key=lambda x: x[1], reverse=True)[:5]
    return keywords

# Feature 5: Generate Custom AI Response (Improvement Suggestion)
def generate_custom_response(prompt):
    st.write("Generating AI response for improvements...")
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# Feature 6: Generate Beta Version Roadmap (based on text data)
def generate_beta_roadmap(text_data):
    st.write("Generating Beta Version Roadmap...")
    prompt = f"Create a detailed beta version roadmap based on the following product data: {text_data[:300]}"
    roadmap = generate_custom_response(prompt)
    return roadmap

# Feature 7: Smart Product Improvement Suggestion
def generate_smart_product_suggestion(text_data, keywords, sentiment):
    """ Generate smart product suggestions based on the analysis of scraped data. """
    if sentiment.polarity < 0:
        suggestion = "There seems to be negative sentiment. Focus on improving customer experience, addressing pain points, and user feedback."
    else:
        suggestion = "The sentiment is positive. You should focus on adding new features, enhancing performance, and improving scalability."
    
    common_keywords = [keyword[0] for keyword in keywords]
    
    if 'usability' in common_keywords or 'interface' in common_keywords:
        suggestion = "Consider improving the user interface for better accessibility and usability."
    elif 'performance' in common_keywords or 'speed' in common_keywords:
        suggestion = "Optimize the product for better performance and responsiveness."
    elif 'support' in common_keywords or 'help' in common_keywords:
        suggestion = "Expand your customer support options, such as FAQs, live chat, or in-app assistance."
    elif 'security' in common_keywords or 'privacy' in common_keywords:
        suggestion = "Strengthen security features to build user trust, focusing on data privacy and protection."
    
    return suggestion

# Streamlit UI Logic
# Option to use default website or input a custom URL
url_input = st.text_input("Enter the competitor/product website URL (Leave blank for default URL):", "")

# Use default URL if no input is provided
url = url_input if url_input else default_url

# Initialize session state for text data
if "text_data" not in st.session_state:
    st.session_state.text_data = None

# Button to trigger scraping and analysis
if st.button("Analyze and Suggest Improvements"):
    # Scrape website
    st.session_state.text_data = scrape_website(url)
    
    if st.session_state.text_data:
        # Show the basic website data
        st.write(f"Scraped content from {url}.")
        
        # Show first few characters of the scraped text
        st.write("Scraped Text (First 500 characters):")
        st.write(st.session_state.text_data[:500])
        
        # Word Frequency Analysis
        analyze_word_frequency(st.session_state.text_data)
        
        # Sentiment Analysis
        sentiment = analyze_sentiment(st.session_state.text_data)
        st.write(f"Sentiment Analysis: Polarity = {sentiment.polarity}, Subjectivity = {sentiment.subjectivity}")
        
        # Extract Keywords
        keywords = extract_keywords(st.session_state.text_data)
        st.write("Top 5 Keywords:")
        st.write(keywords)
        
        # Generate Smart Product Suggestion
        smart_suggestion = generate_smart_product_suggestion(st.session_state.text_data, keywords, sentiment)
        st.write(f"Smart Product Suggestion: {smart_suggestion}")
        
        # Generate a detailed Beta Version Roadmap based on scraped data
        beta_roadmap = generate_beta_roadmap(st.session_state.text_data)
        st.write("Suggested Beta Version Roadmap:")
        st.write(beta_roadmap)

# Button to generate overall summary
if st.button("Generate Overall Advisory"):
    if st.session_state.text_data:
        # Summarize the analysis
        summary_prompt = f"Please summarize the analysis of the following product data: {st.session_state.text_data[:500]}"
        summary = generate_custom_response(summary_prompt)
        st.write("Summary of Analysis:")
        st.write(summary)
    else:
        st.error("No data available to generate advisory. Please check the website URL or try again.")
