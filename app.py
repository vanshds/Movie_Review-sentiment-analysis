import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('sentiment_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('max_sequence_len.pkl', 'rb') as f:
    max_sequence_len = pickle.load(f)

# Netflix dark theme CSS without card
st.markdown("""
    <style>
    /* Remove Streamlit's default padding */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Netflix-like dark background */
    .stApp {
        background-color: #141414;
        color: #e5e5e5;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
    }

    /* Title styling with Netflix red */
    .title {
        font-size: 3rem;
        font-weight: 900;
        color: #e50914;
        text-align: center;
        margin-top: 40px;
        margin-bottom: 10px;
        letter-spacing: 2px;
        text-transform: uppercase;
        user-select: none;
    }

    /* Subheading styling */
    .subheading {
        font-size: 1.2rem;
        color: #bbbbbb;
        text-align: center;
        margin-bottom: 30px;
        user-select: none;
    }

    /* Text area styling */
    textarea {
        background-color: #333333 !important;
        color: #e5e5e5 !important;
        border: 2px solid #e50914 !important;
        border-radius: 10px !important;
        padding: 18px !important;
        font-size: 1rem !important;
        resize: vertical !important;
        min-height: 140px !important;
        max-height: 280px !important;
        width: 100% !important;
        outline: none !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }

    /* Label for textarea */
    label[for="input_area"] > div {
        color: #e50914 !important;  /* Netflix red */
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 12px;
        user-select: none;
    }

    /* Button styling */
    div.stButton > button {
        background-color: #e50914;
        color: white;
        padding: 14px 36px;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 700;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
        margin-top: 20px;
        width: 100%;
        user-select: none;
    }
    div.stButton > button:hover {
        background-color: #b20710;
    }

    /* Success message styling */
    .stAlert-success {
        background-color: #006400 !important; /* dark green */
        color: #d4edda !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 15px 20px;
        margin-top: 20px;
        text-align: center;
    }

    /* Error message styling */
    .stAlert-error {
        background-color: #8b0000 !important; /* dark red */
        color: #f8d7da !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 15px 20px;
        margin-top: 20px;
        text-align: center;
    }

    /* Warning message styling */
    .stAlert-warning {
        background-color: #9a7d00 !important; /* dark yellow */
        color: #fff3cd !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 15px 20px;
        margin-top: 20px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">Movie Review System</div>', unsafe_allow_html=True)

# Subheading to guide user
st.markdown('<div class="subheading">Enter your movie review below to predict its sentiment.</div>', unsafe_allow_html=True)

# Input area
user_input = st.text_area('Movie Review', '', key='input_area')

# Button and prediction logic
if st.button('Predict Sentiment'):
    if user_input.strip() == '':
        st.warning('Please enter a review.')
    else:
        test_seq = tokenizer.texts_to_sequences([user_input])
        test_pad = pad_sequences(test_seq, maxlen=max_sequence_len, padding='pre')
        prediction = model.predict(test_pad)
        if prediction[0][0] > 0.5:
            st.success('Predicted Sentiment: Positive Review')
        else:
            st.error('Predicted Sentiment: Negative Review')







