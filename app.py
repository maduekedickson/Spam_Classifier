import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

# Load the saved model
model_filename = 'spam_classifier_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Set of stopwords
STOPWORDS = set(stopwords.words('english'))

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^0-9a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text

# Streamlit app setup
st.image('image.jpeg', caption='Spam Classifier', use_column_width=True)

st.title("SMS Spam Classifier")

# Create a text input box
user_input = st.text_input("Enter a message to classify:", "")

# Create a button to check for spam
if st.button("Check Spam"):
    if user_input:
        # Preprocess the input text
        clean_input = clean_text(user_input)
        
        # Make a prediction using the model
        prediction = model.predict([clean_input])[0]
        
        # Display the result with custom colors
        if prediction == "spam":
            st.markdown(f"""
                <div style="background-color: red; padding: 10px; border-radius: 5px; color: white; text-align: center;">
                    <h3>This message is classified as SPAM.</h3>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color: green; padding: 10px; border-radius: 5px; color: white; text-align: center;">
                    <h3>This message is classified as NOT SPAM.</h3>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Please enter a message to classify.")
