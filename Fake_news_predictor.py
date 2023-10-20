import streamlit as st
import joblib
import re
import spacy

# Load the trained model from the specific path
model_path = 'fake_news_model.pkl'
model = joblib.load(model_path)

# Load the TF-IDF vectorizer from the specific path
vectorizer_path = 'tfidf_vectorizer.pkl'
vectorizer = joblib.load(vectorizer_path)

# Load Spacy model
nlp = spacy.load('en_core_web_sm')

# Create a function to preprocess user input and make predictions
def predict_fake_news(user_input):
    # Preprocess user input
    user_input = user_input.lower()
    user_input = re.sub(r'[^a-zA-Z0-9\s]', '', user_input)
    user_input = [token.lemma_ for token in nlp(user_input)]
    user_input = ' '.join(user_input)

    # Convert to TF-IDF vectorized form using the loaded vectorizer
    tfidf_input = vectorizer.transform([user_input])

    # Make predictions using the loaded model
    prediction = model.predict(tfidf_input)
    return prediction[0]

def main():
    st.title("Fake News Detection App")

    st.title("Fake News Detection")

    user_input = st.text_area("Enter the news article:")
    if st.button("Predict"):
        prediction = predict_fake_news(user_input)
        if prediction == 0:
            st.error("This news is likely genuine.")
        else:
            st.success("This news is likely fake.")

    st.write("Please enter a news article, and we'll predict if it's genuine or fake.")

if __name__ == '__main__':
    main()
