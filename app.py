import streamlit as st
import joblib

# Load the pre-trained vectorizer and model
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# Set the title of the Streamlit app
st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real.")

# Create a text area for user input
inputn = st.text_area("News Article:", "")

# Button to trigger the prediction
if st.button("Check News"):
    if inputn.strip():  # Check if the input is not empty
        # Transform the input using the vectorizer
        transform_input = vectorizer.transform([inputn])
        
        # Make a prediction using the model
        prediction = model.predict(transform_input)

        # Display the result based on the prediction
        if prediction[0] == 1:
            st.success("The News is Real!")
        else:
            st.error("The News is Fake!")
    else:
        st.warning("Please enter some text to analyze.")