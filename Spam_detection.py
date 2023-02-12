import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
df = pd.read_csv("spam_ham_dataset.csv")

# Extract the text messages and labels
text_messages = df["text"]
labels = df["label_num"]

# Convert the text messages into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_messages)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)

# Train the logistic regression model
clf = LogisticRegression(max_iter=5000)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the confusion matrix and accuracy
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

# Plot the confusion matrix using matplotlib
plt.imshow(cm, cmap='binary')
plt.title("Confusion Matrix")
plt.xticks([0, 1], ["Ham", "Spam"])
plt.yticks([0, 1], ["Ham", "Spam"])
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

# Print the accuracy of the model
print("Accuracy:", acc)

# Function to predict whether a new message is ham or spam
def predict_spam_ham(message):
    message = [message]
    message_vector = vectorizer.transform(message)
    prediction = clf.predict(message_vector)
    if prediction == [0]:
        return "Ham"
    else:
        return "Spam"
    
# Predict whether the following messages are ham or spam
input_message = input("Enter the message you want to know if spam or not: ")
print(predict_spam_ham(input_message))
