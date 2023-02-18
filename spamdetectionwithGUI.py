import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score


# Load the dataset
df = pd.read_csv("spam_assassin.csv")

# Extract the text messages and labels
text_messages = df["text"]
labels = df["target"]

# Convert the text messages into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_messages)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=15)

# Train the logistic regression model
clf = LogisticRegression(max_iter=160000)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the confusion matrix and accuracy
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

# Create the GUI
root = tk.Tk()
root.title("Spam Classification using Logistic Regression")
root.geometry("400x250")

# Create a label and entry for the message
message_label = ttk.Label(root, text="Enter the subject of the email:")
message_label.pack(side="top", padx=10, pady=10)
message_entry = ttk.Entry(root, width=50)
message_entry.pack(side="top", padx=50)

# Function to classify the message as spam or ham
def classify_message():
    message = message_entry.get()
    message_vector = vectorizer.transform([message])
    prediction = clf.predict(message_vector)
    if prediction [0] == 0:
        result = "This is a legitimate mail with a probability of "+ str(f"{(acc * 100):.2f}%")
    else:
        result = "This is a Spam with a probability of "+str(f"{(acc * 100):.2f}%")
    messagebox.showinfo("Classification Result", result)
    message_entry.delete(0, 'end')

# Create a button to classify the message
classify_button = ttk.Button(root, text="Classify", command=classify_message)
classify_button.pack(side="top", padx=10, pady=10)

# Run the GUI
root.mainloop()