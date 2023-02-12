import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load the dataset from excel file
df = pd.read_excel("spam_ham.xlsx")

# Split the dataset into features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict the target on test set
y_pred = clf.predict(X_test)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")
plt.show()

# Plot the ROC curve
from sklearn.metrics import roc_auc_score, roc_curve

y_pred_prob = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Input a sentence or paragraph
new_data = input("Enter a sentence or paragraph: ")

# Convert the input data into the required format
new_data = np.array(new_data).reshape(1, -1)

# Predict whether the input is a spam message or not
prediction = clf.predict(new_data)

if prediction == "spam":
    print("The message is a spam message.")
else:
    print("The message is not a spam message.")