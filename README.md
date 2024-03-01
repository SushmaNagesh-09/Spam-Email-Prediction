# Spam-Email-Prediction

### " The use case for building a spam email prediction model is to automatically filter and classify incoming emails as either spam or non-spam, thereby improving email management efficiency and protecting users from unwanted or potentially harmful content "

Environment and Tools Used:

   1. Language: Python

  2. Libraries: Pandas, Scikit-learn

  3. Functions: TfidfVectorizer, accuracy_score
  
  4. Algorithm: Logistic Regression

Workflow:

  1. Data Collection: Collect a dataset of emails labeled as spam or non-spam (ham).
   
  2. Data Pre-processing: Clean and preprocess the email data, including removing special characters, stop words, and performing stemming or lemmatization.

  3. Feature Extraction: Convert the text data into numerical form using techniques like TF-IDF (Term Frequency-Inverse Document Frequency). This converts each email into a vector representation.
   
  4. Model Training: Split the dataset into training and testing sets. Train a logistic regression model on the training data.
   
  5. Model Evaluation: Evaluate the performance of the trained model on the testing data using an accuracy score or other relevant metrics.
    
  6. Prediction: Use the trained model to predict whether new incoming emails are spam or non-spam.

### How Logistic Regression Works:

Logistic regression is a classification algorithm used to predict the probability of a binary outcome (e.g., spam or non-spam). It models the probability that an instance belongs to a particular class using the logistic function.

### Logistic Regression Algorithm Steps:

1. Model Training:

  *  Given a set of features X (TF-IDF vectors representing emails) and corresponding labels y (spam or non-spam), logistic regression learns the parameters θ that best fit the data.

  *  The logistic regression model calculates the probability p that a given email is spam using the sigmoid (logistic) function:

           p = 1 / 1+e^−z

           where, z = θ^T X

  *  The parameters θ are adjusted during training using optimization techniques like gradient descent to minimize the error between predicted probabilities and actual labels.

2. Model Prediction:

  *  After training, the logistic regression model can predict the probability that a new email is spam or non-spam.
  
  *  If the predicted probability is above a certain threshold (usually 0.5), the email is classified as spam; otherwise, it's classified as non-spam.
