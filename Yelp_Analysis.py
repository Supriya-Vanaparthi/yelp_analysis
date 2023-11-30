#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd

data_file_path = 'C:/Users/ankit/yelp_merged_data.csv'
checkin_df = pd.read_csv(data_file_path)

# Geting number of columns
num_columns = checkin_df.shape[1]

print(f"The number of columns in the dataset is: {num_columns}")
# Getting all the columns in the DataFrame
print("Name of the Columns in the dataset:")
for column in checkin_df.columns:
    print(column)

# Printing shape and details of the dataset
print("Shape of the dataset:")
print(checkin_df.shape)

# Printing Column Names:
print("Column names:")
print(checkin_df.columns)

# Printing Column Name with Data Type
print("Datatype of each column:")
print(checkin_df.dtypes)

# Printing first 5 rows of the dataset
print("Few dataset entries:")
print(checkin_df.head())

# Printing summart of the dataset
checkin_df.describe(include='all')


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt
#Adding and creating a new column named "Length" to count the number of words
checkin_df['length'] = checkin_df['text'].apply(len)
checkin_df.head()

graph = sns.FacetGrid(data=checkin_df,col='stars_x')
graph.map(plt.hist,'length',bins=50,color='blue')


# In[10]:


st_mean = checkin_df.groupby('stars_x').mean(numeric_only=True)
st_mean


# In[11]:


import matplotlib.pyplot as plt

selected_columns = ['useful', 'funny', 'cool']
stval_selected = st_mean[selected_columns]

stval_selected.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Mean Values for "useful", "funny", and "cool" by Stars')
plt.xlabel('Stars')
plt.ylabel('Mean Value')
plt.legend(title='Columns', bbox_to_anchor=(1, 1))

plt.show()


# In[12]:


st_mean.corr()


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = st_mean.corr()

# Creating a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()


# In[50]:


# Create the modified 'categories' column based on the mapping
checkin_df['modified_categories'] = checkin_df['categories'].apply(
    lambda x: 'American' if 'restaurant' in x.lower() and 'american' in x.lower() else
             'Mexican' if 'restaurant' in x.lower() and 'mexican' in x.lower() else
             'Italian' if 'restaurant' in x.lower() and 'italian' in x.lower() else
             'Japanese' if 'restaurant' in x.lower() and 'japanese' in x.lower() else
             'Chinese' if 'restaurant' in x.lower() and 'chinese' in x.lower() else
             'Thai' if 'restaurant' in x.lower() and 'thai' in x.lower() else
             'Mediterranean' if 'restaurant' in x.lower() and 'mediterranean' in x.lower() else
             'French' if 'restaurant' in x.lower() and 'french' in x.lower() else
             'Vietnamese' if 'restaurant' in x.lower() and 'vietnamese' in x.lower() else
             'Greek' if 'restaurant' in x.lower() and 'greek' in x.lower() else
             'Indian' if 'restaurant' in x.lower() and 'indian' in x.lower() else
             'Korean' if 'restaurant' in x.lower() and 'korean' in x.lower() else
             'Hawaiian' if 'restaurant' in x.lower() and 'hawaiian' in x.lower() else
             'African' if 'restaurant' in x.lower() and 'african' in x.lower() else
             'Spanish' if 'restaurant' in x.lower() and 'spanish' in x.lower() else
             'Middle_eastern' if 'restaurant' in x.lower() and 'middle_eastern' in x.lower() else
             'Non-Popular Cuisine'
)


modified_categories_counts = checkin_df['modified_categories'].value_counts()

modified_categories_counts = modified_categories_counts.sort_values(ascending=False)

# Plot the graph of value counts
plt.figure(figsize=(12, 8))
sns.barplot(x=modified_categories_counts, y=modified_categories_counts.index, palette='viridis')
plt.title('Count of Restaurants by Modified Categories')
plt.xlabel('Number of Restaurants')
plt.ylabel('Modified Categories')
plt.show()
# Count the categories in the 'modified_categories' column
#modified_categories_counts = checkin_df['modified_categories'].value_counts()

#for category, count in modified_categories_counts.items():
    #print(f"{category}: {count} restaurants")


# In[81]:


# CLASSIFICATION
checkin_df['review_category'] = pd.cut(checkin_df['stars_x'], bins=[0, 2, 3, 5], labels=['bad', 'average', 'good'], include_lowest=True)

# Filter the DataFrame based on the new 'review_category' column
data_classes = checkin_df[checkin_df['review_category'].isin(['bad', 'average', 'good'])]
data_classes.head()
print(data_classes.shape)

# Separate the dataset into X and Y for prediction
#x = data_classes['text'].iloc[:10000]
x = data_classes['text']
y = data_classes['review_category']
#y = data_classes['review_category'].iloc[:10000]

print(x.head())
print(y.head())


# In[70]:


import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import sent_tokenize
sent_tokenizer = sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer

# CLEANING THE REVIEWS - REMOVAL OF STOPWORDS AND PUNCTUATION
def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[77]:


import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer


x = checkin_df['text'].head(10000)
#x = checkin_df['text'] run this line of code for the whole data set, we tried it on our local machines
# but it is taking forever to complete preprocessing in our huge dataset of 2+ million rows as per requirements.

# CLEANING THE REVIEWS - REMOVAL OF STOPWORDS AND PUNCTUATION
def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# CONVERTING THE WORDS INTO A VECTOR
vocab = CountVectorizer(analyzer=text_process).fit(x)
print(len(vocab.vocabulary_))

r0 = x.iloc[0]
vocab0 = vocab.transform([r0])


# In[78]:


# Transforming the features using the vocabulary
transformed_x = vocab.transform(x)

# Displaying information about the sparse matrix:
print("The shape of the sparse matrix is: ", transformed_x.shape)
print("Number of non-zero occurrences: ", transformed_x.nnz)

# density of the matrix:
density_percentage = (transformed_x.nnz / (transformed_x.shape[0] * transformed_x.shape[1])) * 100
print("Density of the matrix: {:.2f}%".format(density_percentage))


# In[92]:


from sklearn.model_selection import train_test_split
# SPLITTING THE DATASET INTO TRAINING SET AND TESTING SET
x_subset = x[:10000]
y_subset = y[:10000]
x_train, x_test, y_train, y_test = train_test_split(x_subset, y_subset, test_size=0.2, random_state=101)


# In[90]:


vectorizer = CountVectorizer()

# Fit and transform the training data
x_train_vectorized = vectorizer.fit_transform(x_train)

# Transform the test data
x_test_vectorized = vectorizer.transform(x_test)


# In[93]:


# Multinomial Naive Bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)

# Transform the test data
x_test_tfidf = tfidf_vectorizer.transform(x_test)

# Creating and training the Multinomial Naive Bayes classifier
mnb_classifier = MultinomialNB()
mnb_classifier.fit(x_train_tfidf, y_train)

# Making predictions on the test set
predictions_mnb = mnb_classifier.predict(x_test_tfidf)

# Evaluating the Multinomial Naive Bayes classifier
print("Confusion Matrix for Multinomial Naive Bayes:")
print(confusion_matrix(y_test, predictions_mnb))

accuracy = accuracy_score(y_test, predictions_mnb) * 100
print("Accuracy Score: {:.2f}%".format(accuracy))

print("Classification Report:")
print(classification_report(y_test, predictions_mnb))


# In[95]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=101)
rf_classifier.fit(x_train_tfidf, y_train)


predictions_rf = rf_classifier.predict(x_test_tfidf)

# Evaluating the Random Forest Classifier
print("Confusion Matrix for Random Forest Classifier:")
print(confusion_matrix(y_test, predictions_rf))

accuracy = accuracy_score(y_test, predictions_rf) * 100
print("Accuracy Score: {:.2f}%".format(accuracy))

print("Classification Report:")
print(classification_report(y_test, predictions_rf))


# In[96]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(random_state=101)
dt_classifier.fit(x_train_tfidf, y_train)


predictions_dt = dt_classifier.predict(x_test_tfidf)


print("Confusion Matrix for Decision Tree Classifier:")
print(confusion_matrix(y_test, predictions_dt))

accuracy = accuracy_score(y_test, predictions_dt) * 100
print("Accuracy Score: {:.2f}%".format(accuracy))

print("Classification Report:")
print(classification_report(y_test, predictions_dt))


# In[97]:


# Support Vector Machine
from sklearn.svm import SVC
svm_classifier = SVC(kernel='linear', random_state=101)
svm_classifier.fit(x_train_tfidf, y_train)


predictions_svm = svm_classifier.predict(x_test_tfidf)

# Evaluating the Support Vector Machine (SVM) Classifier
print("Confusion Matrix for Support Vector Machine (SVM) Classifier:")
print(confusion_matrix(y_test, predictions_svm))

accuracy = accuracy_score(y_test, predictions_svm) * 100
print("Accuracy Score: {:.2f}%".format(accuracy))

print("Classification Report:")
print(classification_report(y_test, predictions_svm))


# In[100]:


# K Nearest Neighbour Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(x_train_tfidf, y_train)

# Making predictions on the test set
predictions_knn = knn_classifier.predict(x_test_tfidf)


print("Confusion Matrix for k-Nearest Neighbors (KNN) Classifier:")
print(confusion_matrix(y_test, predictions_knn))

accuracy = accuracy_score(y_test, predictions_knn) * 100
print("Accuracy Score: {:.2f}%".format(accuracy))

print("Classification Report:")
print(classification_report(y_test, predictions_knn))


# In[105]:


# XGBoost Classifier
#!pip install xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# Creating and training the XGBoost Classifier
xgb_classifier = xgb.XGBClassifier(
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=100,
    random_state=101
)
xgb_classifier.fit(x_train_tfidf, y_train_encoded)


predictions_xgb = xgb_classifier.predict(x_test_tfidf)
predictions_xgb_labels = label_encoder.inverse_transform(predictions_xgb)

# Evaluating the XGBoost Classifier
print("Confusion Matrix for XGBoost Classifier:")
print(confusion_matrix(y_test, predictions_xgb_labels))

accuracy = accuracy_score(y_test, predictions_xgb_labels) * 100
print("Accuracy Score: {:.2f}%".format(accuracy))

print("Classification Report:")
print(classification_report(y_test, predictions_xgb_labels))


# In[107]:


# MULTILAYER PERCEPTRON CLASSIFIER
from sklearn.neural_network import MLPClassifier

mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=101
)
mlp_classifier.fit(x_train_tfidf, y_train_encoded)

# Making predictions on the test set
predictions_mlp = mlp_classifier.predict(x_test_tfidf)

# Inverse transform the numerical labels back to original string labels
predictions_mlp_labels = label_encoder.inverse_transform(predictions_mlp)

# Evaluating the MLP Classifier
print("Confusion Matrix for MLP Classifier:")
print(confusion_matrix(y_test, predictions_mlp_labels))

accuracy = accuracy_score(y_test, predictions_mlp_labels) * 100
print("Accuracy Score: {:.2f}%".format(accuracy))

print("Classification Report:")
print(classification_report(y_test, predictions_mlp_labels))


# In[112]:


from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#graph for results comparison
results = {'Classifier': [], 'Accuracy': [], 'F1 Score': []}

# Gradient Boosting Classifier
gbi = GradientBoostingClassifier(learning_rate=0.1, max_depth=5, max_features=0.5, random_state=999999)
gbi.fit(x_train_tfidf, y_train)
pred_gbi = gbi.predict(x_test_tfidf)
accuracy_gbi = accuracy_score(y_test, pred_gbi)
f1_gbi = f1_score(y_test, pred_gbi, average='weighted')
results['Classifier'].append('Gradient Boosting')
results['Accuracy'].append(accuracy_gbi)
results['F1 Score'].append(f1_gbi)

# K-Nearest Neighbors (KNN) Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(x_train_tfidf, y_train_encoded)
pred_knn = knn_classifier.predict(x_test_tfidf)
accuracy_knn = accuracy_score(y_test_encoded, pred_knn)
f1_knn = f1_score(y_test_encoded, pred_knn, average='weighted')
results['Classifier'].append('KNN')
results['Accuracy'].append(accuracy_knn)
results['F1 Score'].append(f1_knn)

# Multinomial Naive Bayes Classifier
mnb_classifier = MultinomialNB()
mnb_classifier.fit(x_train_tfidf, y_train_encoded)
pred_mnb = mnb_classifier.predict(x_test_tfidf)
accuracy_mnb = accuracy_score(y_test_encoded, pred_mnb)
f1_mnb = f1_score(y_test_encoded, pred_mnb, average='weighted')
results['Classifier'].append('Multinomial Naive Bayes')
results['Accuracy'].append(accuracy_mnb)
results['F1 Score'].append(f1_mnb)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=101)
dt_classifier.fit(x_train_tfidf, y_train_encoded)
pred_dt = dt_classifier.predict(x_test_tfidf)
accuracy_dt = accuracy_score(y_test_encoded, pred_dt)
f1_dt = f1_score(y_test_encoded, pred_dt, average='weighted')
results['Classifier'].append('Decision Tree')
results['Accuracy'].append(accuracy_dt)
results['F1 Score'].append(f1_dt)

# XGBoost Classifier
xgb_classifier = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8, n_estimators=100, random_state=101)
xgb_classifier.fit(x_train_tfidf, y_train_encoded)
pred_xgb = xgb_classifier.predict(x_test_tfidf)
accuracy_xgb = accuracy_score(y_test_encoded, pred_xgb)
f1_xgb = f1_score(y_test_encoded, pred_xgb, average='weighted')
results['Classifier'].append('XGBoost')
results['Accuracy'].append(accuracy_xgb)
results['F1 Score'].append(f1_xgb)

# Multi-layer Perceptron (MLP) Classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=200, random_state=101)
mlp_classifier.fit(x_train_tfidf, y_train_encoded)
pred_mlp = mlp_classifier.predict(x_test_tfidf)
accuracy_mlp = accuracy_score(y_test_encoded, pred_mlp)
f1_mlp = f1_score(y_test_encoded, pred_mlp, average='weighted')
results['Classifier'].append('MLP')
results['Accuracy'].append(accuracy_mlp)
results['F1 Score'].append(f1_mlp)

# Convert results to DataFrame for easy plotting
import pandas as pd
df_results = pd.DataFrame(results)

# Plotting the results
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")
sns.barplot(x='Classifier', y='Accuracy', data=df_results, color='skyblue', label='Accuracy')
sns.barplot(x='Classifier', y='F1 Score', data=df_results, color='orange', label='F1 Score')
plt.title('Classifier Comparison')
plt.legend(loc='upper right')
plt.show()


# In[ ]:




