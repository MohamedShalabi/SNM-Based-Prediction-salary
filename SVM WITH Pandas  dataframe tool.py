#importing the libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import seaborn as sn


#reading the data 
df = pd.read_csv('income_data.txt' , delimiter = ', ' , header = None , engine = 'python')
df =df[df.index <= 24999]
def handling_categories(df):
    columns = df.columns.values
    
    for column in columns :
        text_vals = {}
        def convert_to_int(val):
            return text_vals[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_values = set(column_contents)
            x = 0
            for unique in unique_values :
                if  unique not in text_vals:
                    text_vals[unique] = x
                    x+=1
            df[column] = list(map(convert_to_int , df[column]))
    return df
df = handling_categories(df)

                    
X = df.iloc[: , :-1]
Y = df.iloc[: , -1]
        
#converting to numpy array
X = np.array(dataset).astype(str)
# Convert string data to numerical data
label_encoder = [] 
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
    if item.isdigit(): 
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)
ALL =pd.DataFrame(X_encoded)
''' make a correlation'''
pd.options.display.float_format = '{:.3f}'.format
make_correlation = ALL.corr()
plt.figure(figsize=(14,11))
sn.heatmap(make_correlation,annot = True)
plt.show()
# Create SVM classifier
classifier = OneVsOneClassifier(LinearSVC(random_state=3))

# Train the classifier
classifier.fit(X, y)

# Cross validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=5)
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
confusion_mat = confusion_matrix(y_test_pred, y_test)
# Compute the F1 score of the SVM classifier
f1 = cross_validation.cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")

# Predict output for a test datapoint
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

# Encode test datapoint
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))
        count += 1 

input_data_encoded = np.array(input_data_encoded)

# Run classifier on encoded datapoint and print output
predicted_class = classifier.predict([input_data_encoded])
real =label_encoder[-1].inverse_transform(predicted_class)[0]