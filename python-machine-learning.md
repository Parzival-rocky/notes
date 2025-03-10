---
date: [15:40, 2025-03-07]
---


## High level steps in machine learning 

1. Import Data 
    - usually in csv format 
2. Clean Data
3. Split the Data into: 
    - training 
    - test
4. Create a model
5. Train the model
6. Make Predictions
7. Evaluate and Improve

## Libraries and Tools

Numpy:
- Provides multidimensional array

Pandas:
- A data analysis library
- Provides us with a data frame, which is: 
    - excel like data structure

MatPlotLib: 
- 2D plotting library for create graphs and plots

Scikit-Learn: 
- A very popular machine learning library providing pre-configured models

Jupyter notebook: 
- Easier to test and visualize the data

## Examples

### Pandas demo:
```python 
import pandas as pd

# returns a dataset ofbject which is similar to an excel spreadsheet
df = pd.read_csv("fileName.csv")

# display dataframe
df

# display row x col
df.shape

# display col information
# this usually helps to get an overall understanding of our dataset
df.describe()

# display data in 2D array
df.values
```

### Desgning a simple model:

```python 
import pandas as pd
# for importing a models we use sklearn
# check their library for models and how to use them
from sklearn.tree import DecisionTreeClassifier

# for testing the model 
from sklearn.model_selection import train_test_split

# for testing accuracy by comparing test data with prediction
from sklearn.metrics import accuracy_score

"""
Let's say you're given a dataset like the below:
- Age Gender Genre
   20    1    Jazz
   60    0    HipHop
   ...

And asked to predict the genre based on the age and gender
"""

data = pd.read_csv("data.csv")

# split the dataset into two: input and output
# it doesn't modify the original dataset, instead creates a new one
# named X because it's convention
# X contains Age, Gender (Input set)
X = data.drop(columns=["Genre"])

# named y because it's convention (Output data set)
y = data["Genre"]

# splitting the dataset into 80% data and 20% test data
# return a tuple
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create a new instance of the class
model = DecisionTreeClassifier()

# train the model
model.fit(X_train, y_train)

# ask the model to make prediction
predictions = model.predict(X_test)

# testing accuracy
score = accuracy_score(y_test, predictions)
```

### Model persistance

To store the model first:
```python 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# for saving and loading models
from sklearn.externals import joblib

data = pd.read_csv("data.csv")
X = data.drop(columns=["Genre"])
y = data["Genre"]

model = DecisionTreeClassifier()
model.fit(X, y)

# to sore the model
joblib.dump(model, "filename.joblib")
```

After the model is stored, here is how we use it:
```python 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

# to load the model
model = joblib.load("filename.joblib")

# we can now use our model to predict
predictions = model.predict([[21, 1]])
```




#computer-science

### Links
[[]]

