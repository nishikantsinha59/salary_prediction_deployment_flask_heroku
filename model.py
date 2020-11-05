import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle

data = {"Index":[0,1,2,3,4,5,6,7],"experience":[np.NaN,np.NaN,'five','two','seven','three','ten','eleven'],
        "test_score":[8,8,6,10,9,7,np.NaN,7],"interview_score":[9,6,7,10,6,10,7,8],
        "salary":[50000,45000,60000,65000,70000,62000,72000,80000]}
df = pd.DataFrame(data)

df['experience'].fillna(0,inplace=True)
df['test_score'].fillna(df['test_score'].mean(), inplace=True)
df.set_index('Index', inplace=True)

# Separate independent features
X = df.iloc[:,:3]

# Coverting numbers in words to integers values
def covert_to_numeric(word):
	word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'tweleve':12,0:0}
	return word_dict[word]

X['experience'] = X['experience'].apply(lambda x:covert_to_numeric(x))

Y = df.iloc[:,-1]

# Splitting into training and test set
# Since we have a very small dataset, we will train the model with all available data

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# Fitting model with trainig data
regressor.fit(X,Y)

# Saving the model to the disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading the model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))

