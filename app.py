#Importing necessary libraries and modules.
from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#Creating a flask app
app = Flask(__name__)
#Unpickling the pkl file containing our trained model.
model = pickle.load(open('logmodel.pkl', 'rb'))
#Creating a route for homepage
@app.route('/')
def home():
    #returns the homepage html
    return render_template('index.html')
#Creating another route for the prediction
@app.route('/predict',methods=['POST'])
def predict():
    # Get all features values from the form
    input_features = [float(x) for x in request.form.values()]
    # Forming array of the input features
    features_values = [np.array(input_features)]
    #Since we peformed feature selection on the dataset we will use the 23 feature names.
    #Creating list of feature names in the same order.
    features_names = ['mean radius', 'mean texture', 'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 
       'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 
       'fractal dimension error', 'worst texture', 'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']
    #Creating a dataframe with the input feature values and respective feature names.   
    df = pd.DataFrame(features_values, columns=features_names)
    sc=StandardScaler()
    df=sc.fit_transform(df)
    #Prediction of given input stored in output variable.
    output = model.predict(df)
    #Since Malignant is signified by 0 and Benign is signified by 1 in the dataset
    # We are checking the int output and storing the result as benign or malignant as a string to display on the page.    
    if output == 0:
        result = "malignant cancer"
    else:
        result = "benign cancer"
        
    #Returning the prediction page with the predicted text.
    return render_template('index.html', prediction='Patient has {}'.format(result))
if __name__ == "__main__":
    #Run the app
    app.run(debug=True)
