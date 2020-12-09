
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

from numpy import mean, std
import pickle


model_name = "model.pkl"
model_path = "model_config_file/"

def save_model(model):
    # Saving the model
    path = model_path + model_name
    pickle.dump(model, open(path, 'wb'))

# MAPE value
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_regression_model(dataset_scaled, final_data, target_col ):
    
    # create model
    model = LinearRegression()

    # evaluate model
    X = dataset_scaled
    y = final_data[target_col]
    test_size = 0.25

    # train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    
    #Mape value
    mape = round(mean_absolute_percentage_error(y_test,predictions),2)

    print ("MAPE   Score : ",mape)
    
    """
    #confusion matrix
    conf_matrix = confusion_matrix(y_test,predictions)
    #roc_auc_score
    model_roc_auc = roc_auc_score(y_test,predictions) 
    print ("Area under curve : ",model_roc_auc,"\n")
    """
    save_model(model)

    return mape
    """
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    accuracy = mean(scores)

    save_model(model)
    print("Model train and save is completed")

    return accuracy
    """