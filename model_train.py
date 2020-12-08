
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,  roc_auc_score


from numpy import mean, std
import pickle


model_name = "model.pkl"
model_path = "model_config_file/"

def save_model(model):
    # Saving the model
    path = model_path + model_name
    pickle.dump(model, open(path, 'wb'))


def train_classification_model(dataset_scaled, final_data, target_col ):
    """
    Performing KFold Cross Validation and training the model
    """
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    # create model
    model = LogisticRegression(C=1.0, class_weight="balanced", dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=300, multi_class='ovr', n_jobs=100,
          penalty='l1', random_state=None, solver='saga', tol=0.0001,
          verbose=0, warm_start=False)

    # evaluate model

    X = dataset_scaled
    y = final_data[target_col]
    test_size = 0.25

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model.fit(X_train,y_train)
    predictions   = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print ("\n Classification report : \n",classification_report(y_test,predictions))

    accuracy = accuracy_score(y_test,predictions)
    print ("Accuracy   Score : ",accuracy)
    
    """
    #confusion matrix
    conf_matrix = confusion_matrix(y_test,predictions)
    #roc_auc_score
    model_roc_auc = roc_auc_score(y_test,predictions) 
    print ("Area under curve : ",model_roc_auc,"\n")
    """
    save_model(model)

    return accuracy
    """
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    accuracy = mean(scores)

    save_model(model)
    print("Model train and save is completed")

    return accuracy
    """