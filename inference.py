
import pickle

model_path = "model_config_file/model.pkl"
def perform_classification(data):

    #pickle.load(open(model_name, 'rb'))
    trained_model = pickle.load(open(model_path,'rb'))
    print("Trained Model:\n", trained_model)

    result = trained_model.predict(data)

    data["Predicted Value"] = result
    print(data)
    return data