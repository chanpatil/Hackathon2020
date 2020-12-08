# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 23:46:53 2020

@author: Chanabasagoudap
"""
import os
import pandas as pd
import json
import numpy as np
from flask import jsonify , send_from_directory, send_file
from flask import Flask, render_template, request,url_for, flash, redirect, jsonify
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap
#import matplotlib.pyplot as plt
import base64
import time
from pandas_profiling import ProfileReport

import data_preprocessing as dp
import model_train as mt
import inference as inf

# creates a Flask application, named app
app = Flask(__name__)
Bootstrap(app)
bootstrapLink= '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css">'

# a route where we will display a welcome message via an HTML template

import os
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads/train')
TEST_FOLDER = os.path.join(path, 'uploads/test')
DFProfile_FOLDER = os.path.join(path, 'uploads/DFProfile')

# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
if not os.path.isdir(DFProfile_FOLDER):
    os.mkdir(DFProfile_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["TEST_FOLDER"] = TEST_FOLDER
app.config['DFProfile_FOLDER'] = DFProfile_FOLDER


# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(["csv","xlsx","json"])

###############################################################################
# Check the file is having proper extension or not
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



###############################################################################

@app.route("/")
def hello():
    return render_template('index.html')


@app.route('/upload_files', methods=['POST'])
def upload_files():
    if request.method == 'POST':
        mypath = "uploads/train"
        for root, dirs, files in os.walk(mypath):
            for file in files:
                os.remove(os.path.join(root, file))
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('files[]')
        #print(len(files))
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                print(filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return render_template("index.html")

@app.route('/data_profiling', methods = ['GET', 'POST'])
def data_profiling():
    if request.method == 'POST':

        if os.path.exists("uploads/DFProfile/DFReport.html"):
            os.remove("uploads/DFProfile/DFReport.html")
        
        mypath = "uploads/DFProfile"
        for root, dirs, files in os.walk(mypath):
            for file in files:
                os.remove(os.path.join(root, file))
        f = request.files['file']
        f.save(os.path.join(app.config['DFProfile_FOLDER'], f.filename))
        
        filename = None
        for root, dirs, files in os.walk(mypath):
            for file in files:
                filename = file
        
        # #return render_template("results/DFReport.html")
        print("File Name",filename)
        
        path = mypath + "/" +filename
        print("Path:",path)
        if ".csv" in filename:
            datasetprofile = pd.read_csv(path)
            profile = datasetprofile.profile_report(title='AiZen Data Profiling Report')
            profile.to_file(output_file="uploads/DFProfile/DFReport.html")
        elif ".xlsx" in filename:
            print("Excel FIle")
            datasetprofile = pd.read_excel(path, sheet_name=0)
            profile = datasetprofile.profile_report(title='AiZen Data Profiling Report')
            profile.to_file(output_file="uploads/DFProfile/DFReport.html")
            
        #return render_template("DFReport.html")
        #return send_file("uploads/DFProfile/DFReport.html", attachment_filename="DFReport.html")
        return send_from_directory(mypath, "DFReport.html", as_attachment=True)

def file_readers(training_path):
    path, dirs, files = next(os.walk(training_path))
    if len(files) == 3:
        for filename in files:
            if ".csv" in filename:
                dataset = pd.read_csv(training_path +"/"+ filename)
            elif ".xlsx" in filename and filename=="Feature_Selected.xlsx":
                feature_selected = pd.read_excel(training_path +"/"+ filename , sheet_name=0)
            elif ".xlsx" in filename:
                dataset = pd.read_excel(training_path +"/"+ filename , sheet_name=0)
            elif ".json" in filename:
                with open(training_path + "/"+ filename) as f:
                    fearure_info = json.load(f)

        return dataset, feature_selected, fearure_info
    else:
        return "Please upload all 3 files."


@app.route('/train_classifier', methods = ['GET', 'POST'])
def train_classifier():
    """
    This API enable the user to train the classifier model by utilising the file present
    in the folder "uploads/train".
    dataset_dup = dp_dup.remove_duplicate_records(dataset)
    dataset_nan = dp_nan.deal_with_nan(dataset_dup)
    dataset_cat = dp_cat.deal_with_categorical_data(dataset_nan)
    dataset_out = dp_out.deal_with_outlier(dataset_cat)
    dataset_scale = dp_scale.scale_feature(dataset_out)

    """
    if request.method == 'POST':
        training_path = "uploads/train"
        dataset, feature_selected, feature_info = file_readers(training_path)
        """
        print("Dataset",dataset.head(3))
        print("feature_selected",feature_selected)
        print("fearure_info",fearure_info)
        """
       
        # Getting Important columns from Feature Selected.xlsx
        selectedFeature = list(feature_selected[feature_selected["Required"] == True]["Feature Name"].values)
        print("Feature Selected:", selectedFeature)

        # Getting the feature information from Feature Info.json
        ignore_cols = feature_info["ignore_cols"]
        target_col = feature_info["target_col"]
        print("\nIgnore Cols:", ignore_cols)
        print("\nTarget Cols:", target_col)

        # Will remove the duplicates in dataset
        dataset_dup = dp.remove_duplicate_records(dataset)

         # Dataset as per the client data
        print("Before :Shape of Dataframe :", dataset.shape)

        # Taking the selected features only after removal of duplicates.
        target_var = dataset_dup[target_col]
        dataset_dup = dataset_dup[selectedFeature].copy()

        print("Length of target variable:",len(target_var))
        print("After :Shape of Dataframe after feature selection:", dataset_dup.shape)


        # Will deal with NaN value in dataset
        dataset_nan = dp.deal_with_nan(dataset_dup)
        #print(dataset_nan.shape)

        # Need to scale the data
        dataset_scaled = dp.feature_transformation(dataset_nan)
        print("Scaled Dataset:\n",dataset_scaled)
        
        
        # Now from here trainig starts
        final_data = dataset_scaled.copy()
        final_data[target_col] = target_var

        # Perform K-Fold Cross Validation for training.

        print("*********************************************")
        print(final_data.columns)
        df_result = mt.train_classification_model(dataset_scaled, final_data, target_col)

        result = "Training is completed and accuracy is = " + str(df_result)
        
        return result
  
    else:
        return "Only POST Method is allowed."

@app.route('/train_regressor', methods = ['GET', 'POST'])
def train_regressor():
    if request.method == 'POST':
        """
        This API enable the user to train the Regressor model by utilising the file present
        in the folder "uploads/train".
        """
        training_path = "uploads/train"
        dataset, feature_selected, fearure_info = file_readers(training_path)
        """
        print("Dataset",dataset.head(3))
        print("feature_selected",feature_selected)
        print("fearure_info",fearure_info)

        """
        return "Work in Progress"
    else:
        return "Only POST Method is allowed."
      

@app.route('/evaluate_model', methods = ['GET', 'POST'])
def evaluate_model():
    if request.method == 'POST':
        # Need to worki on the Creation of model
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['TEST_FOLDER'], filename))
        
        path, dirs, files = next(os.walk("uploads/test"))
        print(len(files))
        if len(files) == 1:
            for filename in files:
                path = "uploads/test" + "/" + filename
                if "csv" in filename:
                    test_data = pd.read_csv(path)
                elif "xlsx" in filename:
                    test_data = pd.read_excel(path)
        
        #print(test_data)
        
        training_path = "uploads/train"
        dataset, feature_selected, feature_info = file_readers(training_path)
       
        # Getting Important columns from Feature Selected.xlsx
        selectedFeature = list(feature_selected[feature_selected["Required"] == True]["Feature Name"].values)
        test_data = test_data[selectedFeature]

        # Will deal with NaN value in dataset
        dataset_nan = dp.deal_with_nan(test_data)
        #print(dataset_nan.shape)

        # Need to scale the data
        dataset_scaled = dp.feature_transformation(dataset_nan)
        print("Scaled Dataset:\n",dataset_scaled)

        # Performing the forecasting based on the saved model
        result = inf.perform_classification(dataset_scaled)
        
        path = "results/"
        filename = "AiZen_Predicted_value.xlsx"
        filepath = path + filename
        result.to_excel(filepath)

        return send_from_directory(path, filename, as_attachment=True)


@app.route('/refresh', methods = ['GET', 'POST'])
def refresh():
    if request.method == 'POST':
        # Need to worki on the Creation of model
        
        return "Work in Progress"

# run the application
if __name__ == "__main__":
    app.run()
    

"""
1>
            div1 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><img src='data:image/jpeg;base64,"+encoded_string1+"'/></div>"    
            div2 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><img src='data:image/jpeg;base64,"+encoded_string2+"'/></div>"    
            div3 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><img src='data:image/jpeg;base64,"+encoded_string3+"'/></div>"    
            div4 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><img src='data:image/jpeg;base64,"+encoded_string4+"'/></div>"    
            div5 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><img src='data:image/jpeg;base64,"+encoded_string5+"'/></div>"    
            div6 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><img src='data:image/jpeg;base64,"+encoded_string6+"'/></div>"
           
            div = div2+div3+div4+div5+div6+div1

            
     
    return bootstrapLink +"<div class='container container-fluid'><div class='panel panel-default'><div class='panel-heading text-center'><h3 class='panel-title'><strong>Sentiment Analysis</strong></h3></div><br>" + div

2>
            div1 = "<table border='1'><tr><th>Most famous Aspect Employee Talk About</th><tr><td>"+str(aspect_count)+"</td></tr></table>"
            div2 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><p>"+title4+"<br><img src='data:image/jpeg;base64,"+encoded_string4+"'/></div>"
            return ""+div1+"<br>"+div2+""


3>
# Visulization 1 ---- Feedback from each office Location
            #tot_feedback_count = len(dataset.Branch)
            viz_cnt,hitech_cnt, bachu_cnt, knodapur_cnt, kolhaput_cnt= 0, 0, 0, 0, 0
            for location in dataset.Branch:
                if location == "Visakhapatnam":
                    viz_cnt += 1
                if location == "Hitech City":
                    hitech_cnt += 1
                if location == "Bachupally":
                    bachu_cnt += 1
                if location == "Kondapur":
                    knodapur_cnt += 1
                if location == "Kolhapur":
                    kolhaput_cnt += 1
            
            location = ["Visakhapatnam","Hitech","Bachupally","Kondapur","Kolhapur"]
            count_index = np.arange(len(location)) 
            location_cnt = [viz_cnt,hitech_cnt, bachu_cnt,knodapur_cnt, kolhaput_cnt]
            XLabel1 = "MOURIT Tech Office Location"
            YLabel1 = "Feedback from different Office Location"
            title1 = "Feedback from Different Office Loaction of MOURI Tech"
            plot_img_addr1 = plot_bar_chart(count_index,location, location_cnt,XLabel1,YLabel1,title1)

            # Calling function to extact the base64 fror title1
            encoded_string1 = extract_base64(plot_img_addr1)
                     
            div1 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><p>"+title1+"<br><img src='data:image/jpeg;base64,"+encoded_string1+"'/></div>"
            
           # Visualization 2 :------  Internet Performacnce
           
            internet_sat , internet_unsat , internet_NA = 0, 0 ,0
            for emotion in dataset["internet performance"]:
                if emotion == "Yes":
                    internet_sat += 1
                if emotion == "No":
                    internet_unsat += 1
                if emotion == "N/A":
                    internet_NA += 1
            
            emotion = ["Satisfied", "Unsatisfied"," Not Applicable"]
            emotion_cnt = np.arange(len(emotion))
            internet_cnt = [internet_sat,internet_unsat, internet_NA]
            XLabel2 = " Happiness index"
            YLabel2 = " Internet Performance of different Office Location"
            title2 = "Overall MOURI Tech internet Performance"
            plot_img_addr2 = plot_bar_chart(emotion_cnt,emotion, internet_cnt,XLabel2,YLabel2,title2)
            
            # Calling function to extact the base64 fror title2
            encoded_string2 = extract_base64(plot_img_addr2)
            
            div2 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><p>"+title2+"<br><img src='data:image/jpeg;base64,"+encoded_string2+"'/></div>"


            # Visualization 3 :----------------------- Skype Zoom and WebEX connectivity
            
            szw_sat ,szw_unsat ,skw_NA = 0, 0, 0
            
            for connectivity in dataset["Skype/Zoom/Webex_Call_stabality"]:
                if connectivity =="Yes":
                    szw_sat += 1
                if connectivity == "No":
                    szw_unsat +=1
                if connectivity =="N/A":
                    skw_NA += 1
            
            szw_cnt = [szw_sat, szw_unsat,skw_NA ]
            XLabel3 = " Happiness index"
            YLabel3 = " Skype Zoom and WebEX connectivity different Office Location"
            title3 = "Overall MOURI Tech Skype Zoom and WebEX connectivity  Performance"
            
            #Plotting and saving a graph
            plot_img_addr3 = plot_bar_chart(emotion_cnt,emotion, szw_cnt,XLabel3,YLabel3,title3)
            # Calling function to extact the base64 fror title3
            encoded_string3 = extract_base64(plot_img_addr3)
            div3 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><p>"+title3+"<br><img src='data:image/jpeg;base64,"+encoded_string3+"'/></div>"
            
            return ""+div1+"<br>"+div2+"<br>"+div3+""
         
4>
numerical_data = dataset._get_numeric_data().columns
            categorical_data = [c for i, c in enumerate(dataset.columns) if dataset.dtypes[i] in [np.object]]
            Nan_Values = dataset.isnull().sum()

            primary_keys = [] 
            
            # Dynamically generating the possible Primary Keys
            for feature in dataset.columns:
                if len(dataset[feature]) == dataset[feature].nunique():
                    primary_keys.append(feature)
            #print("Primary Key",primary_keys)
            
            # Dynamically generating the count of primary keys
            pk_count_list = []
            for feature in primary_keys:
                pk_count_list.append(len(dataset[feature].unique()))
            
            # Column Analysis
            column_analysis = {}
            for features in dataset.columns:
                column_analysis[features] = dataset[features].describe()
            #print("Column Analysis",column_analysis)
            
            result = bootstrapLink+"<div class='container container-fluid'><div class='panel panel-default'><div class='panel-heading text-center'><h3 class='panel-title'> Types of Descriptive Analysis</h3></div><div class='panel-body'><div class='row'><div class='col-md-12'><table class='table table-bordered table-hover'><tr><th>Column with Numeric Values</th><th>Categorical Data</th><th>NaN values Columns</th><th>Primary Keys</th><th>Primary Keys Count List</th><th>Columns Analysis</th></tr><tr><td>"+str(numerical_data)+"</td><td>"+str(categorical_data)+"</td><td>"+str(Nan_Values)+"</td><td>"+str(primary_keys)+"</td><td>"+str(pk_count_list)+"</td><td>"+str(column_analysis)+"</td></tr></table></div></div></div></div></div>"
            
            
6> 
result = { 
                    "Shape of Dataset" : list(dataset.shape),
                    "Columns in Dataset" : list(dataset.columns),
                    "Count of Columns" : str(len(dataset.columns)),
                    "Need DataPreprocessing w.r.t NaN" : str(dataset.isnull().values.any()),
                    "Total Null Value count" : str(dataset.isnull().sum().sum()),
                    "Column with Null Values" : list(dataset.columns[dataset.isnull().any()])}
            #print("Result is ",result)
            #return json.dumps(result)
            return bootstrapLink+"<div class='container container-fluid'><div class='panel panel-default'><div class='panel-heading text-center'><h3 class='panel-title'>Dataset Analysis</h3></div><div class='panel-body'><div class='row'><div class='col-md-12'><table class='table table-bordered table-hover'><tr><th>Shape of Dataset(rows/columns)</th><th>Columns in Dataset</th><th>Count of Columns</th><th>Need DataPreprocessing w.r.t NaN</th><th>Total Null Value count</th><th>Column with Null Values</th></tr><tr><td>"+str(dataset.shape)+"</td><td>"+str(dataset.columns)+"</td><td>"+str(len(dataset.columns))+"</td><td>"+str(dataset.isnull().values.any())+"</td><td>"+str(dataset.isnull().sum().sum())+"</td><td>"+str(dataset.columns[dataset.isnull().any()])+"</td></tr></table></div></div></div></div></div>"                                     
            

"""
"""
        intermediate = """""""
        <div class="container container-fluid">
   <div class="panel panel-default">
      <div class="panel-heading text-center">
         <h3 class="panel-title"><strong>Exploratory Data Analysis</strong></h3>
      </div>
      <div class="panel-body">
         <div class="row">
            <div class="col-md-12">
        
        <div class="col-md-8">
                  <form action = "/home" method = "POST" enctype="multipart/form-data">
                     <p allign="Center"> File Upload Successfull !!! </p>
                     <br>
                     <input type = "submit" style="width: 60%;" value="Upload 3 Files" class="btn-xclg btn btn-primary"/>
                  </form>
               </div>

            </div>
         </div>
      </div>
   </div>
</div>
               """
               