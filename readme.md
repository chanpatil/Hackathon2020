*************************************** AiZen *********************************************

@app.route("/")                                  
			---> API to get the home page 

@app.route('/upload_files', methods=['POST'])    
			---> API to upload the files
					1. Dataset.csv/xlsx
					2. Feature_Selected.xlsx
					3. Feature_Info.json

@app.route('/data_profiling', methods = ['GET', 'POST'])
			---> API to perform the Data Profiling

@app.route('/train_classifier', methods = ['GET', 'POST'])
			---> API to train the classifier model by utilising the file uploaded from /upload_files

@app.route('/train_regressor', methods = ['GET', 'POST'])
			---> API to train the regression model by utilising the file uploaded from /upload_files

@app.route('/Reboot_Engine', methods = ['GET', 'POST'])
			---> API to delete the file stored in above API's for DataProfiling/training/testing of dataset

