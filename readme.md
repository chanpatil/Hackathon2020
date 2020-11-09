@app.route("/")                                  
			---> API to get the home page 

@app.route('/upload_files', methods=['POST'])    
			---> API to upload the files
					1. Dataset.csv/xlsx
					2. Feature_Selected.xlsx
					3. Feature_Info.json

@app.route('/data_profiling', methods = ['GET', 'POST'])
			---> API to perform the Data Profling
