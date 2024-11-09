# from flask import Flask,render_template,request
# from src.pipeline.predict_pipeline import PredictPipeline,CustomData

# application=Flask(__name__)
# app=application

# @app.route('/')
# def index():
#     return render_template("index.html")

# @app.route('/predict', methods=["GET",'POST'])
# def predict_datapoint():
#     if request.method=="GET":
#         return render_template('home.html')
    
#     # ,BHk,Size,Area Type,City,Furnishing Status,Tenant Preferred,Bathroom
#     else:
#         data=CustomData(
#             BHK=request.form.get('BHK'),
#             Size=request.form.get('Size'),
#             AreaType=request.form.get('AreaType'),
#             City=request.form.get('City'),
#             FurnishingStatus=request.form.get('FurnishingStatus'),
#             TenantPreferred=request.form.get('TenantPreferred'),
#             Bathroom=request.form.get('Bathroom'),
# )
#         pred_df=data.get_data_as_data_frame()
#         print(pred_df)
#         print("Before Prediction")

#         predict_pipeline=PredictPipeline()
#         print("Mid Prediction")
#         results=predict_pipeline.predict(pred_df)
#         print("after Prediction")
#         return render_template('home.html',results=results[0])
    

 

# if __name__=="__main__":
#    app.run(host="0.0.0.0",debug=True)   


from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Retrieve form data
            BHK = int(request.form['BHK'])
            Size = float(request.form['Size'])
            Area_Type = request.form['Area_Type']
            City = request.form['City']
            Furnishing_Status = request.form['Furnishing_Status']
            Tenant_Preferred = request.form['Tenant_Preferred']
            Bathroom = int(request.form['Bathroom'])
            
            # Create an object of CustomData
            data = CustomData(BHK, Size, Area_Type, City, Furnishing_Status, Tenant_Preferred, Bathroom)
 # Area_Type', 'Area_Locality', 'City',
    #    'Furnishing_Status', 'Tenant_Preferred'   
    # 
    #         
            # Convert to DataFrame
            input_data = data.get_data_as_data_frame()
            
            # Initialize PredictPipeline and get prediction
            pipeline = PredictPipeline()
            prediction = pipeline.predict(input_data)
            
            # Render the results page with the prediction
            return render_template('home.html', results=prediction[0])  # Show the prediction
            
        except Exception as e:
            # Handle any exceptions and display an error message
            return render_template('home.html', results=f"Error: {str(e)}")
    
    # If GET method, just render the form
    return render_template('home.html', results=None)

if __name__ == '__main__':
    app.run(debug=True)
