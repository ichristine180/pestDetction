from flask import Flask, render_template, request
from predections import predictPest
from categories import categories
import os
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/results', methods = ['GET','POST'])
def results():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('/home/chris/myapp/pestProject/static/upload/', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred = predictPest(img=file_path)

        return render_template('results.html', res=categories()[int(pred)], user_image='/upload/'+filename)

if __name__ == '__main__':
    app.run(debug=True)
