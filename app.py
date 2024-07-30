from tensorflow.keras.models import load_model
from flask import Flask, request, url_for, render_template, redirect, session, Response
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from flask_wtf.file import file_required, file_allowed
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense,Input
from tensorflow.keras.models import Model

import cv2
import numpy as np
import os

app = Flask(__name__)
app.config['SECRET_KEY']='jaykumar'

model = load_model('brain_tumor.h5')

app = Flask(__name__)
app.config['SECRET_KEY']='jaykumar'

class BrainForm(FlaskForm):
    style={'class': 'form-control', 'style':'width:25%;'}
    image = FileField("",validators=[file_required(),file_allowed(['jpg','png','jpeg'],'Images Only!')],render_kw=style)
    submit = SubmitField("Analyze",render_kw={'class':'btn btn-outline-primary'})


@app.route('/empty_page')
def empty_page():
    filename = session.get('filename', None)
    os.remove(os.path.join(UPLOAD_FOLDER, filename))
    return redirect(url_for('index'))


def predict(model,sample):
    img = cv2.imread(sample)
    img = cv2.resize(img,(150,150))
    img = np.reshape(img,(1,150,150,3))
    return np.argmax(model.predict(img))

def tumor_name(value):
    if value==0:
        return 'Glioma Tumor'
    elif value==1:
        return 'Meningioma Tumor'
    elif value==2:
        return 'No Tumor Found'
    elif value==3:
        return 'Pituitary Tumor'
x=0

@app.route('/result', methods=['POST', 'GET'])
def prediction():
    pred_val = predict(model,x)
    result = tumor_name(pred_val)
    os.remove(x)
    return render_template('prediction.html',result=result)


@app.route('/')
def index2():
    return render_template('index2.html')

@app.route('/pred', methods=['POST', 'GET'])
def index():
    form = BrainForm()

    if form.validate_on_submit():

        assets_dir = './static'
        img = form.image.data
        img_name = secure_filename(img.filename)

        img.save(os.path.join(assets_dir, img_name))
        global x
        x=os.path.join(assets_dir, img_name)

        return redirect(url_for('prediction'))

    return render_template('index.html',form=form)


# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            # Redirect to main page after successful login
            return redirect(url_for('main'))
    return render_template('login.html')



@app.route('/signup')
def signup():
    # Add logic to handle signup functionality
    return render_template('signup.html')

@app.route('/main', methods=['GET', 'POST'])
def main():
    return render_template('main.html')

@app.route('/home',methods=['POST', 'GET'])
def home():
    return render_template('home.html')

@app.route('/create_model', methods=['POST', 'GET'])
def model_architecture():
    effnet = EfficientNetB1(weights="imagenet", include_top=False, input_shape=(224,224, 3))
    model = effnet.output
    model = GlobalAveragePooling2D()(model)
    model = Dropout(0.5)(model)
    model = Dense(4, activation="softmax")(model)
    model = Model(inputs=effnet.input, outputs=model)
    
    # Generate model summary as a string
    model_summary_str = []
    model.summary(print_fn=lambda x: model_summary_str.append(x))

    return render_template('create_model.html', model_summary=model_summary_str)


@app.route('/evaluation_matrix',methods=['POST', 'GET'])
def evaluation_matrix():
    return render_template('evaluation_matrix.html')

if __name__=="__main__":
    app.run(port=3000,debug="true")
    

