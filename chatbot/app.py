# Create flask app.py
from flask import Flask, render_template, url_for, redirect, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from datetime import datetime
import os
import pickle
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding

app = Flask(__name__, static_folder='static')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///{}'.format(os.path.join(app.root_path, 'database.db'))
db = SQLAlchemy(app)
app.app_context().push()
bcrypt = Bcrypt(app)
app.config['SECRET_KEY'] = 'thisisasecretkey'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.String(255), nullable=False)
    response = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')
                                           
                                           ##COMPLETE TRAINING

# Load the JSON file
with open('chatbot\data\data.json', 'r') as f:
    data = json.load(f)
training_sentences = []
training_labels = []
labels = []
responses = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])   
num_classes = len(labels)
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)
vocab_size = 2000
embedding_dim = 16
max_len = 30
oov_token = "<OOV>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(LSTM(16, return_sequences=False))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

model.summary()
epochs = 300
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

# to save the trained model
model.save("chatbot\models\chat_model.h5")

import pickle

# to save the fitted tokenizer
with open('chatbot\\models\\tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# to save the fitted label encoder
with open('chatbot\\models\\label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

# Load the intents JSON file
with open('chatbot\data\data.json', 'r') as f:
    intents = json.load(f)
# Load the model and tokenizer, and instantiate the label encoder

                            #LOADING AND CHATBOT
model = load_model('chatbot\models\chat_model.h5')

# Load the fitted tokenizer
with open('chatbot\\models\\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the fitted label encoder
with open('chatbot\\models\\label_encoder.pickle', 'rb') as ecn_file:
    lbl_encoder = pickle.load(ecn_file)

max_len=30

def chatbot_response(message):
    try:
        # Get user input message
        msg = request.args.get('msg')
        
        # Convert input message into sequence of integers using tokenizer
        msg_seq = tokenizer.texts_to_sequences([msg])
        # Pad the sequence to have fixed length
        msg_seq = pad_sequences(msg_seq, maxlen=max_len, truncating='post')
        
        # Predict the class of the input message
        pred = model.predict(msg_seq)[0]
        # Get the index of the predicted class
        pred_idx = np.argmax(pred)
        # Convert the index back to the original label
        tag = lbl_encoder.inverse_transform([pred_idx])[0]
        
        # Check the confidence level of the prediction
        confidence = pred[pred_idx]
        if confidence < 0.5:
            response = "I'm sorry, but I didn't understand your concern. Please keep it simple, rephrase the sentence or try asking something else related to mental health"
        else:
            # Find the appropriate response for the predicted label
            for intent in intents['intents']:
                if intent['tag'] == tag:
                    response = np.random.choice(intent['responses'])
                    break
        
        return response
    except Exception as e:
        print("Error in chatbot_response:", e)
        return "I'm sorry, I couldn't understand that."


@app.route("/chatbot")
@login_required
def chatbot():
    # Retrieve chats associated with the current user
    chats = Chat.query.filter_by(user_id=current_user.id).all()

    # Create a list of dictionaries to hold chat messages and responses
    chat_history = []
    for chat in chats:
        chat_history.append({'message': chat.message, 'response': chat.response})

    return render_template("index.html", chat_history=chat_history)



@app.route("/get")
@login_required
def get_bot_response():
    try:
        msg = request.args.get('msg')
        response = chatbot_response(msg)
        chat = Chat(user_id=current_user.id, message=msg, response=response)
        db.session.add(chat)
        db.session.commit()
        return response
    except Exception as e:
        print("Error in get_bot_response:", e)
        return "An error occurred, please try again."

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('chatbot'))
    return render_template('login.html', form=form)


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    return response
if __name__ == "__main__":
    app.run()
