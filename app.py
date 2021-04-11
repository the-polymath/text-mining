from flask import Flask
from email_class import *

app = Flask(__name__)
@app.route('/')
def predict():
    return "Demo page"

@app.route('/<string:message>')
def func2(message):
    type = message_classifier(message)

    string = "The message is : " + type
    return string

app.run(port=4000, debug=True)

# string = "http://127.0.0.1:4000/" + message
