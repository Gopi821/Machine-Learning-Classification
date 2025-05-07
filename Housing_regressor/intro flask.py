from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Welcome to flask</p>"
app.run(host='127.0.0.1',port=5000)