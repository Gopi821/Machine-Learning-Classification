from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return 'welcom to flask today we try first app'

#app.run(host='0.0.0.0', port=81)

app.run(host ='127.0.0.1', port=5000)


