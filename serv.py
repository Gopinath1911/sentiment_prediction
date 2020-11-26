from flask import Flask,request
import predict as pr
app=Flask(__name__)

@app.route("/")
def hlo():
    return "hello"
@app.route("/pred")
def user_prediction():
    input_text=request.args.get("message")
    return pr.sentiment_prediction(input_text)

if __name__=="__main__":
    app.run(debug=True)