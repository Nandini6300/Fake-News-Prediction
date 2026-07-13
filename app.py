from flask import Flask, render_template, request
from predict import predict_news

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    if news.strip() == "":
        return render_template(
            "index.html",
            prediction="Please enter some news text."
        )

    result = predict_news(news)

    return render_template(
        "index.html",
        prediction=result,
        news=news
    )


if __name__ == "__main__":
    app.run(debug=True)