import os
from flask import Flask, request, render_template
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import Row

spark = (
    SparkSession.builder
    .appName("SentimentWeb")
    .master("local[*]")
    .getOrCreate()
)

MODEL_PATH = "models/lr_sentiment_pipeline"
model = PipelineModel.load(MODEL_PATH)

app = Flask(__name__)
PORT = int(os.getenv("PORT", 5000))

def predict_sentiment(text):
    df = spark.createDataFrame([Row(text=text)])
    pred = (
        model.transform(df)
             .select("probability")       
             .collect()[0][0][1]         
    )
    return float(pred)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_text = request.form["user_text"]
        score     = predict_sentiment(user_text)
        label     = "Positive 😀" if score >= 0.5 else "Negative 🙁"
        return render_template("index.html",
                               user_text=user_text,
                               score=f"{score:.3f}",
                               label=label)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)