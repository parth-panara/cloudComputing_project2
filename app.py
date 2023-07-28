from flask import Flask, render_template
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import quinn
from pyspark.sql import functions as F
import pandas as pd

# Create a SparkSession
spark = SparkSession.builder.appName("Parth's_WineQualityPrediction").getOrCreate()

app = Flask(__name__)

def omit_quotations(s):
    return s.replace('"', '')

def load_data(file_path):
    # Create a SparkSession
    spark = SparkSession.builder.appName("Load_Data").getOrCreate()
    data = spark.read.format("csv").options(header='true', inferSchema='true', sep=';').load(file_path)
    data = data.toDF(*[omit_quotations(col) for col in data.columns])
    print("Data loaded from:", file_path)
    print(data.toPandas().head())
    return data


def preprocess_data(data):
    data = quinn.with_columns_renamed(omit_quotations)(data)
    data = data.withColumnRenamed('quality', 'label')
    data = data.withColumn("label", F.when(data["label"] > 5, 1).otherwise(0))
    print("Data has been formatted.")
    print(data.toPandas().head())
    return data

def calculate_f1_score(validation_data, best_model):
    # Make predictions using the best model
    predictions = best_model.transform(validation_data)

    # Evaluate the model using F1 score
    evaluator = MulticlassClassificationEvaluator(metricName="f1")
    f1_score = evaluator.evaluate(predictions)

    return f1_score

@app.route('/')
def display_output():
    # Load the trained model
    cvModel_best = "/home/hadoop/cloudComputing_project2/bestwinequalitymodel"
    best_model = PipelineModel.load(cvModel_best)

    # Load the validation data
    validation_data = load_data("/home/hadoop/cloudComputing_project2/ValidationDataset.csv")

    # Preprocess the validation data
    validation_data = preprocess_data(validation_data)

    # Calculate F1 score using the loaded model and validation data
    f1_score = calculate_f1_score(validation_data, best_model)

    # Make predictions using the best model
    predictions = best_model.transform(validation_data)
    
    # Save the predictions DataFrame to Parquet format
    predictions.write.parquet('/app/predictions.parquet', mode='overwrite')

    # Stop the Spark session
    spark.stop()

    # Render the index.html template with the F1 score
    return render_template('index.html', f1_score=f1_score)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)






