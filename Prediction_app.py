import quinn
import pandas as pd

from app import app
from pyspark.ml.feature import VectorAssembler, Normalizer, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline, PipelineModel
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import functions as F

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Parth's_WineQualityPrediction") \
    .getOrCreate()

def omit_quotations(s):
    return s.replace('"', '')

def load_data(file_path):
    data = spark.read.format("csv").options(header='true', inferSchema='true', sep=';').load(file_path)
    data = data.toDF(*[omit_quotations(col) for col in data.columns])
    print("The data loaded from:", file_path)
    print(data.toPandas().head())
    return data

def preprocess_data(data):
    data = quinn.with_columns_renamed(omit_quotations)(data)
    data = data.withColumnRenamed('quality', 'label') 
    # Threshold winequality to determine positive (1) and negative (0) biinary labels
    # because RandomForestClassifier and GBTClassifier models in Spark require label values to be binary (0 or 1) for binary classification tasks.
    data=data.withColumn("label", F.when(data["label"] > 5, 1).otherwise(0))
    print("The data has been formatted.")
    print(data.toPandas().head())
    return data
    
def create_pipeline(classifier):
    assembler = VectorAssembler(
        inputCols=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                   "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
        outputCol="features"
    )
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    pipeline = Pipeline(stages=[assembler, scaler, classifier])
    return pipeline

    
# Load the training and validation datasets
train_data = load_data("/home/hadoop/cloudComputing_project2/TrainingDataset.csv")
validation_data = load_data("/home/hadoop/cloudComputing_project2/ValidationDataset.csv")

# Load the best logistic regression model which has the best f1 score
cvModel_best = "/home/hadoop/cloudComputing_project2/bestwinequalitymodel"
best_model = PipelineModel.load(cvModel_best)

# Preprocess the validation data (perform the same preprocessing as in training app)
validation_data = preprocess_data(validation_data)

# Make predictions using the logistic regression model because this model has best f1 score
predictions = best_model.transform(validation_data)

# Evaluate the best model on the data and calculate the F1 score
evaluator = MulticlassClassificationEvaluator(metricName="f1")
f1_score = evaluator.evaluate(predictions)

print()
# Print the F1 score
print("F1 Score of the Prediction App model: {:.3f}".format(f1_score))
print()
predictions.show()

# Stop the Spark session
spark.stop()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
