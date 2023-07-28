import numpy as np
import pandas as pd
import quinn
import random
import sys 

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import col, desc
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler, Normalizer, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.util import DefaultParamsReader, DefaultParamsWriter
from pyspark.mllib.evaluation import MulticlassMetrics


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
    
def create_pipeline(classifier, num_classes):
    assembler = VectorAssembler(
        inputCols=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                   "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
        outputCol="features"
    )
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    pipeline = Pipeline(stages=[assembler, scaler, classifier])
    return pipeline

def evaluate_model(model, data, evaluator):
    predictions = model.transform(data)
    score = evaluator.evaluate(predictions)
    return score

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Parth's_WineQualityPrediction") \
    .getOrCreate()

# Load the training and validation datasets
train_data = load_data("/home/hadoop/cloudComputing_project2/TrainingDataset.csv")
validation_data = load_data("/home/hadoop/cloudComputing_project2/ValidationDataset.csv")

# Perform data preprocessing
train_data = preprocess_data(train_data)
validation_data = preprocess_data(validation_data)

# Determine the number of classes
num_classes = train_data.select("label").distinct().count()

# Define the classifiers
lr_model = LogisticRegression()
rf_model = RandomForestClassifier()
gbt_model = GBTClassifier()

# Create the pipelines
pipeline_lr = create_pipeline(lr_model, num_classes=num_classes)
pipeline_rf = create_pipeline(rf_model, num_classes=num_classes)
pipeline_gbt = create_pipeline(gbt_model, num_classes=num_classes)

# Set up the parameter grid for hyperparameter tuning
param_grid = ParamGridBuilder().build()

# Define the evaluator
evaluator = MulticlassClassificationEvaluator(metricName="f1")
accuracy_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

# Perform cross-validation and train the models
cv_lr = CrossValidator(estimator=pipeline_lr, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
cv_rf = CrossValidator(estimator=pipeline_rf, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
cv_gbt = CrossValidator(estimator=pipeline_gbt, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)

# Fit the models on the training data
cvModel_lr = cv_lr.fit(train_data)
cvModel_rf = cv_rf.fit(train_data)
cvModel_gbt = cv_gbt.fit(train_data)

# Make predictions on the validation data
predictions_lr = cvModel_lr.transform(validation_data)
predictions_rf = cvModel_rf.transform(validation_data)
predictions_gbt = cvModel_gbt.transform(validation_data)

# Evaluate the models
f1_score_lr = evaluate_model(cvModel_lr, validation_data, evaluator)
f1_score_rf = evaluate_model(cvModel_rf, validation_data, evaluator)
f1_score_gbt = evaluate_model(cvModel_gbt, validation_data, evaluator)

print()
print("F1 Score for Logistic Regression Model: {:.3f}".format(f1_score_lr))
print("F1 Score for Random Forest Classifier Model: {:.3f}".format(f1_score_rf))
print("F1 Score for GBT Classifier Model: {:.3f}".format(f1_score_gbt))

# Define a dictionary to map the model objects to their names
model_names = {
    cvModel_lr: "Logistic Regression Model",
    cvModel_rf: "Random Forest Classifier Model",
    cvModel_gbt: "GBT Classifier Model"
}

best_model = None
best_f1_score = 0.0

# Find the best performing model
if f1_score_lr > best_f1_score:
    best_model = cvModel_lr
    best_f1_score = f1_score_lr
    best_model_name = model_names[cvModel_lr]

if f1_score_rf > best_f1_score:
    best_model = cvModel_rf
    best_f1_score = f1_score_rf
    best_model_name = model_names[cvModel_rf]

if f1_score_gbt > best_f1_score:
    best_model = cvModel_gbt
    best_f1_score = f1_score_gbt
    best_model_name = model_names[cvModel_gbt]

print()
print("Best F1 Score: {:.3f}".format(best_f1_score))

print()
print("Best Model: ", best_model_name)

accuracy = accuracy_evaluator.evaluate(best_model.transform(validation_data))

print("Accuracy of the Best Model: {:.3f}".format(accuracy))
print()
print("Best model ",best_model_name, " is selected for prediction application")
print()

# Save the model metadata
best_model_path = "/home/hadoop/cloudComputing_project2/bestwinequalitymodel"
cvModel_best = cvModel_lr.bestModel
cvModel_best.write().overwrite().save(best_model_path)

print("Best model saved to:", best_model_path)

# Stop the Spark session
spark.stop()
   