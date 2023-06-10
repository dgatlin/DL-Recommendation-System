from pyspark.sql import SparkSession
from ML_Algos.models import MatrixFactorization

from pipeline import MFSparkPipeline
from config import core

from processing import data_manager


# Create Spark session
spark = SparkSession.builder.appName("deeprec").master("local[2]").getOrCreate()

# ************************************************************************

# Load data
file_path = core.config.app_config.training_data_file
df = spark.read.option("inferSchema", "true").csv(file_path).coalesce(2)

# ************************************************************************

nums = data_manager.load_nums()

# Define the model
model = MatrixFactorization(num_users=nums[0], num_items=nums[1])

# ************************************************************************

# Create a pipeline with vector assembler and SparkTorch model
pipeline = MFSparkPipeline(model, df)

# Fit the pipeline on the DataFrame
pipeline_model = pipeline.fit(df)

# Save the model
pipeline_model.save(core.config.app_config.trained_models)

# Stop Spark session
spark.stop()
