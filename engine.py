from sparktorch import serialize_torch_obj, SparkTorch
import torch
import torch.nn as nn
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import Pipeline

spark = SparkSession.builder.appName("examples").master('local[2]').getOrCreate()
df = spark.read.option("inferSchema", "true").csv('mnist_train.csv').coalesce(2)

network = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
    nn.Softmax(dim=1)
)

# Build the pytorch object
torch_obj = serialize_torch_obj(
    model=network,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam,
    lr=0.0001
)

# Setup features
vector_assembler = VectorAssembler(inputCols=df.columns[1:785], outputCol='features')

# Create a SparkTorch Model with torch distributed. Barrier execution is on by default for this mode.
spark_model = SparkTorch(
    inputCol='features',
    labelCol='_c0',
    predictionCol='predictions',
    torchObj=torch_obj,
    iters=50,
    verbose=1
)

# Can be used in a pipeline and saved.
p = Pipeline(stages=[vector_assembler, spark_model]).fit(df)
p.save('simple_dnn')