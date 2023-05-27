import torch
from pyspark.ml.feature import VectorAssembler
from sparktorch import serialize_torch_obj, SparkTorch
from pyspark.ml.pipeline import Pipeline

from config import core


class MFSparkPipeline:
    """
    class encapsulates the necessary components and methods to
    construct and utilize a pipeline for training and predicting
    with SparkTorch and PyTorch models. It initializes a vector
    assembler, serializes the PyTorch object, creates a
    SparkTorch model, and defines a pipeline consisting of the
    vector assembler and the SparkTorch model.

    """

    def __init__(self, model, df):
        self.vector_assembler = VectorAssembler(
            inputCols=df.columns[4:6], outputCol=core.config.model_config.output_col
        )
        self.torch_obj = serialize_torch_obj(
            model=model,
            criterion=torch.nn.MSELoss(),
            optimizer=torch.optim.SGD,
            momentum=core.config.model_config.momentum,
            lr=core.config.model_config.lr,
        )
        self.spark_model = SparkTorch(
            inputCol=core.config.model_config.output_col,
            labelCol=core.config.model_config.label_col,
            predictionCol=core.config.model_config.prediction_col,
            torchObj=self.torch_obj,
            iters=core.config.model_config.iterations,
            verbose=core.config.model_config.verbose,
        )
        self.pipeline = Pipeline(stages=[self.vector_assembler, self.spark_model])

    def fit(self, df):
        return self.pipeline.fit(df)

    def get_pipeline(self):
        return self.pipeline
