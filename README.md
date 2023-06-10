# Deep Learning Recommendation System with Spark Training
A recommendation system is an artificial intelligence or AI algorithm, usually associated with
machine learning, that uses data to suggest or recommend additional products to consumers.
These can be based on various criteria, including past purchases, search history, demographic
information, and other factors.

One such example is Matrix factorization is a technique used in collaborative filtering-based recommendation systems.
It aims to decompose a user-item interaction matrix into two lower-rank matrices, representing
user and item embeddings. The underlying assumption is that the latent factors in these embeddings
capture the inherent characteristics and preferences of users and items.


## Matrix Factorization
![My Image](img/MF.png)

 Matrix factorization can be integrated into a recommendation engine by using neural networks to learn the embeddings.
 The user and item indices are passed through embedding layers, which map them to low-dimensional latent spaces.
 These embeddings are then combined and processed through additional layers to predict the user's preference or
 rating for a particular item.


## Spark Distributed Model Training
![My Image](img/spark.png)


Training a deep learning matrix factorization recommendation engine with Apache Spark offers numerous
benefits. Firstly, Apache Spark provides a distributed computing framework, allowing the training
process to leverage the power of a cluster of machines, enabling faster and scalable training. This
scalability is crucial when dealing with large datasets and complex models. Secondly, Apache Spark's
built-in support for data preprocessing and transformation tasks simplifies the data preparation process,
ensuring efficient feature engineering for the recommendation engine.

## Example

```python
from pyspark.sql import SparkSession
from ML_Algos.models import MatrixFactorization
from config import core
from pipeline import MFSparkPipeline
from processing import data_manager

# Create Spark session
spark = SparkSession.builder.appName("deeprec").master("local[2]").getOrCreate()

# Load data
file_path = core.config.app_config.training_data_file
df = spark.read.option("inferSchema", "true").csv(file_path).coalesce(2)

nums = data_manager.load_nums()

# Define the model
model = MatrixFactorization(num_users=nums[0], num_items=nums[1])

# Create a pipeline with vector assembler and SparkTorch model
pipeline = MFSparkPipeline(model, df)

# Fit the pipeline on the DataFrame
pipeline_model = pipeline.fit(df)

# Save the model
pipeline_model.save(core.config.app_config.trained_models)

# Stop Spark session
spark.stop()
```

## Special Note
Spark is able to deal with much bigger work loads than most options. If your data is larger than 1TB,
Spark is probably the way to go. However, [Dask](https://www.dask.org/) might also not be the best
suited tool for the project. There are other Pythonic solutions for Big Data, such as [Ray](https://www.ray.io/) and
[Modin](https://modin.readthedocs.io/en/stable/), [Vaex](https://vaex.io/) and [Rapids](https://rapids.ai/); all have their
pros and cons. But with more than 1TB of data, Spark is probably the best option.

## Literature and Inspiration
* [An Introduction to Recommender Systems (+9 Easy Examples)](https://www.iteratorshq.com/blog/an-introduction-recommender-systems-9-easy-examples/)
* [An In-Depth Guide to How Recommender Systems Work](https://builtin.com/data-science/recommender-systems)
* [Matrix Factorization for Recommender Systems](https://www.diva-portal.org/smash/get/diva2:633561/FULLTEXT01.pdf)
* [Matrix Factorization Techniques for Recommender Systems](https://www.asc.ohio-state.edu/statistics/dmsl//Koren_2009.pdf)
* [An Introduction to Distributed Deep Learning](http://seba1511.net/dist_blog/)
* [HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](https://arxiv.org/pdf/1106.5730.pdf)
* [Scaling Distributed Machine Learning with the Parameter Server](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf)
