# Deep Learning Recommendation System with Spark Training 
A recommendation system is an artificial intelligence or AI algorithm, usually associated with 
machine learning, that uses data to suggest or recommend additional products to consumers. 
These can be based on various criteria, including past purchases, search history, demographic 
information, and other factors.


## Matrix Factorization
![My Image](./img/MF.png)

* Matrix factorization is a simple embedding model. Given the feedback matrix A = R^ , where  is the number of users (or queries) and  is the number of items, the model learns:
    * A user embedding matrix  of size  M where each row represents a user and each column represents a latent feature.
    * An item embedding matrix  of size N  where each row represents an item and each column represents a latent feature.
    * A user bias vector  of size  where each element represents a user.
    * An item bias vector  of size  where each element represents an item.




## Spark Distributed Model Training
![My Image](./img/spark.png)


Training a deep learning matrix factorization recommendation engine with Apache Spark offers numerous 
benefits. Firstly, Apache Spark provides a distributed computing framework, allowing the training 
process to leverage the power of a cluster of machines, enabling faster and scalable training. This 
scalability is crucial when dealing with large datasets and complex models. Secondly, Apache Spark's 
built-in support for data preprocessing and transformation tasks simplifies the data preparation process, 
ensuring efficient feature engineering for the recommendation engine. 

Additionally, Spark's integration with popular deep learning libraries like TensorFlow and PyTorch 
enables seamless integration of deep learning algorithms into the recommendation system. The combination 
of Spark's distributed computing capabilities and deep learning frameworks optimizes the training process,
leading to more accurate and effective recommendations.  Overall, training a deep learning matrix 
factorization recommendation engine with Apache Spark enhances performance, scalability, and productivity, 
enabling the development of high-quality recommendation systems at scale.

## Literature and Inspiration

* [An Introduction to Recommender Systems (+9 Easy Examples)](https://www.iteratorshq.com/blog/an-introduction-recommender-systems-9-easy-examples/)
* [An In-Depth Guide to How Recommender Systems Work](https://builtin.com/data-science/recommender-systems)
* [Matrix Factorization Techniques for Recommender Systems](https://www.asc.ohio-state.edu/statistics/dmsl//Koren_2009.pdf)
* [An Introduction to Distributed Deep Learning](http://seba1511.net/dist_blog/)
* [HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](https://arxiv.org/pdf/1106.5730.pdf)
* [Scaling Distributed Machine Learning with the Parameter Server](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf)
