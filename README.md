# Deep Learning Recommendation System with Spark Training 



# Matrix Factorization

* Matrix factorization is a simple embedding model. Given the feedback matrix A , where  is the number of users (or queries) and  is the number of items, the model learns:
    * A user embedding matrix  of size  M where each row represents a user and each column represents a latent feature.
    * An item embedding matrix  of size N  where each row represents an item and each column represents a latent feature.
    * A user bias vector  of size  where each element represents a user.
    * An item bias vector  of size  where each element represents an item.

![My Image](./img/MF.png)

* `BiasMF`: [Matrix Factorization Techniques for Recommender Systems](https://www.asc.ohio-state.edu/statistics/dmsl//Koren_2009.pdf)


# Pytorch 


# Spark 


# Literature and Inspiration

* Distributed training: http://seba1511.net/dist_blog/
* HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent: https://arxiv.org/pdf/1106.5730.pdf
* Scaling Distributed Machine Learning with the Parameter Server: https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf