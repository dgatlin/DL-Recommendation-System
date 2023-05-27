import torch


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


# Matrix factorization model
class MatrixFactorization(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=10):
        """
        This function takes three parameters: num_users, num_items, and embedding_dim.
        It initializes the embedding layers and biases using the EmbeddingLayer class.
        These initialization steps create the necessary embedding layers and biases for
        the matrix factorization model.

        :param num_users:
        :param num_items:
        :param embedding_dim:
        """
        super().__init__()
        self.user_embedding = EmbeddingLayer(num_users, embedding_dim)
        self.item_embedding = EmbeddingLayer(num_items, embedding_dim)
        self.user_bias = EmbeddingLayer(num_users, 1)
        self.item_bias = EmbeddingLayer(num_items, 1)

    def forward(self, vector):
        """
        This forward method takes a vector as input, where vector[0]
        represents the user indices and vector[1] represents the item
        indices. It performs the forward pass of the matrix factorization
        model by applying the embedding layers to the user and item indices,
        computing the dot product of the embeddings, and adding the user
        and item biases. The result is returned as the predicted ratings.

        :param vector:
        :return:
        """
        users = vector[0].long()
        items = vector[1].long()
        ues = self.user_embedding(users)
        uis = self.item_embedding(items)
        ubs = self.user_bias(users)
        ibs = self.item_bias(items)
        return (ues * uis).sum(1) + ubs.squeeze() + ibs.squeeze()
