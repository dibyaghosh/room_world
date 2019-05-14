import numpy as np

class RewardAbstraction:
    def reward(self, state, goal):
        raise NotImplementedError()

class Embedding:
    def get_embedding(self, state):
        return self.get_embeddings(state[None])[0]

    def get_embeddings(self, states):
        raise NotImplementedError()

class ReconstructedEmbedding:
    def get_embedding(self, state):
        return self.get_embeddings(state[None])[0]

    def get_embeddings(self, states):
        raise NotImplementedError()
   
    def get_state(self, embedding):
        return self.get_states(embedding[None])[0]
    
    def get_states(self, embeddings):
        raise NotImplementedError()

class IdentityEmbedding(ReconstructedEmbedding):
    def get_embedding(self, state):
        return state

    def get_embeddings(self, states):
        return states
    
    def get_state(self, embedding):
        return embedding

    def get_states(self, embeddings):
        return embeddings

class EmbeddingReward(RewardAbstraction):
    def __init__(self, embedding, scale=1):
        self.embedding = embedding
        self.scale = scale
    
    def reward(self, state, goal):
        state_embedding = self.embedding.get_embedding(state)
        goal_embedding = self.embedding.get_embedding(goal)
        return -1 * self.scale * np.linalg.norm(state_embedding - goal_embedding)

class StateReward(EmbeddingReward):
    def __init__(self, scale=1):
        super().__init__(IdentityEmbedding(), scale)