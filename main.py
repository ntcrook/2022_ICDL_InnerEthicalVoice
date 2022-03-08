import numpy as np
import math

def positionalEmbedding2D(x,y,embeddingDim):
    D = float(embeddingDim)
    embedding = np.empty(embeddingDim)
    for i in range(0,embeddingDim//2):
        if i % 2 == 0:
            embedding[i] = math.cos(x/10000**(4*i/D))
            embedding[i + embeddingDim//2] = math.cos(y/10000**(4*i/D))
        else:
            embedding[i] = math.sin(x/10000**(4*i/D))
            embedding[i + embeddingDim//2] = math.sin(y/10000**(4*i/D))
    return embedding

e = positionalEmbedding2D(10,20,512)
np.savetxt("embeddingTest.csv", e, delimiter = ",")

print('Embedding saved')