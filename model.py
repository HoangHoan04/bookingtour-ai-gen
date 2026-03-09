import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

TOUR_VECTORS = {
    1: [1, 0, 0],
    2: [0, 1, 0],
    3: [0, 0, 1],
    4: [1, 1, 0],
}

def recommend(user_vector, top_k=3):
    tour_ids = list(TOUR_VECTORS.keys())
    vectors = np.array(list(TOUR_VECTORS.values()))

    scores = cosine_similarity([user_vector], vectors)[0]
    ranked = sorted(zip(tour_ids, scores), key=lambda x: x[1], reverse=True)

    return ranked[:top_k]