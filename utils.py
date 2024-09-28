import numpy as np


def faces_distance_cosine(known_face_encodings, face_encoding_2check, tolerance=-0.00000001):
    if len(known_face_encodings) == 0:
        return np.empty((0,))

    if str(type(known_face_encodings[0])) != "<class 'numpy.ndarray'>":
        raise ValueError('Elements of known face encodings must be NumPy ndarry')

    result = []
    for row in known_face_encodings:
        result.append(cosine_similarity(row, face_encoding_2check))

    return result


def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)
