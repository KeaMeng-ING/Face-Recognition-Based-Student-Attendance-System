import insightface
from insightface.app import FaceAnalysis
import cv2

app = FaceAnalysis(name='buffalo_sc')
app.prepare(ctx_id=-1)  # -1 = CPU

def align_face(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)

    if len(faces) == 0:
        return None  # no face detected, discard image

    face = faces[0]  # take the largest face
    # insightface gives you the aligned 112x112 crop directly
    return face.normed_embedding, face.embedding
