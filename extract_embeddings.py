import os, pickle
import face_recognition
from models import Employee
from app import app

with app.app_context():
    for emp in Employee.query.all():
        folder = os.path.join('static','captures',str(emp.id))
        if not os.path.isdir(folder): continue
        embeddings=[]
        for f in os.listdir(folder):
            img=face_recognition.load_image_file(os.path.join(folder,f))
            e=face_recognition.face_encodings(img)
            if e: embeddings.append(e[0])
        os.makedirs('static/embeddings', exist_ok=True)
        with open(f'static/embeddings/{emp.id}_embeddings.pkl','wb') as out:
            pickle.dump(embeddings,out)
        print(f"Saved {len(embeddings)} embeddings for {emp.name}")
