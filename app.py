import os
import uuid
import cv2
import pickle
import numpy as np
import face_recognition
from datetime import datetime
from io import StringIO
import csv
from flask import (
    Flask, render_template, redirect, url_for,
    flash, request, jsonify, Response
)
from flask_login import (
    LoginManager, login_user, login_required,
    logout_user
)
from flask_mail import Mail, Message
from werkzeug.security import check_password_hash
from models import db, Admin, Employee, Attendance
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
mail = Mail(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(Admin, int(user_id))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        admin = Admin.query.filter_by(username=username).first()
        if admin and check_password_hash(admin.password_hash, password):
            login_user(admin)
            return redirect(url_for("dashboard"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/")
@login_required
def dashboard():
    employees = Employee.query.all()
    return render_template("dashboard.html", employees=employees)

@app.route("/employee/new", methods=["GET", "POST"])
@login_required
def new_employee():
    if request.method == "POST":
        name = request.form.get("name")
        department = request.form.get("department")
        email = request.form.get("email")
        contact = request.form.get("contact")
        emp = Employee(
            name=name,
            department=department,
            email=email,
            contact=contact,
            face_data_folder=None,
        )
        db.session.add(emp)
        db.session.commit()
        flash("Employee added. Now capture face.", "success")
        return redirect(url_for("capture_face", id=emp.id))
    return render_template("employee_form.html", employee=None)

@app.route("/employee/edit/<int:id>", methods=["GET", "POST"])
@login_required
def edit_employee(id):
    emp = Employee.query.get_or_404(id)
    if request.method == "POST":
        emp.name = request.form.get("name")
        emp.department = request.form.get("department")
        emp.email = request.form.get("email")
        emp.contact = request.form.get("contact")
        db.session.commit()
        flash("Employee updated.", "success")
        return redirect(url_for("dashboard"))
    return render_template("employee_form.html", employee=emp)

@app.route("/employee/delete/<int:id>", methods=["POST"])
@login_required
def delete_employee(id):
    emp = Employee.query.get_or_404(id)
    if emp.face_data_folder and os.path.exists(emp.face_data_folder):
        for file in os.listdir(emp.face_data_folder):
            os.remove(os.path.join(emp.face_data_folder, file))
        os.rmdir(emp.face_data_folder)
    emb_path = os.path.join("static", "embeddings", f"{emp.id}_embeddings.pkl")
    if os.path.exists(emb_path):
        os.remove(emb_path)
    db.session.delete(emp)
    db.session.commit()
    flash("Employee deleted.", "warning")
    return redirect(url_for("dashboard"))

@app.route("/capture/<int:id>", methods=["GET", "POST"])
@login_required
def capture_face(id):
    emp = Employee.query.get_or_404(id)
    if request.method == "POST":
        video = request.files.get("video")
        if not video:
            return jsonify(message="No video uploaded"), 400
        raw_folder = os.path.join("static", "captures", "raw")
        os.makedirs(raw_folder, exist_ok=True)
        raw_path = os.path.join(raw_folder, f"{uuid.uuid4()}.webm")
        video.save(raw_path)
        out_folder = os.path.join("static", "captures", str(emp.id))
        os.makedirs(out_folder, exist_ok=True)

        vid = cv2.VideoCapture(raw_path)
        frame_count = 0
        ear_threshold = 0.25
        consecutive_frames = 0
        saved_images = []

        def eye_aspect_ratio(eye):
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            C = np.linalg.norm(eye[0] - eye[3])
            return (A + B) / (2.0 * C)

        while True:
            ret, frame = vid.read()
            if not ret:
                break
            if frame_count % 5 != 0:
                frame_count += 1
                continue
            small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            landmarks_list = face_recognition.face_landmarks(rgb)
            if landmarks_list:
                lm = landmarks_list[0]
                if "left_eye" in lm and "right_eye" in lm:
                    left = np.array(lm["left_eye"]); right = np.array(lm["right_eye"])
                    ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
                    if ear < ear_threshold:
                        consecutive_frames += 1
                    else:
                        if consecutive_frames >= 2:
                            path = os.path.join(out_folder, f"{emp.id}_{frame_count}.jpg")
                            cv2.imwrite(path, frame)
                            saved_images.append(path)
                        consecutive_frames = 0
                else:
                    if frame_count % 30 == 0:
                        path = os.path.join(out_folder, f"{emp.id}_{frame_count}.jpg")
                        cv2.imwrite(path, frame)
                        saved_images.append(path)
            frame_count += 1

        vid.release(); os.remove(raw_path)
        emp.face_data_folder = out_folder; db.session.commit()

        embeddings = []
        for img_path in saved_images:
            try:
                encs = face_recognition.face_encodings(face_recognition.load_image_file(img_path))
                if encs:
                    embeddings.append(encs[0])
            except Exception:
                pass

        emb_folder = os.path.join("static", "embeddings")
        os.makedirs(emb_folder, exist_ok=True)
        with open(os.path.join(emb_folder, f"{emp.id}_embeddings.pkl"), "wb") as f:
            pickle.dump(embeddings, f)

        return jsonify(message=f"Face data captured! {len(saved_images)} images, {len(embeddings)} embeddings saved"), 200

    return render_template("capture.html", employee=emp)

@app.route("/attendance_mark", methods=["GET", "POST"])
def attendance_mark_page():
    if request.method == "POST":
        video = request.files.get("video")
        if not video:
            return jsonify(message="No video"), 400
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.webm")
        video.save(tmp)
        vid = cv2.VideoCapture(tmp)
        ret, frame = vid.read()
        vid.release(); os.remove(tmp)
        if not ret:
            return jsonify(message="No frame detected"), 400
        encs = face_recognition.face_encodings(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not encs:
            return jsonify(message="No face detected"), 400
        live = encs[0]; recognized = None
        for emp in Employee.query.all():
            p = os.path.join("static", "embeddings", f"{emp.id}_embeddings.pkl")
            if os.path.exists(p):
                with open(p,"rb") as f:
                    stored = pickle.load(f)
                if any(face_recognition.compare_faces(stored, live, tolerance=0.6)):
                    recognized = emp; break
        if not recognized:
            return jsonify(message="Invalid Face – Attendance Not Marked"), 400
        today = datetime.utcnow().date()
        if Attendance.query.filter(
            Attendance.employee_id==recognized.id,
            db.func.date(Attendance.timestamp)==today
        ).first():
            return jsonify(message="Attendance already marked today"), 409
        att = Attendance(employee_id=recognized.id, timestamp=datetime.utcnow())
        db.session.add(att); db.session.commit()
        try:
            msg = Message("Attendance Confirmation",
                          sender=app.config["MAIL_USERNAME"],
                          recipients=[recognized.email])
            msg.body = f"Your attendance was marked at {att.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            mail.send(msg)
        except Exception:
            pass
        return jsonify(message=f"Attendance Marked! {recognized.name}"), 200

    return render_template("attendance_mark.html")

@app.route("/attendance_dashboard")
@login_required
def attendance_dashboard():
    date = request.args.get("date")
    emp_name = request.args.get("employee")
    q = Attendance.query.join(Employee)
    if date:
        try:
            d = datetime.fromisoformat(date).date()
            q = q.filter(db.func.date(Attendance.timestamp)==d)
        except Exception:
            pass
    if emp_name:
        q = q.filter(Employee.name.ilike(f"%{emp_name}%"))
    records = q.order_by(Attendance.timestamp.desc()).all()
    unique_days = {r.timestamp.date() for r in records}
    return render_template(
        "attendance_dashboard.html",
        records=records,
        unique_days_count=len(unique_days)
    )

@app.route("/export_attendance_csv")
@login_required
def export_csv():
    date = request.args.get("date")
    emp_name = request.args.get("employee")
    q = Attendance.query.join(Employee)
    if date:
        try:
            d = datetime.fromisoformat(date).date()
            q = q.filter(db.func.date(Attendance.timestamp)==d)
        except Exception:
            pass
    if emp_name:
        q = q.filter(Employee.name.ilike(f"%{emp_name}%"))
    records = q.order_by(Attendance.timestamp.desc()).all()
    output = StringIO(); writer = csv.writer(output)
    writer.writerow(["Name","Department","Date","Time"])
    for r in records:
        writer.writerow([
            r.employee.name,
            r.employee.department,
            r.timestamp.strftime('%Y-%m-%d'),
            r.timestamp.strftime('%H:%M:%S')
        ])
    csv_data = output.getvalue(); output.close()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition":"attachment;filename=attendance_records.csv"}
    )

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)
