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
        raw_filename = f"{uuid.uuid4()}.webm"
        raw_path = os.path.join(raw_folder, raw_filename)
        video.save(raw_path)

        out_folder = os.path.join("static", "captures", str(emp.id))
        os.makedirs(out_folder, exist_ok=True)

        vid = cv2.VideoCapture(raw_path)
        frame_count = 0
        blink_count = 0
        ear_threshold = 0.25
        consecutive_frames = 0

        def eye_aspect_ratio(eye):
            A = np.linalg.norm(eye[1] - eye)
            B = np.linalg.norm(eye[1] - eye[2])
            C = np.linalg.norm(eye - eye[2])
            return (A + B) / (2.0 * C)

        saved_images = []

        while True:
            ret, frame = vid.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_landmarks(rgb)
            if faces:
                landmarks = faces
                leftEye = landmarks["left_eye"]
                rightEye = landmarks["right_eye"]
                leftEAR = eye_aspect_ratio(np.array(leftEye))
                rightEAR = eye_aspect_ratio(np.array(rightEye))
                ear = (leftEAR + rightEAR) / 2.0

                if ear < ear_threshold:
                    consecutive_frames += 1
                else:
                    if consecutive_frames >= 2:
                        blink_count += 1
                        filename = f"{emp.id}_{frame_count}.jpg"
                        path = os.path.join(out_folder, filename)
                        cv2.imwrite(path, frame)
                        saved_images.append(path)
                    consecutive_frames = 0
            frame_count += 1

        vid.release()
        os.remove(raw_path)

        emp.face_data_folder = out_folder
        db.session.commit()

        embeddings = []
        for img_path in saved_images:
            img = face_recognition.load_image_file(img_path)
            encs = face_recognition.face_encodings(img)
            if encs:
                embeddings.append(encs[0])

        embeddings_folder = os.path.join("static", "embeddings")
        os.makedirs(embeddings_folder, exist_ok=True)
        emb_file_path = os.path.join(embeddings_folder, f"{emp.id}_embeddings.pkl")

        with open(emb_file_path, "wb") as f:
            pickle.dump(embeddings, f)

        return jsonify(message="Face data captured and embeddings saved"), 200

    return render_template("capture.html", employee=emp)

@app.route("/attendance_mark", methods=["GET", "POST"])
def attendance_mark_page():
    if request.method == "POST":
        video = request.files.get("video")
        if not video:
            return jsonify(message="No video"), 400

        import tempfile
        temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.webm")
        video.save(temp_path)

        vid = cv2.VideoCapture(temp_path)
        ret, frame = vid.read()
        vid.release()

        if os.path.exists(temp_path):
            os.remove(temp_path)
        if not ret:
            return jsonify(message="No frame detected"), 400

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if not encs:
            return jsonify(message="No face detected"), 400

        live_enc = encs[0]
        recognized_emp = None

        for emp in Employee.query.all():
            emb_path = os.path.join("static", "embeddings", f"{emp.id}_embeddings.pkl")
            if not os.path.exists(emb_path):
                continue

            with open(emb_path, "rb") as f:
                stored = pickle.load(f)

            matches = face_recognition.compare_faces(stored, live_enc, tolerance=0.6)
            if any(matches):
                recognized_emp = emp
                break

        if not recognized_emp:
            return jsonify(message="Invalid Face – Attendance Not Marked"), 400

        today = datetime.utcnow().date()
        existing = Attendance.query.filter(
            Attendance.employee_id == recognized_emp.id,
            db.func.date(Attendance.timestamp) == today
        ).first()

        if existing:
            return jsonify(message=f"Attendance already marked today at {existing.timestamp.strftime('%H:%M')}"), 409

        att = Attendance(employee_id=recognized_emp.id, timestamp=datetime.utcnow())
        db.session.add(att)
        db.session.commit()

        try:
            msg = Message(
                "Attendance Confirmation",
                sender=app.config["MAIL_USERNAME"],
                recipients=[recognized_emp.email],
            )
            msg.body = f"Your attendance was marked successfully at {att.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            mail.send(msg)
        except Exception as e:
            print(f"Email send error: {e}")

        return jsonify(message=f"Attendance Marked Successfully!<br>{recognized_emp.name} - {recognized_emp.department}"), 200

    return render_template("attendance_mark.html")

@app.route("/attendance_dashboard")
@login_required
def attendance_dashboard():
    date = request.args.get("date")
    employee_name = request.args.get("employee")
    q = Attendance.query.join(Employee)

    if date:
        try:
            day = datetime.fromisoformat(date)
            q = q.filter(db.func.date(Attendance.timestamp) == day.date())
        except Exception:
            pass

    if employee_name:
        q = q.filter(Employee.name.ilike(f"%{employee_name}%"))

    records = q.order_by(Attendance.timestamp.desc()).all()
    return render_template("attendance_dashboard.html", records=records)

@app.route("/export_attendance_csv")
@login_required
def export_csv():
    date = request.args.get("date")
    employee_name = request.args.get("employee")
    q = Attendance.query.join(Employee)

    if date:
        try:
            day = datetime.fromisoformat(date)
            q = q.filter(db.func.date(Attendance.timestamp) == day.date())
        except Exception:
            pass

    if employee_name:
        q = q.filter(Employee.name.ilike(f"%{employee_name}%"))

    records = q.order_by(Attendance.timestamp.desc()).all()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name", "Department", "Date", "Time"])

    for r in records:
        writer.writerow([
            r.employee.name,
            r.employee.department,
            r.timestamp.date(),
            r.timestamp.strftime('%H:%M:%S')
        ])

    csv_data = output.getvalue()
    output.close()

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=attendance_records.csv"}
    )


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)
