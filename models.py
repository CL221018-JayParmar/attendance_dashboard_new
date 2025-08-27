from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class Admin(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), nullable=False)
    contact = db.Column(db.String(20), nullable=False)
    face_data_folder = db.Column(db.String(200))

    attendances = db.relationship(
        'Attendance',
        backref='employee',
        lazy=True,
        cascade='all, delete-orphan',
        passive_deletes=True
    )

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id', ondelete='CASCADE'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
