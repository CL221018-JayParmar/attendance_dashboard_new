import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = 'bdc154ec3e6818b91a45ea863dc817b1'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir,'data','employees.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = 'jay8085135710@gmail.com'
    MAIL_PASSWORD = 'Jparmar@1234'  # Gmail App Password recommend karein
    MAIL_DEFAULT_SENDER = 'jay8085135710@gmail.com'
