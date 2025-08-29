import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = 'ENTER YOUR SECRET KEY '
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir,'data','employees.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = 'ENTER YOUR USERNAME'
    MAIL_PASSWORD = 'ENTER YOUR PASSWORD'
    MAIL_DEFAULT_SENDER ='ENTER EMAIL'
