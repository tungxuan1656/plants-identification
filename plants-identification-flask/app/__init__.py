import logging
from flask import Flask
import os
from logging.handlers import RotatingFileHandler


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'secret_key'


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(Config)
    # app.config.from_envvar('FLASK_RUN_SETTINGS')
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)

    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    if not app.debug and not app.testing:
        # logging
        if not os.path.exists('logs'):
            os.mkdir('logs')

        file_handler = RotatingFileHandler('logs/server.log', maxBytes=10240, backupCount=10)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Server started')

    return app

# from app import routes
