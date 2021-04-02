from flask import render_template
import os
from app.main import bp
from app.ai import plants_mobilenetv2


@bp.route('/')
@bp.route('/index')
def index():
    arr_class_names = plants_mobilenetv2.get_class_names()
    return render_template('plants/plants.html', class_names=arr_class_names)
