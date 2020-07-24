web: gunicorn facemask_detctor.wsgi:detector --log-file - --log-level debug
python manage.py collectstatic --noinput
manage.py migrate