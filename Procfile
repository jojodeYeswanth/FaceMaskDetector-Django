web: gunicorn facemask_detector.wsgi:detector --log-file - --log-level debug
python manage.py collectstatic --noinput
manage.py migrate