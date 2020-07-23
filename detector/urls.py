from django.conf.urls.static import static
from django.contrib import admin
from django.conf.urls import url
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
                  url(r'^/(?P<stream_path>(.*?))/$', views.dynamic_stream, name="videostream"),
                  url(r'^detect-video/$', views.indexscreen, name='detect_video'),

                  url(r'^admin/', admin.site.urls),
                  path('', views.show_video, name='home'),
                  path('upload/', views.upload, name='upload'),

              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
