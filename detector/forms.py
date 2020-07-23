from django import forms
from detector.models import Video


class VideoForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ["name", "video_file"]