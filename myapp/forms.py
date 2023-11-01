from django import forms
from django.forms import ClearableFileInput
from django.core.validators import FileExtensionValidator

class ImageUploadForm(forms.Form):
    image = forms.FileField(
        label='Upload Image',
        widget=ClearableFileInput(attrs={'multiple': False}),
        help_text='Allowed formats: PNG, JPG, TIFF',
        required=True,
        error_messages={'required': 'Please select an image file.'},
        validators=[FileExtensionValidator(allowed_extensions=['png', 'jpg', 'tif', 'tiff'])]
    )