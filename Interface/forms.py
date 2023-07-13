from django import forms

class MyForm(forms.Form):
    xes_file = forms.FileField()
    choices = (
        ('Due Date Compliance', 'Due Date Compliance'),
        ('Lead Time', 'Lead Time'),
        ('Rework', 'Rework')
    )
    dropdown = forms.ChoiceField(choices=choices)
