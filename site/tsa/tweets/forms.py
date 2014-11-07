from django import forms


class QueryForm(forms.Form):
    query = forms.CharField(label='Search', max_length=100)