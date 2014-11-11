from django import forms


class QueryForm(forms.Form):
    query = forms.CharField(label='Search', max_length=100)
    limit = forms.IntegerField(label='Limit')
    name = forms.CharField(label='Name', max_length=100)
    page = forms.IntegerField(widget=forms.HiddenInput(), initial=1)