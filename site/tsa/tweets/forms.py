from django import forms


class QueryForm(forms.Form):
    query = forms.CharField(label='Search', max_length=100, required=False)
    limit = forms.IntegerField(label='Limit', required=False)
    name = forms.CharField(label='Name', max_length=100, required=False)
    page = forms.IntegerField(widget=forms.HiddenInput(), initial=1)