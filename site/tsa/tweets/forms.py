from django import forms
from TwitterSentimentAnalysis.neuralNetworks import AIEnum


class QueryForm(forms.Form):
    query = forms.CharField(label='Search', max_length=100, required=False)
    limit = forms.IntegerField(label='Limit', required=False)
    name = forms.CharField(label='Name', max_length=100, required=False)
    page = forms.IntegerField(widget=forms.HiddenInput(), initial=1, required=False)


class AnalysisForm(forms.Form):
    def __init__(self, tweet_sets, *args, **kwargs):
        super(AnalysisForm, self).__init__(*args, **kwargs)
        self.fields['tweet_sets'] = forms.ChoiceField(choices=tweet_sets)

    ai_types = forms.TypedChoiceField(choices=AIEnum.choices(), coerce=str)
