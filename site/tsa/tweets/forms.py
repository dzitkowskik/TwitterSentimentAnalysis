import inspect
from django import forms
import enum
from TwitterSentimentAnalysis.ai import AIEnum
from TwitterSentimentAnalysis.statistics import StatisticEnum
from models import ArtificialIntelligence


class ActionEnum(enum.Enum):
    Create = 1
    Load = 2

    @classmethod
    def choices(cls):
        members = inspect.getmembers(cls, lambda memb: not(inspect.isroutine(memb)))
        props = [m for m in members if not(m[0][:2] == '__')]
        return tuple([(p[1].value, p[0]) for p in props])


class QueryForm(forms.Form):
    query = forms.CharField(label='Search', max_length=100, required=False)
    limit = forms.IntegerField(label='Limit', required=False)
    name = forms.CharField(label='Name', max_length=100, required=False)
    page = forms.IntegerField(widget=forms.HiddenInput(), initial=1, required=False)


class AnalysisForm(forms.Form):
    def __init__(self, tweet_sets, saved_ais, *args, **kwargs):
        super(AnalysisForm, self).__init__(*args, **kwargs)
        self.fields['tweet_sets'] = forms.ChoiceField(
            choices=tweet_sets,
            required=False)
        self.fields['saved_ais'] = forms.ChoiceField(
            choices=saved_ais,
            required=False)

    ai_types = forms.TypedChoiceField(
        choices=AIEnum.choices(),
        coerce=str,
        initial=AIEnum.MultiClassClassificationNeuralNetwork.name)
    action = forms.ChoiceField(
        choices=ActionEnum.choices(),
        widget=forms.RadioSelect(),
        initial=ActionEnum.Create.value)
    custom_tweet_set = forms.BooleanField(
        label="Custom tweet set", initial=True, required=False)
    save_results = forms.BooleanField(
        label="Save", initial=True, required=False)
    name = forms.CharField(
        label='Name', max_length=40, required=False)


class StatisticsForm(forms.Form):
    tweet_sets = forms.ModelChoiceField(
        queryset=ArtificialIntelligence.objects.all(),
        empty_label=None)

    statistic_types = forms.TypedChoiceField(
        choices=StatisticEnum.choices(),
        coerce=str,
        initial=StatisticEnum.sample.name)