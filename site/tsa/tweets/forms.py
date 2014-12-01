import inspect

from django import forms
import enum

from TwitterSentimentAnalysis.ai import AIEnum
from statistics import StatisticEnum
from models import ArtificialIntelligence


class ActionEnum(enum.Enum):
    """An enum indicating whether creation of new Analysis should be performed or an old one should be loaded"""
    Create = 1
    Load = 2

    @classmethod
    def choices(cls):
        """
        A method to return all possible choices for Action enum values

        :return: a list of tuples of all possible enum choices
        """
        members = inspect.getmembers(cls, lambda memb: not(inspect.isroutine(memb)))
        props = [m for m in members if not(m[0][:2] == '__')]
        return tuple([(p[1].value, p[0]) for p in props])


class QueryForm(forms.Form):
    """A form used for making queries in search view of a web site

    It provides information like search query, a query limit, or how query results should be saved
    """
    query = forms.CharField(label='Search', max_length=100, required=False)
    limit = forms.IntegerField(label='Limit', required=False)
    name = forms.CharField(label='Name', max_length=100, required=False)
    page = forms.IntegerField(
        widget=forms.HiddenInput(), initial=1, required=False)
    undersample = forms.BooleanField(
        label="Undersample", initial=False, required=False)


class AnalysisForm(forms.ModelForm):
    """A form based on ArtificialIntelligence model used for performing analysis of a data

    It provides information like an action to perform (create/load) which AI to use and on which data
    """
    def __init__(self, tweet_sets, saved_ais, *args, **kwargs):
        super(AnalysisForm, self).__init__(*args, **kwargs)
        self.fields['tweet_sets'] = forms.ChoiceField(
            choices=tweet_sets)
        self.fields['saved_ais'] = forms.ChoiceField(
            choices=saved_ais)
        for key in self.fields:
            self.fields[key].required = False

    ai_types = forms.TypedChoiceField(
        choices=AIEnum.choices(),
        coerce=str,
        initial=AIEnum.MultiClassClassificationNeuralNetwork.name)
    action = forms.ChoiceField(
        choices=ActionEnum.choices(),
        widget=forms.RadioSelect(),
        initial=ActionEnum.Create.value)
    custom_tweet_set = forms.BooleanField(
        label="Custom tweet set", initial=True)
    save_results = forms.BooleanField(label="Save", initial=True)

    class Meta:
        model = ArtificialIntelligence
        fields = ('name',)


class StatisticsForm(forms.Form):
    """A form used for displaying charts from analyzed data

    It provides information like what type of chart to show or from which set of analyzed data
    """
    tweet_sets = forms.ModelChoiceField(
        queryset=ArtificialIntelligence.objects.all(),
        empty_label=None)

    statistic_types = forms.TypedChoiceField(
        choices=StatisticEnum.choices(),
        coerce=str,
        initial=StatisticEnum.sample.name)