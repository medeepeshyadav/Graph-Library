from django import forms

class AForm(forms.Form):
    input_file = forms.FileField(label='Input File')
    graph_type = forms.ChoiceField(choices=(("directed", "directed"), ("undirected", "undirected")), label='Graph Type')
    feature_type = forms.ChoiceField(choices=(("basic", "basic"), ("advanced", "advance"), ("all", "all")), label='Feature Type')
    test_size = forms.DecimalField(max_value=1.0, min_value=0.0, label='Test Size')
    n_jobs = forms.IntegerField(min_value=1, label='No. of Workers')


class BForm(forms.Form):
    key = forms.UUIDField(label='Enter Key')