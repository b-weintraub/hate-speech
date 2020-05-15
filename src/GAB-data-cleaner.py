import json
import pandas as pd
import collections
import sys

#convert json to list of dictionaries
def import_data(filepath):
    reports = []
    with open(filepath) as f:
        for i in f:
            reports.append(json.loads(i))
    return reports

#flatten nested dictionaries
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

#load flattened dictionary into dataframe as single row entry
def create_dataframe(reports):
    reports_df = pd.DataFrame()
    for report in reports:
        try:
            
            report_dict = flatten(report)
            report_df = pd.DataFrame(report_dict, index=[0])
            reports_df = reports_df.append(report_df, ignore_index=True)
        except:
            print('error in report')
    return reports_df


reports = import_data('GABPOSTS_2018-01')

create_dataframe(reports)