import h5py
import os
import numpy as np
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import click
"""
This 
"""

def convert_to_float_or_string(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.replace('.', '', 1).isdigit():
        return float(value)
    return str(value)

class SentimentAnalyzer:
    def __init__(self, file_name) -> None:
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.max_len = self.sentiment_pipeline.tokenizer.model_max_length
        self.file = h5py.File(file_name,'r+')

    def get_sqls(self, key):
        shot_data = self.file[key]
        text_sql = str(shot_data['text_sql'][()])[:self.max_len]
        sent_output = self.sentiment_pipeline(text_sql)[0]
        sent_score = sent_output['score'] * -1 if sent_output['label'] == 'NEGATIVE' else sent_output['score']
        data = self.file[key]
        sql_vars = [s for s in data.keys() if 'sql' in s and s != "text_sql"]
        out_dic = {s: convert_to_float_or_string(data[s][()]) for s in sql_vars}
        out_dic['sentiment'] = sent_score
        out_dic["text_sql"] = text_sql
        return out_dic

    def build_df(self, out_dir, check_point_freq=1000):
        dicts = []
        for i, k in enumerate(tqdm(self.file.keys())):
            if i % check_point_freq == 0:
                pd.DataFrame(dicts).to_pickle(os.path.join(out_dir, f"checkpoint_{i}.pkl"))
            if k != "spatial_coordinates" and k != "times":
                try:
                    sqls = self.get_sqls(k)
                    dicts.append(sqls)
                except:
                    pass
        return pd.DataFrame(dicts)

@click.command()
@click.argument('in_dir', type=click.Path(exists=True))
@click.argument('out_dir', type=click.Path(exists=True))
@click.argument('file_name', type=click.STRING)
def run(in_dir, out_dir, file_name):
    anal = SentimentAnalyzer(os.path.join(in_dir, file_name))
    df = anal.build_df(out_dir=out_dir)
    pd.to_pickle(df, os.path.join(out_dir, file_name.replace(".h5", ".pkl")))

if __name__ == "__main__":
    run()