"""Strict ID/order and integer checks, supplementing official self-check."""
import argparse
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--submission',required=True)
    p.add_argument('--test-csv',required=True)
    p.add_argument('--template',default=None)
    a=p.parse_args()
    s,t=pd.read_csv(a.submission),pd.read_csv(a.test_csv)
    if list(s.columns)!=['sample_id','label']: raise ValueError('Unexpected columns')
    if len(s)!=len(t) or not s.sample_id.equals(t.sample_id): raise ValueError('Test ID/order mismatch')
    if not s.sample_id.is_unique or s.sample_id.isna().any(): raise ValueError('Duplicate/null IDs')
    if not pd.api.types.is_integer_dtype(s.label) or not s.label.isin(range(1,7)).all():
        raise ValueError('Labels must be integers 1..6')
    if a.template:
        template=pd.read_csv(a.template)
        if not s.sample_id.equals(template.sample_id): raise ValueError('Template ID/order mismatch')
    print(f'PASS strict submission format: {len(s)} rows, exact ID/order, integer labels 1..6')


if __name__=='__main__': main()
