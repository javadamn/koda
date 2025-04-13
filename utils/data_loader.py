# utils/data_loader.py
import pandas as pd
import re
import ast

def clean_data(df):
    df = df.drop_duplicates(subset=['pairID'])
    df = df.dropna(subset=['b1 biomass', 'b2 biomass'])
    df = df[(df['b1 biomass'] != 0) & (df['b2 biomass'] != 0)]
    return df, df['pairID'].nunique()

def parse_crossfed_mets(raw_crossfed_mets):
    if not isinstance(raw_crossfed_mets, str):
        return {}
    cleaned_str = re.sub(r'np\.float64\((-?\d+\.?\d*(e[-+]?\d+)?)\)', r'\1', raw_crossfed_mets)
    parsed = ast.literal_eval(cleaned_str)
    return {k: [float(v[0]), float(v[1]), float(v[2])] for k, v in parsed.items()}

def read_pairwise_data(csv_path):
    df = pd.read_csv(csv_path)
    df, num_pairs = clean_data(df)

    unique_strains = sorted(df['pairID'].str.split('__', expand=True).stack().unique())
    df[['bacteria1', 'bacteria2']] = df['pairID'].str.split('__', expand=True)
    df['crossfed metabolites'] = df['crossfed mets'].apply(parse_crossfed_mets)

    biomass_df = pd.concat([
        df[['bacteria1', 'b1 biomass']].rename(columns={'bacteria1': 'bacteria', 'b1 biomass': 'biomass'}),
        df[['bacteria2', 'b2 biomass']].rename(columns={'bacteria2': 'bacteria', 'b2 biomass': 'biomass'})
    ])
    strain_mean_biomass = biomass_df.groupby('bacteria')['biomass'].mean().to_dict()

    pairwise_data = df[['bacteria1', 'bacteria2', 'b1 biomass', 'b2 biomass', 'crossfed metabolites']].rename(
        columns={'b1 biomass': 'bacteria1 biomass', 'b2 biomass': 'bacteria2 biomass'}
    ).to_dict(orient='records')

    return pairwise_data, len(unique_strains), num_pairs, unique_strains, strain_mean_biomass
