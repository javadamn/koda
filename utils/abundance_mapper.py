# utils/.py
import pandas as pd
import ast

def get_abundance(abundance_file, species_models_file, xml_models_file):
    abundance_df = pd.read_csv(abundance_file)
    models_species_df = pd.read_csv(species_models_file)
    xml_models_df = pd.read_csv(xml_models_file)

    models_species_df['models'] = models_species_df['models'].apply(ast.literal_eval)
    species_to_models = models_species_df.explode('models')
    merged_df = species_to_models.merge(xml_models_df, left_on='models', right_on='Strain', how='inner')

    species_to_model_ids = merged_df.groupby('species')['ID'].apply(list).to_dict()
    samples = abundance_df['samples']
    abundance_df = abundance_df.set_index('samples')
    microbial_abundance = pd.DataFrame(index=samples)

    for species, model_ids in species_to_model_ids.items():
        if species in abundance_df.columns:
            divided_abundance = abundance_df[species] / len(model_ids)
            for model_id in model_ids:
                microbial_abundance[model_id] = microbial_abundance.get(model_id, 0) + divided_abundance

    return microbial_abundance.fillna(0), xml_models_df
