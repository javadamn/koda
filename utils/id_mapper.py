# utils/.py
def build_id_to_strain_map(xml_models_df):
    xml_models_df['Strain'] = xml_models_df['Strain'].str.replace('.xml', '', regex=False)
    return xml_models_df.set_index('ID')['Strain'].to_dict()

def replace_ids_with_names(pairwise_data, id_to_strain):
    for entry in pairwise_data:
        if entry['bacteria1'].isdigit():
            entry['bacteria1'] = id_to_strain.get(int(entry['bacteria1']), entry['bacteria1'])
        if entry['bacteria2'].isdigit():
            entry['bacteria2'] = id_to_strain.get(int(entry['bacteria2']), entry['bacteria2'])
    return pairwise_data
