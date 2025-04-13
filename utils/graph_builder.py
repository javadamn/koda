# graphrag/graph_builder.py
import networkx as nx
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def create_graphrag_knowledge_graph(pairwise_data, strain_mean_biomass, microbial_abundance, metabolite_info):
    G = nx.DiGraph()

    for pair in pairwise_data:
        pair["crossfed metabolites"] = {
            met_id[3:].rstrip('(e)'): values
            for met_id, values in pair["crossfed metabolites"].items()
        }

    for strain, biomass in strain_mean_biomass.items():
        if pd.isna(biomass):
            logger.warning(f"Skipping strain {strain} with NaN biomass")
            continue
        G.add_node(strain,
                   type="microbe",
                   abundance=microbial_abundance.get(strain, None))

    met_data_lookup = {m['reaction']: m for m in metabolite_info}

    for pair in pairwise_data:
        s1, s2 = pair["bacteria1"], pair["bacteria2"]
        for met_id, (flux, *_) in pair["crossfed metabolites"].items():
            met_data = met_data_lookup.get(met_id + '(e)')
            if met_data and met_id not in G:
                G.add_node(met_id,
                           type="metabolite",
                           name=met_data['name'])

        if s1 not in G or s2 not in G:
            logger.debug(f"Skipping pair {s1}-{s2}: Missing node(s)")
            continue

        G.add_edge(s1, s2,
                   type="interaction",
                   source_biomass=pair["bacteria1 biomass"],
                   target_biomass=pair["bacteria2 biomass"])

        for met_id, (flux, *_) in pair["crossfed metabolites"].items():
            if met_id not in G:
                logger.warning(f"Missing metabolite: {met_id}")
                continue
            met_data = met_data_lookup.get(met_id + '(e)')
            G.add_edge(s1, met_id, type="produces", flux=flux,
                       description=f"{s1} produces {met_data['name']}")
            G.add_edge(met_id, s2, type="consumes", flux=flux,
                       description=f"{s2} consumes {met_data['name']}")

    return G
