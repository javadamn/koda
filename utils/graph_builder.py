import networkx as nx
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def create_graphrag_knowledge_graph(pairwise_data, strain_mean_biomass,
                                    microbial_abundance, metabolite_info, subsystem_scores_per_instance):
    
    G = nx.DiGraph()

    for pair in pairwise_data:
        pair["crossfed metabolites"] = {met_id[3:].rstrip('(e)'): values for met_id, values in pair["crossfed metabolites"].items()}

    for strain, biomass in strain_mean_biomass.items():
        if pd.isna(biomass):
            logger.warning(f"Skipping strain {strain} with NaN biomass")
            continue
        G.add_node(strain,
                   type="microbe",
                   abundance=microbial_abundance.get(strain, None))

    met_data_lookup = {m['reaction']: m for m in metabolite_info}

    #metabolite-microbe nodes
    for pair in pairwise_data:
        s1, s2 = pair["bacteria1"], pair["bacteria2"]
        for met_id, (flux, *_) in pair["crossfed metabolites"].items():
            met_data = met_data_lookup.get(met_id + '(e)')
            if met_data and met_id not in G:
                G.add_node(met_data['name'], type="metabolite", name=met_data['name'])

        if s1 not in G or s2 not in G:
            logger.debug(f"Skipping pair {s1}-{s2}: Missing node(s)")
            continue

        G.add_edge(s1, s2,
                   type="cross-feeds with",
                   source_biomass=pair["bacteria1 biomass"],
                   target_biomass=pair["bacteria2 biomass"])

        for met_id, (flux, *_) in pair["crossfed metabolites"].items():
            met_key = met_id + '(e)'
            met_data = met_data_lookup.get(met_key)
            if not met_data:
                logger.warning(f"Missing metabolite info for {met_id}")
                continue
            met_name = met_data['name']
            if met_name not in G:
                G.add_node(met_name, type="metabolite", name=met_name)
            G.add_edge(s1, met_name, type="produces", flux=flux,description=f"{s1} produces {met_name}")
            G.add_edge(met_name, s2, type="consumes", flux=flux,description=f"{s2} consumes {met_name}")
    
    #pathway-microbe links
    for microbe, instance_dict in subsystem_scores_per_instance.items():
        for instance, subsys_scores in instance_dict.items():
            for subsystem, score in subsys_scores.items():
                if not subsystem:
                    continue  #skip blank

                if not G.has_node(subsystem):
                    G.add_node(subsystem, type="subsystem", name=subsystem)

                # Link microbe to subsystem
                if microbe in G:
                    G.add_edge(subsystem, microbe,
                               type="involved-in",
                               subsystem_score=score,
                               instance=instance,
                               description=f"{subsystem} is active in {microbe} with importance score of {score:.3f}")

    return G
