import networkx as nx
import os
import pandas as pd
import logging
from neo4j import GraphDatabase
from config.paths import (
    PAIRWISE_PATH, STRAINS_CSV, ABUNDANCE_CSV, MODELS_CSV
)

from utils.data_loader import read_pairwise_data
from utils.abundance_mapper import get_abundance
from utils.vmhcache import VMHCacheClient
from utils.id_mapper import build_id_to_strain_map, replace_ids_with_names
from utils.save_load_graph import save_graph, load_graph
from collections import defaultdict
from neo4j import GraphDatabase
import networkx as nx
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def create_graphrag_knowledge_graph(pairwise_data, strain_mean_biomass,
                                    microbial_abundance, metabolite_info, subsystem_scores_per_microbe):
    
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
                   type="cross feeds with",
                   source_biomass=pair["bacteria1 biomass"],
                   target_biomass=pair["bacteria2 biomass"])

        for met_id, (flux, *_) in pair["crossfed metabolites"].items():
            met_data = met_data_lookup.get(met_id + '(e)')
            if not met_data:
                logger.warning(f"Missing metabolite info for {met_id}")
                continue
            met_name = met_data['name']
            if met_name not in G:
                G.add_node(met_name, type="metabolite", name=met_name)
            G.add_edge(s1, met_name, type="produces", flux=flux,description=f"{s1} produces {met_name} with a flux of {abs(flux)}")
            G.add_edge(met_name, s2, type="consumes", flux=flux,description=f"{s2} consumes {met_name} with a flux of {abs(flux)}")
    
    #pathway-microbe links
    for pathway in subsystem_scores_per_microbe['Pathways']:
        G.add_node(pathway, type='pathway', name=pathway)

    microbes = subsystem_scores_per_microbe.columns[1:]
    for microbe in microbes:
        for idx, row in subsystem_scores_per_microbe.iterrows():
            pathway = row['Pathways']
            score = row[microbe]
            if score > 0.0010: 
                G.add_edge(microbe, pathway, type='involved_in', subsystem_score=score, description=f"{pathway} is involved in {microbe} as a pathway with importance score of {score}")

    return G



# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def convert_nx_to_neo4j(G, neo4j_uri, neo4j_user, neo4j_password):
    # Initialize the Neo4j driver
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def create_node(tx, node_id, labels, properties):
        # MERGE node by name and set properties
        query = (
            f"MERGE (n:{':'.join(labels)} {{name: $name}}) "
            "SET n += $props"
        )
        tx.run(query, name=node_id, props=properties)

    def create_relationship(tx, source_name, source_labels, target_name, target_labels, rel_type, rel_props):
        # MATCH nodes and CREATE relationship with properties
        query = (
            f"MATCH (a:{':'.join(source_labels)} {{name: $source_name}}), "
            f"(b:{':'.join(target_labels)} {{name: $target_name}}) "
            f"CREATE (a)-[r:{rel_type}]->(b) "
            "SET r += $props"
        )
        tx.run(query, source_name=source_name, target_name=target_name, props=rel_props)

    with driver.session() as session:
        #clear existing data
        session.run("MATCH (n) DETACH DELETE n")

        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            if 'type' not in node_data:
                print(f"Skipping node {node_id}: Missing 'type' attribute")
                continue

            labels = []
            properties = {}

            if node_data['type'] == 'microbe':
                labels = ['Microbe']
                properties = {'name': node_id, 'abundance': node_data.get('abundance')}

            elif node_data['type'] == 'metabolite':
                labels = ['Metabolite']
                properties = {'name': node_data.get('name', node_id)}

            elif node_data['type'] == 'pathway':
                labels = ['Pathway']
                properties = {'name': node_data.get('name', node_id)}

            else:
                print(f"Skipping node {node_id}: Unknown type '{node_data['type']}'")
                continue

            #remove None values from properties
            properties = {k: v for k, v in properties.items() if v is not None}

            #create the node in Neo4j
            session.execute_write(create_node, node_id, labels, properties)

        #create all relationships
        for u, v, data in G.edges(data=True):
            if 'type' not in data:
                print(f"Skipping edge {u}->{v}: Missing 'type' attribute")
                continue

            #source and target node data
            u_data = G.nodes[u]
            v_data = G.nodes[v]

            #labels for source and target nodes
            source_labels = []
            if u_data.get('type') == 'microbe':
                source_labels = ['Microbe']
            elif u_data.get('type') == 'metabolite':
                source_labels = ['Metabolite']
            elif u_data.get('type') == 'pathway':
                source_labels = ['Pathway']

            target_labels = []
            if v_data.get('type') == 'microbe':
                target_labels = ['Microbe']
            elif v_data.get('type') == 'metabolite':
                target_labels = ['Metabolite']
            elif v_data.get('type') == 'pathway':
                target_labels = ['Pathway']

            rel_type = data['type'].upper().replace(' ', '_').replace('-', '_')
            rel_props = {k: v for k, v in data.items() if k != 'type' and v is not None}

            session.execute_write(create_relationship, u, source_labels, v, target_labels, rel_type, rel_props)

    driver.close()


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Javadad6908")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
pairwise_data, _, _, _, strain_mean_biomass = read_pairwise_data(PAIRWISE_PATH)

vmh_client = VMHCacheClient()
list_EXrxns = sorted({met for pair in pairwise_data for met in pair["crossfed metabolites"]})
list_EXrxns = [rxn[3:] for rxn in list_EXrxns]
metabolite_info = vmh_client.get_met_info(list_EXrxns)

microbial_abundance, xml_models_df = get_abundance(ABUNDANCE_CSV, MODELS_CSV, STRAINS_CSV)

id_to_strain = build_id_to_strain_map(xml_models_df)
pairwise_data = replace_ids_with_names(pairwise_data, id_to_strain)

microbial_abundance.columns = microbial_abundance.columns.astype(str)
strain_mean_biomass_df = pd.DataFrame([strain_mean_biomass]).rename(columns=str)
common_cols = microbial_abundance.columns.intersection(strain_mean_biomass_df.columns)
microbial_abundance = microbial_abundance[common_cols]
strain_mean_biomass_df = strain_mean_biomass_df[common_cols]
strain_mean_biomass = {id_to_strain[int(k)]: v for k, v in strain_mean_biomass.items() if int(k) in id_to_strain}

subsystem_scores_per_microbe = pd.read_csv("/home/javad/pyprojects/MO_GEMs_Score/chain_results/subsystem_scores_per_microbe.csv")

G = create_graphrag_knowledge_graph(pairwise_data, strain_mean_biomass, microbial_abundance, metabolite_info, subsystem_scores_per_microbe)
save_graph(G)
print(f"Number of nodes in G: {G.number_of_nodes()}")
print(f"Number of edges in G: {G.number_of_edges()}")

count = 0
for u, v, d in G.edges(data=True):
    if d.get('type') == 'involved_in':
        print(f"{u} --({d['type']}, score={d.get('subsystem_score')})--> {v}")
        count += 1
    if count >= 5:
        break

# for u, v, data in G.edges(data=True):
#     print(f"Edge from {u} to {v}: {data}")

convert_nx_to_neo4j(G, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
print("NetworkX graph 'G' has been converted and loaded into Neo4j.")
