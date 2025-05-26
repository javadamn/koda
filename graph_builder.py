import networkx as nx
import os
import pandas as pd
import logging
from neo4j import GraphDatabase
from configs.paths import (PAIRWISE_PATH, STRAINS_CSV, ABUNDANCE_CSV, MODELS_CSV)
from utils.data_loader import read_pairwise_data
from utils.abundance_mapper import get_abundance
from utils.vmhcache import VMHCacheClient
from utils.id_mapper import build_id_to_strain_map, replace_ids_with_names
from utils.save_load_graph import save_graph, load_graph

logger = logging.getLogger(__name__)

def create_graphrag_knowledge_graph(pairwise_data, strain_mean_biomass,
                                    microbial_abundance, metabolite_info, subsystem_scores_per_microbe, gene_df):
    
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

    #microbe-KEGG orthology relationships
    for _, row in gene_df.iterrows():
        microbe = row['model']
        kegg_orthology = row['kegg_orthology']
        # print(kegg_orthology)
        if pd.isna(kegg_orthology) or pd.isna(microbe):
            logger.warning(f"Skipping row with missing KO or microbe: {row}")
            continue
            
        if kegg_orthology not in G:
            G.add_node(kegg_orthology, type='kegg_orthology', name=kegg_orthology)
        if not G.has_edge(microbe, kegg_orthology):
            G.add_edge(microbe, kegg_orthology, type='has_kegg_orthology', 
                        description=f'{microbe} has an essential gene with kegg orthology (KO) of {kegg_orthology} and the KO description is {row.get("description")}')


    return G


def convert_nx_to_neo4j(G: nx.DiGraph, neo4j_uri: str, neo4j_user: str,
                        neo4j_password: str):
    """
    Converts a NetworkX DiGraph to a Neo4j graph database.

    Args:
        G: The NetworkX DiGraph to convert.
        neo4j_uri: The URI of the Neo4j database (e.g., "neo4j://localhost:7687").
        neo4j_user: The Neo4j username.
        neo4j_password: The Neo4j password.
    """

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def create_node(tx, node_id: str, labels: list[str], properties: dict):
        """
        Creates or updates a node in Neo4j.

        Args:
            tx: The Neo4j transaction.
            node_id: The unique identifier of the node.
            labels: A list of Neo4j labels for the node.
            properties: A dictionary of node properties.
        """
        query = (
            f"MERGE (n:{':'.join(labels)} {{name: $name}}) "
            "SET n += $props"
        )
        tx.run(query, name=node_id, props=properties)

    def create_relationship(tx, source_name: str, source_labels: list[str],
                            target_name: str, target_labels: list[str],
                            rel_type: str, rel_props: dict):
        """
        Creates a relationship between two nodes in Neo4j.

        Args:
            tx: The Neo4j transaction.
            source_name: The name of the source node.
            source_labels: A list of labels for the source node.
            target_name: The name of the target node.
            target_labels: A list of labels for the target node.
            rel_type: The type of the relationship.
            rel_props: A dictionary of relationship properties.
        """
        query = (
            f"MATCH (a:{':'.join(source_labels)} {{name: $source_name}}), "
            f"(b:{':'.join(target_labels)} {{name: $target_name}}) "
            f"CREATE (a)-[r:{rel_type}]->(b) "
            "SET r += $props"
        )
        tx.run(query, source_name=source_name, target_name=target_name,
               props=rel_props)

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

        for node_id, node_data in G.nodes(data=True):
            if 'type' not in node_data:
                logger.warning(f"Skipping node {node_id}: Missing 'type' attribute")
                continue

            labels = []
            properties = {}

            node_type = node_data['type']
            if node_type == 'microbe':
                labels = ['Microbe']
                properties = {'name': node_id,
                              'abundance': node_data.get('abundance')}
            elif node_type == 'metabolite':
                labels = ['Metabolite']
                properties = {'name': node_data.get('name', node_id)}
            elif node_type == 'pathway':
                labels = ['Pathway']
                properties = {'name': node_data.get('name', node_id)}
            elif node_type == 'kegg_orthology': 
                labels = ['KO']
                properties = {'name': node_data.get('name', node_id)}
            else:
                logger.warning(
                    f"Skipping node {node_id}: Unknown type '{node_type}'")
                continue

            # Remove None values from properties
            properties = {k: v for k, v in properties.items() if v is not None}

            # node in Neo4j
            session.execute_write(create_node, node_id, labels, properties)

        # relationships
        for u, v, data in G.edges(data=True):
            if 'type' not in data:
                logger.warning(f"Skipping edge {u}->{v}: Missing 'type' attribute")
                continue

            u_data = G.nodes[u]
            v_data = G.nodes[v]

            source_labels = _get_node_labels(u_data)  
            target_labels = _get_node_labels(v_data)  

            rel_type = data['type'].upper().replace(' ', '_').replace('-', '_')
            rel_props = {k: v for k, v in data.items() if
                         k != 'type' and v is not None}

            session.execute_write(create_relationship, u, source_labels, v,
                                 target_labels, rel_type, rel_props)

    driver.close()


def _get_node_labels(node_data: dict) -> list[str]:
    """Helper function to get Neo4j labels for a node."""
    node_type = node_data.get('type')
    if node_type == 'microbe':
        return ['Microbe']
    elif node_type == 'metabolite':
        return ['Metabolite']
    elif node_type == 'pathway':
        return ['Pathway']
    elif node_type == 'kegg_orthology':
        return ['KO']
    else:
        return [] 

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "here_is_password") ## the pasword

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
gene_df = pd.read_csv('/home/javad/pyprojects/MO_GEMs_Score/GraphRAG/geneKnockout/gene_df_info.csv')

G = create_graphrag_knowledge_graph(pairwise_data, strain_mean_biomass, microbial_abundance,
                                     metabolite_info, subsystem_scores_per_microbe, gene_df)
print('Graph is built....................')
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
print('Now converting to Neo4j..............')
convert_nx_to_neo4j(G, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
print("NetworkX graph 'G' has been converted and loaded into Neo4j.")
