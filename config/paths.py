from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_PATH / "data"
AGORA_PATH = ROOT_PATH / "AGORA"
GENRA_PATH = AGORA_PATH / "Genra_GEMs"

STRAINS_CSV = DATA_PATH / "anaerobic_strains.csv"
ABUNDANCE_CSV = DATA_PATH / "SamplesSpeciesRelativeAbundance_modified_final.csv"
MODELS_CSV = DATA_PATH / "ListofModelsforSpecies.csv"
PAIRWISE_PATH = DATA_PATH / "pairwise_results.csv"
