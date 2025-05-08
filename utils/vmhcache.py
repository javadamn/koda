import requests
import csv
import os


class VMHCacheClient:
    def __init__(
        self,
        cache_file="/Users/mohsen/Documents/vs projects/community_graphRAG/data/vmh_metabolite_cache.csv",
    ):
        self.cache_file = cache_file
        self.api_url = "https://www.vmh.life/_api/metabolites/?abbreviation="
        self.cache = {}
        self.session = requests.Session()
        self._load_cache()

    def _load_cache(self):
        """Load the local cache from a CSV file."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.cache[row["reaction"]] = row

    def _save_cache(self):
        """Save the current cache to a CSV file."""
        with open(self.cache_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "reaction",
                    "met_abbreviation",
                    "name",
                    "description",
                    "seed_id",
                    "kegg_id",
                    "pubchem_id",
                ],
            )
            writer.writeheader()
            for row in self.cache.values():
                writer.writerow(row)

    def _extract_met_abbr(self, rx_id):
        """Get the metabolite abbreviation from an exchange reaction ID."""
        if rx_id.endswith("(e)"):
            rx_id = rx_id[:-3]
        return rx_id[3:] if rx_id.startswith("EX_") else None

    def _fetch_met_data(self, met_abbr):
        try:
            url = self.api_url + met_abbr

            response = self.session.get(url, timeout=5)
            response.raise_for_status()

            data = response.json()

            if isinstance(data, dict) and "results" in data and data["results"]:
                return data["results"][0]
            else:
                return None

        except Exception as e:
            return {"error": str(e)}

    def get_met_info(self, rxnList):
        """Get info for a list of exchange reactions, using cache if possible."""
        updated = False
        results = []

        for rx in rxnList:
            if rx in self.cache:
                results.append(self.cache[rx])
                continue

            met_abbr = self._extract_met_abbr(rx)
            if not met_abbr:
                self.cache[rx] = {
                    "reaction": rx,
                    "met_abbreviation": "",
                    "name": "",
                    "description": "",
                    "seed_id": "",
                    "kegg_id": "",
                    "pubchem_id": "",
                }
            else:
                data = self._fetch_met_data(met_abbr)
                if not data or "error" in data:
                    print(data)
                    self.cache[rx] = {
                        "reaction": rx,
                        "met_abbreviation": met_abbr,
                        "name": "",
                        "description": "",
                        "seed_id": "",
                        "kegg_id": "",
                        "pubchem_id": "",
                    }
                else:
                    self.cache[rx] = {
                        "reaction": rx,
                        "met_abbreviation": met_abbr,
                        "name": data.get("fullName", ""),
                        "description": data.get("description", ""),
                        "seed_id": data.get("seed", ""),
                        "kegg_id": data.get("keggId", ""),
                        "pubchem_id": data.get("pubChemId", ""),
                    }
            updated = True
            results.append(self.cache[rx])

        if updated:
            self._save_cache()

        return results
