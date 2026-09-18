from pathlib import Path

import pandas as pd
import requests

DEFAULT_DATA_URL = "https://breathecode.herokuapp.com/asset/internal-link?id=439&path=housing.csv"
FEATURE_COLUMNS = ("Latitude", "Longitude", "MedInc")
import pandas as pd

# load the .env file variables
def load_housing_data(path: Path, url: str = DEFAULT_DATA_URL) -> pd.DataFrame:
    """Load the housing CSV locally, downloading it once when absent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)

    data = pd.read_csv(path)
    missing = sorted(set(FEATURE_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    selected = data.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
    if len(selected) < 6:
        raise ValueError("At least six complete rows are required to create six clusters.")
    return selected.reset_index(drop=True)
