import pandas as pd

### Just a sample function below
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a excel file into a pandas DataFrame."""
    return pd.read_excel(file_path)