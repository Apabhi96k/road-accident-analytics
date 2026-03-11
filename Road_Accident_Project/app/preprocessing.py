import pandas as pd
import os

def load_data():

    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "india_road_accidents_2005_2026.csv"
    )

    df = pd.read_csv(path)

    return df