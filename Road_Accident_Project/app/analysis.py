def accident_stats(df):

    total_accidents = df["Accidents"].sum()

    total_casualties = df["Casualties"].sum()

    avg_casualties = df["Casualties"].mean()

    return total_accidents, total_casualties, avg_casualties