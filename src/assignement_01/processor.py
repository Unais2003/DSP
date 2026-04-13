import pandas as pd

"""
df = pd.read_csv("../../data/assignement_01/raw/Bahrain_time_series.csv")
filtered_driver_df = df.loc[df["Driver"] == "VER"]
filtered_driver_df.to_csv(
    "../../data/assignement_01/processed/filtered_VER.csv",
    index=False,
)

telem_vars_df = filtered_driver_df[
    ["Driver", "RPM", "nGear", "Speed", "LapNumber", "Time"]
]

telem_vars_df.to_csv(
    "../../data/assignement_01/processed/selected_vars_VER.csv",
    index=False,
)
"""

features_df = pd.read_csv("../../data/assignement_01/processed/selected_vars_VER.csv")
print(features_df)

"""
df = pd.read_csv("../../data/assignement_01/processed/filtered_HAM.csv")

df_raw = pd.read_csv("../../data/assignement_01/raw/Bahrain_time_series.csv")

print(df_raw.groupby("Driver")["LapNumber"].unique())

print(df_raw.groupby("Driver")["Driver"].count())

print(df_raw.describe()["LapNumber"])

print(df_raw[df_raw["LapNumber"] == 1]["LapNumber"].count())  # 797795
print(df_raw[df_raw["LapNumber"] != 1]["LapNumber"].count())  # 40932

"""
# print(df[["Driver", "RPM", "Speed", "nGear", "LapNumber"]])
