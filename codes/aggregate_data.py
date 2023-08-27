################################################################################
# 0. Setup
################################################################################
import numpy as np
import json
import pandas as pd
import geopandas as gpd

################################################################################
# 1. Importing Data
################################################################################

# Import Municipalities Geo-Spatial Data (shapefile)
df_geo = gpd.read_file("data/raw/Com01012022_g_WGS84.shx")

# Import Municipalities Electoral Data
df_turnout = pd.read_csv(
    "https://raw.githubusercontent.com/ondata/elezioni-politiche-2022/main/affluenza-risultati/dati/affluenza/affluenzaComuni.csv"
)
url_camera = "https://raw.githubusercontent.com/ondata/elezioni-politiche-2022/main/affluenza-risultati/dati/risultati/camera-italia-comune.csv"
df = pd.read_csv(url_camera)

# Import Municipalities Metadata
url_municipalities = "https://raw.githubusercontent.com/ondata/elezioni-politiche-2022/main/affluenza-risultati/risorse/anagraficaComuni.csv"
df_municipalities = pd.read_csv(url_municipalities)

# Import json file with national final results for comparison
with open("data/raw/national_results.json") as json_file:
    dict_camera_national = json.load(json_file)
df_national = pd.DataFrame(list(dict_camera_national["camera"].items()))
df_national.columns = ["lista", "perc_nazionale"]

################################################################################
# 2. Aggregating, Cleaning and Manipulating Data
################################################################################

# Fill missing percentage values
# The assumption is that they are missing because no votes were received
# (checked on Lucca province, seems plausible)
df["perc"].fillna(0, inplace=True)

# Format electoral codes
df["codice_red"] = df["codice"].apply(lambda x: x.replace("-", "")[-7:])
df_municipalities["CODICE ELETTORALE"] = df_municipalities["CODICE ELETTORALE"].astype(
    str
)
df_municipalities["codice_red"] = df_municipalities["CODICE ELETTORALE"].apply(
    lambda x: x.replace("-", "")[-7:]
)

# Note that large municipalities (like Milan and Rome) are split into multiple electoral codes
# print(df.groupby(["codice_red"])["codice"].nunique().sort_values().tail(10))

# Filter data on Lucca province
df_lucca = df_municipalities.query('SIGLA=="LU"').copy()
# Merge municipalities electoral data with metadata
final_df_camera = df.merge(
    df_lucca, left_on="codice_red", right_on="codice_red", how="inner"
)
# Sort by municipality name
final_df_camera.sort_values(by=["DESCRIZIONE COMUNE"], inplace=True)

# Sanity check 1: 33 municipalities in Lucca province
print(
    "SANITY CHECK 1: Number of Municipalities in Lucca Province:",
    len(final_df_camera.groupby(["DESCRIZIONE COMUNE"])),
)
# Sanity check 2: do percentages sum to 100% at the municipality level?
# Some municipalities have more/less 100% of votes, but it's because of rounding errors
print(
    "SANITY CHECK 2: Cumulative pctgs at the Municipalities Level - min:{}, max: {}".format(
        final_df_camera.groupby(["DESCRIZIONE COMUNE", "cogn", "nome"])["perc"]
        .sum()
        .groupby("DESCRIZIONE COMUNE")
        .sum()
        .min(),
        final_df_camera.groupby(["DESCRIZIONE COMUNE", "cogn", "nome"])["perc"]
        .sum()
        .groupby("DESCRIZIONE COMUNE")
        .sum()
        .max(),
    )
)

final_df_camera["perc"] = np.where(
    final_df_camera["voti"] == 0, 0, final_df_camera["perc"]
)

# Aggregate parties into coalitions
csx = [
    "PARTITO DEMOCRATICO - ITALIA DEMOCRATICA E PROGRESSISTA",
    "IMPEGNO CIVICO LUIGI DI MAIO - CENTRO DEMOCRATICO",
    "+EUROPA",
    "ALLEANZA VERDI E SINISTRA",
]

cdx = [
    "NOI MODERATI/LUPI - TOTI - BRUGNARO - UDC",
    "FRATELLI D'ITALIA CON GIORGIA MELONI",
    "FORZA ITALIA",
    "LEGA PER SALVINI PREMIER",
]

m5s = ["MOVIMENTO 5 STELLE"]
aziv = ["AZIONE - ITALIA VIVA - CALENDA"]

final_df_camera["coalizione"] = np.where(
    final_df_camera["desc_lis"].isin(csx),
    "CSX",
    np.where(
        final_df_camera["desc_lis"].isin(cdx),
        "CDX",
        np.where(
            final_df_camera["desc_lis"].isin(m5s),
            "M5S",
            np.where(final_df_camera["desc_lis"].isin(aziv), "AZIV", "OTHERS"),
        ),
    ),
)

# Compare local results with national results\
final_df_camera = final_df_camera.merge(
    df_national, left_on="desc_lis", right_on="lista", how="left"
)
final_df_camera["diff"] = final_df_camera["perc"] - final_df_camera["perc_nazionale"]

# Get the winning coalition for each municipality
df_winner = final_df_camera.loc[
    final_df_camera.groupby(["DESCRIZIONE COMUNE"])["perc"].idxmax(),
    ["perc", "desc_lis", "CODICE ISTAT", "coalizione"],
].copy()

# Get results at the collation level
final_df_camera_coalizioni = (
    final_df_camera.groupby(["DESCRIZIONE COMUNE", "CODICE ISTAT", "coalizione"])[
        ["perc", "perc_nazionale"]
    ]
    .agg({"perc": "sum", "perc_nazionale": "sum"})
    .reset_index()
)
final_df_camera_coalizioni["diff"] = (
    final_df_camera_coalizioni["perc"] - final_df_camera_coalizioni["perc_nazionale"]
)

# Reshape data from long to wide
final_df_wide = pd.pivot(
    final_df_camera_coalizioni,
    index=["DESCRIZIONE COMUNE", "CODICE ISTAT"],
    columns="coalizione",
    values=["perc", "perc_nazionale", "diff"],
)

# Re-arange the new columns in the correct order
final_df_wide.reset_index(drop=False, inplace=True)
final_df_wide.columns = [
    "comune",
    "codice_istat",
    "AZIV",
    "CDX",
    "CSX",
    "M5S",
    "OTHERS",
    "AZIV_nazionale",
    "CDX_nazionale",
    "CSX_nazionale",
    "M5S_nazionale",
    "OTHERS_nazionale",
    "AZIV_delta",
    "CDX_delta",
    "CSX_delta",
    "M5S_delta",
    "OTHERS_delta",
]

# Merge electoral and geo-spatial data
final_df_wide_geo = final_df_wide.merge(
    df_geo, left_on="codice_istat", right_on="PRO_COM", how="inner"
)

# Add info on winning coalition
final_df_wide_geo = final_df_wide_geo.merge(
    df_winner, left_on="codice_istat", right_on="CODICE ISTAT", how="inner"
)

# Add info on turnout
final_df_wide_geo = final_df_wide_geo.merge(
    df_turnout[["elettori", "%h23_prec", "CODICE ISTAT"]],
    left_on=["codice_istat"],
    right_on=["CODICE ISTAT"],
    how="left",
)

# Assign ids to municipalities
final_df_wide_geo['id'] = final_df_wide_geo.groupby('comune').ngroup() +1
final_df_wide_geo['id'] = final_df_wide_geo['id'].astype('str')

# Shorten names for plotting
final_df_wide_geo['comune'] = np.where(final_df_wide_geo['comune'] == 'CASTELNUOVO DI GARFAGNANA', 'CASTELNUOVO DI G. NA', final_df_wide_geo['comune'])
final_df_wide_geo['comune'] = np.where(final_df_wide_geo['comune'] == 'CASTIGLIONE DI GARFAGNANA', 'CASTIGLIONE DI G. NA', final_df_wide_geo['comune'])
final_df_wide_geo['comune'] = np.where(final_df_wide_geo['comune'] == 'SAN ROMANO IN GARFAGNANA', 'SAN ROMANO IN G. NA', final_df_wide_geo['comune'])

# Create column with name and id for plotting
final_df_wide_geo['id_name'] = final_df_wide_geo['comune']+ ' (' + final_df_wide_geo['id'] + ')'

# Format the winning coalition column
final_df_wide_geo['desc_lis'].replace("FRATELLI D'ITALIA CON GIORGIA MELONI", 'FDI', inplace=True)

################################################################################
# 3. Data Export
################################################################################

df_export = gpd.GeoDataFrame(final_df_wide_geo)
df_export.to_file('data/processed/final_lucca_camera.shp')  

# # Export data in CSV format
# final_df_wide_geo.to_csv("data/processed/final_lucca_camera.csv", index=False)

print("Data Successfully Processed and Exported!")
