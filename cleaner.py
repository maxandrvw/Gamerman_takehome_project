import numpy as np
import pandas as pd
import os

def clean_group_dates(groups): # converts datetime objects to pandas datetime, returns a list of data frames with fixed datetime objects
    result = list()

    for group_name, group_df in groups: # cycles through zone groups, prevents issues when localizing for timezones due to identical date-time objects when ungrouped
        group_df["Time Stamp"] = pd.to_datetime(group_df['Time Stamp'], format='%m/%d/%Y %H:%M')
        group_df["Time Stamp"] = group_df["Time Stamp"].dt.tz_localize('America/New_York', ambiguous='infer') # localizes to New York timezone, grouping + ambigous = 'infer' handles daylight savings
        
        result.append(group_df)
    
    return result

def combine_CSV(folder): # combines CSV files in folder with identical column names, additionally cleans datetime objects with clean_group_dates, returns combined dataframe
    frames = list()

    for file in os.listdir(folder): # cycles through CSV files

        if file == ".DS_Store": # hidden file case
            continue

        pathway = folder + "/" + file
        temp = pd.read_csv(pathway) # imports CSV as Pandas dataframe

        groups = temp.groupby("Name")
        clean_groups = clean_group_dates(groups) # cleaning the dates is handled here instead of in clean_columns to prevent issues when localizing for timezones

        frames = frames + clean_groups

    composite = pd.concat(frames, ignore_index = True) # combines dataframes
    composite = composite.sort_values("Time Stamp") # resets indexing

    return composite

def data_cleaner(dataframe): # cleans dataframe, adds calculated variables of interest doesn't handle cleaning datetime objects, returns cleaned + pivoted dataset
    df_copy = dataframe.copy()

    df_copy = df_copy.rename(columns = { # edits column names
        "Time Stamp"                            : "Time Stamp",
        "Name"                                  : "Zone",
        "PTID"                                  : "PTID",
        "LBMP ($/MWHr)"                         : "LBMP",
        "Marginal Cost Losses ($/MWHr)"         : "Marginal Cost Losses",
        "Marginal Cost Congestion ($/MWHr)"     : "Marginal Cost Congestion"
        })

    # df_copy["Time Stamp"] = pd.to_datetime(df_copy["Time Stamp"], errors='coerce') # edits datatypes, only 'Name'/'Zone' differ from required format, thus the rest are commented out
    df_copy["Zone"] = df_copy["Zone"].astype("category")
    # df_copy["PTID"] = df_copy["PTID"].astype("int64")
    # df_copy["LBMP"] = df_copy["LBMP"].astype("float64")
    # df_copy["Marginal Cost Losses"] = df_copy["Marginal Cost Losses"].astype("float64")
    # df_copy["Marginal Cost Congestion"] = df_copy["Marginal Cost Congestion"].astype("float64")

    zones = [ # list for NY zones
    "CAPITL", 
    "CENTRL", 
    "DUNWOD", 
    "GENESE", 
    "HUD VL", 
    "LONGIL", 
    "MHK VL", 
    "MILLWD", 
    "N.Y.C.", 
    "NORTH", 
    "WEST"
    ]

    df_filtered = df_copy[df_copy["Zone"].isin(zones)] # filters out Zones not in NY

    df_filtered = df_filtered.drop(columns = "PTID") # drops PTID column, not relevant to analysis, can be infered from 'Name'/'Zone'

    df_filtered = df_filtered.drop_duplicates().dropna() # drops duplicate rows and rows with NA values

    df_filtered = df_filtered.sort_values(by = "Time Stamp")

    return df_filtered

def main(): # runs all cleaning processes, saves data as csv to specified location
    path = "2025 Monthly LBMP"
    save_location = "Clean/2025_LBMP_clean.csv"
    df = combine_CSV(path)
    df_clean = data_cleaner(df)
    df_clean.to_csv(save_location, index = False)

if __name__ == '__main__':
    main()