import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def format_CSV(pathway): # formats CSV for analysis
    df = pd.read_csv(pathway)

    df_copy = df.copy()

    df_copy["Time Stamp"] = pd.to_datetime(df_copy["Time Stamp"], utc = True)

    df_pivot = df_copy.pivot(index = "Time Stamp", columns = "Zone", \
        values = "LBMP") # creates pivot table for ease of analysis

    df_pivot["SPREAD"] = df_pivot["N.Y.C."] - df_pivot["WEST"] # calculates spread in LBMP between and Zone J (N.Y.C.) and Zone A (West), key variable of interest

    df_pivot["hour"] = df_pivot.index.hour # generates time-based predictor variables
    df_pivot["month"] = df_pivot.index.month
    df_pivot["day_of_week"] = df_pivot.index.dayofweek
    df_pivot["is_weekend"] = (df_pivot["day_of_week"] >= 5).astype(int)

    features = ["hour", "month", "day_of_week", "is_weekend"] # features for predictions
    df_model = df_pivot[features + ["SPREAD"]].dropna() 

    return df_model

def generate_model(dataframe, features): # generates random forest model
    df = dataframe

    X = df[features] # features of interest, inhereted from generate_plots function
    y = df["SPREAD"] # prediction variable

    split = int(len(df) * 0.83) # train/test split, first 10 months to train, last 2 test
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(n_estimators = 100, random_state = 42) # fit model
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test) # evaluate model
    print(f"R²: {r2_score(y_test, y_pred):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} $/MWh")

    return model, y_test, y_pred

def generate_plots(dataframe): # generates plots for analysis
    df = dataframe

    features = ["hour", "month", "day_of_week", "is_weekend"] # features used for predictions

    model, y_test, y_pred = generate_model(df, features) # generates random forest model

    plt.style.use('dark_background') # overarching aesthetic choices
    plt.rcParams['figure.dpi'] = 400
    plt.rcParams["font.family"] = "Avenir"
    bar_fill = "blueviolet"
    line_fill_1 = "cyan"
    line_fill_2 = "r"

    # plot 1 — predicted vs actual spread of LBMP
    plt.figure(figsize = (12, 4))
    test_range = 200 # hours 
    plt.plot(y_test.values[:test_range], label = "Actual", alpha = 0.7, color = line_fill_1)
    plt.plot(y_pred[:test_range], label = "Predicted", alpha = 0.7, color = line_fill_2)
    plt.title("Predicted vs Actual NYC-WEST LBMP Spread, First 200 Hours")
    plt.xlabel("Hours")
    plt.ylabel("$/MWh")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("Plots/plot_1.jpg")
    plt.close()

    # plot 2 — feature importance 
    importances = pd.Series(model.feature_importances_, index = features).sort_values()
    plt.figure(figsize = (12, 8))
    importances.plot(kind = "barh", color = bar_fill)
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.yticks(ticks = range(len(importances)), labels = ["Weekend", "Day of Week", "Hour", "Month"]) 
    plt.tight_layout()
    # plt.show()
    plt.savefig("Plots/plot_2.jpg")
    plt.close()

    # plot - average spread by hour
    hourly = df.groupby("hour")["SPREAD"].mean()
    plt.figure(figsize = (12, 8))
    hourly.plot(kind = "bar", color = bar_fill)
    plt.title("Average NYC-WEST LBMP Spread by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("$/MWh")
    plt.xticks(rotation = 0)
    plt.tight_layout()
    # plt.show()
    plt.savefig("Plots/plot_3.jpg")
    plt.close()

    # Chart 3 - average spread by month
    monthly = df.groupby("month")["SPREAD"].mean()
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", \
        "Sep", "Oct", "Nov", "Dec"]
    plt.figure(figsize = (12, 6))
    monthly.plot(kind = "bar", color = bar_fill)
    plt.title("Average NYC-WEST LBMP Spread by Month")
    plt.xlabel("Month")
    plt.ylabel("$/MWh")
    plt.xticks(ticks = range(12), labels = month_labels, rotation = 0)
    plt.tight_layout()
    # plt.show()
    plt.savefig("Plots/plot_4.jpg")
    plt.close()

    # Chart 4 - single day (reload your original single day file)
    df_day = pd.read_csv("Clean/20260303damlbmp_zone.csv") # random day, unrelated to main model, just for illustrating general spread over course of day
    zones = ["CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL", "LONGIL", \
        "MHK VL", "MILLWD", "N.Y.C.", "NORTH", "WEST"]
    df_day = df_day[df_day["Name"].isin(zones)]
    df_day_pivot = df_day.pivot(index = "Time Stamp", columns = "Name", values = "LBMP ($/MWHr)")
    df_day_pivot["SPREAD"] = df_day_pivot["N.Y.C."] - df_day_pivot["WEST"]
    plt.figure(figsize = (12, 6))
    df_day_pivot["SPREAD"].plot(color = line_fill_1)
    plt.title("NYC vs WEST LBMP Spread - March 3rd 2026")
    plt.xlabel("Hour")
    plt.xticks(rotation = 20)
    plt.ylabel("$/MWh")
    plt.tight_layout()
    # plt.show()
    plt.savefig("Plots/plot_5.jpg")
    plt.close()

def main(): # runs all formatting, analysis, and figure generation processes
    path = "Clean/2025_LBMP_clean.csv"
    df = format_CSV(path)
    generate_plots(df)

if __name__ == '__main__':
    main()