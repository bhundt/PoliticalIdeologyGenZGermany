import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


def load_relevant_data(folder: str) -> pd.DataFrame:
    """Load relevant data from data folder.
    The data is stored in two SPSS files, one for the years up to 2018, and one for 2021.
    """
    df_upto2018 = pd.read_spss(os.path.join(folder, "upto2018.sav"), convert_categoricals=False)
    df_2021 = pd.read_spss(os.path.join(folder, "2021.sav"), convert_categoricals=False)

    df_2021["year"] = 2021
    df_upto2018.year = df_upto2018.year.astype(int)

    return pd.concat([df_upto2018, df_2021], ignore_index=True).reset_index(drop=True)


def remove_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns we need for analysis"""
    df = df[["year", "age", "sex", "pa01"]]
    return df


def transform_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Transform sex variable from numerical values to strings."""
    df.replace({"sex": {1: "man", 2: "woman", 3: "diverse"}}, inplace=True)
    return df


def filter_to_gen_z(df: pd.DataFrame, min_age: int = 18, max_age: int = 30) -> pd.DataFrame:
    """Filter dataframe by age to only look at Gen Z"""
    df = df[(df.age >= min_age) & (df.age < max_age)].reset_index(drop=True)
    return df


def transform_ideology(df: pd.DataFrame) -> pd.DataFrame:
    """Transform ideaology variable from numerical values to strings.
    The respondents were asked to rate their political orientation on a scale from 1 to 10.
    1 being "left", 10 being "right".
    """
    df.replace(
        {
            "pa01": {
                1: "Liberal",
                2: "Liberal",
                3: "Liberal",
                4: "Liberal",
                5: "Liberal",
                6: "Conservative",
                7: "Conservative",
                8: "Conservative",
                9: "Conservative",
                10: "Conservative",
            }
        },
        inplace=True,
    )
    return df


def make_diff(df):
    """Compute % liberal minus % conservative.
    Credit to Allen Downey, https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/examples/ideology_gap.ipynb
    """
    year = df["year"]
    column = df["pa01"]

    xtab = pd.crosstab(year, column, normalize="index")
    diff = xtab["Liberal"] - xtab["Conservative"]

    return diff * 100


def make_lowess(series, frac=0.5):
    """Use LOWESS to compute a smooth line.
    Credit to Allen Downey, https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/examples/ideology_gap.ipynb

    series: pd.Series

    returns: pd.Series
    """
    y = series.values
    x = series.index.values

    smooth = lowess(y, x, frac=frac)
    index, data = np.transpose(smooth)

    return pd.Series(data, index=index)


def plot_series_lowess(series, color, plot_series=True, frac=0.5, **options):
    """Plots a series of data points and a smooth line.
    Credit to Allen Downey, https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/examples/ideology_gap.ipynb

    series: pd.Series
    color: string or tuple
    """
    if "label" not in options:
        options["label"] = series.name

    if plot_series or len(series) == 1:
        x = series.index
        y = series.values
        plt.plot(x, y, "o", color=color, alpha=0.3, label="_")

    if not plot_series and len(series) == 1:
        x = series.index
        y = series.values
        plt.plot(x, y, "o", color=color, alpha=0.6, label=options["label"])

    if len(series) > 1:
        smooth = make_lowess(series, frac=frac)
        smooth.plot(color=color, **options)


def decorate(**options):
    """Decorate the current axes.
    Credit to Allen Downey, https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/examples/ideology_gap.ipynb

    Call decorate with keyword arguments like
    decorate(title='Title',
             xlabel='x',
             ylabel='y')

    The keyword arguments can be any of the axis properties
    https://matplotlib.org/api/axes_api.html
    """
    ax = plt.gca()
    ax.set(**options)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)

    plt.tight_layout()


def decorate_plot(title):
    decorate(xlabel="Year", ylabel="% liberal - % conservative", title=title)


def make_plot(df, title=""):
    """Plot % liberal - % conservative for male and female respondents.
    Credit to Allen Downey, https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/examples/ideology_gap.ipynb
    """
    male = df[df.sex == "man"]
    female = df[df.sex == "woman"]

    diff_male = make_diff(male)
    diff_female = make_diff(female)

    plot_series_lowess(diff_male, color="C0", label="Male")
    plot_series_lowess(diff_female, color="C1", label="Female")
    decorate_plot(title)


def savefig(filename, **options):
    if "dpi" not in options:
        options["dpi"] = 300
    plt.savefig(filename, **options)


if __name__ == "__main__":
    df = load_relevant_data("data")
    df = remove_unwanted_columns(df)
    df = transform_sex(df)
    df = filter_to_gen_z(df)
    df = transform_ideology(df)

    make_plot(df, title="Gen Z - Germany")
    savefig("products/ideology_gap1.png")
