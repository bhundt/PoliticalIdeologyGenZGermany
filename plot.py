import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


def _read_2021(folder: str) -> pd.DataFrame:
    df = pd.read_spss(os.path.join(folder, "2021.sav"), convert_categoricals=False)
    df["year"] = 2021
    return df


def _read_upto2018(folder: str) -> pd.DataFrame:
    df = pd.read_spss(os.path.join(folder, "upto2018.sav"), convert_categoricals=False)
    df.year = df.year.astype(int)
    return df


def load_relevant_data(folder: str) -> pd.DataFrame:
    """Load relevant data from data folder.
    The data is stored in two SPSS files, one for the years up to 2018, and one for 2021.
    """
    df_2021 = _read_2021(folder)
    df_upto2018 = _read_upto2018(folder)

    df = pd.concat([df_upto2018, df_2021], ignore_index=True).reset_index(drop=True)
    return df


def remove_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns we need for analysis
    pa01: political orientation on a scale from 1 to 10
    wghtptew: weighting factor which needs to be applied to the data to make it representative between east and west Germany
    """
    df = df[["year", "age", "sex", "pa01", "wghtpew"]]
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


def resample_rows_weighted(df, column):
    """Resamples a DataFrame using probabilities proportional to given column.
    Credit to Allen Downey, https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/examples/ideology_gap.ipynb

    df: DataFrame
    column: string column name to use as weights

    returns: DataFrame
    """
    weights = df[column]
    sample = df.sample(n=len(df), replace=True, weights=weights)
    return sample


def resample_by_year(df, column):
    """Resample rows within each year.
    Credit to Allen Downey, https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/examples/ideology_gap.ipynb

    df: DataFrame
    column: string name of weight variable

    returns DataFrame
    """
    grouped = df.groupby("year")
    samples = [resample_rows_weighted(group, column) for _, group in grouped]
    sample = pd.concat(samples, ignore_index=True)
    return sample


def percentile_rows(series_seq, ps):
    """Computes percentiles from aligned series.
    Credit to Allen Downey, https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/examples/ideology_gap.ipynb

    series_seq: list of sequences
    ps: cumulative probabilities

    returns: Series of x-values, NumPy array with selected rows
    """
    df = pd.concat(series_seq, axis=1).dropna()
    xs = df.index
    array = df.values.transpose()
    array = np.sort(array, axis=0)
    nrows, _ = array.shape

    ps = np.asarray(ps)
    indices = (ps * nrows).astype(int)
    rows = array[indices]
    return xs, rows


def plot_percentiles(series_seq, ps=None, label=None, **options):
    """Plot the low, median, and high percentiles.
    Credit to Allen Downey, https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/examples/ideology_gap.ipynb

    series_seq: sequence of Series
    ps: percentiles to use for low, medium and high
    label: string label for the median line
    options: options passed plt.plot and plt.fill_between
    """
    if ps is None:
        ps = [0.05, 0.5, 0.95]
    assert len(ps) == 3

    xs, rows = percentile_rows(series_seq, ps)
    low, med, high = rows
    plt.plot(xs, med, alpha=0.5, label=label, **options)
    plt.fill_between(xs, low, high, linewidth=0, alpha=0.2, **options)


def resample_diffs(df, query, iters=101):
    """
    Credit to Allen Downey, https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/examples/ideology_gap.ipynb
    """
    diffs = []
    for i in range(iters):
        sample = resample_by_year(df, "wghtpew").query(query)
        diff = make_diff(sample)
        diffs.append(diff)
    return diffs


if __name__ == "__main__":
    df = load_relevant_data("data")
    # TODO: fix nan in pa01
    print(df["wghtpew"])
    df = remove_unwanted_columns(df)
    df = transform_sex(df)
    df = filter_to_gen_z(df)
    df = transform_ideology(df)
    print(df["wghtpew"])

    # this is  simple, non weighted analysis
    make_plot(df, title="Age < 30 - Germany")
    savefig("products/ideology_gap_unweighted.png")

    # this is the weighted version
    male = df.query('sex=="man"')
    female = df.query('sex=="woman"')

    diff_male = make_diff(male)
    diff_female = make_diff(female)

    diffs_male = resample_diffs(df, 'sex=="man"')
    diffs_female = resample_diffs(df, 'sex=="woman"')

    plot_percentiles(diffs_male)
    plot_percentiles(diffs_female)

    diff_male.plot(style=".", color="C0", label="Male")
    diff_female.plot(style=".", color="C1", label="Female")

    decorate_plot("Age < 30 with sampling weights")
    savefig("products/ideology_gap_weighted.png")
