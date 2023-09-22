import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from scipy.stats import pearsonr
from platform import python_version
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List
import itertools
import warnings

warnings.simplefilter("ignore")


def corrfunc(x: List[float], y: List[float], **kws) -> None:
    """Annotate a plot with the Pearson correlation coefficient and p-value.

    Parameters
    ----------
    x : List[float]
        The x-axis data.
    y : List[float]
        The y-axis data.
    **kws : dict
        Additional keyword arguments to pass to the plot function.
    """
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r), xy=(0.1, 0.99), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p), xy=(0.4, 0.99), xycoords=ax.transAxes)


def multicollinearity_diagnose(
    df: pd.DataFrame, target: str, to_return: bool = False, max_vif: int = 5
) -> pd.DataFrame:
    """Detect multicollinearity between explanatory variables in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the explanatory variables and the target variable.
    target : str
        The name of the target variable.
    to_return : bool, optional
        Whether to return a DataFrame with the VIF and tolerance values for each pair of variables, by default False.
    max_vif : int, optional
        The threshold for detecting multicollinearity based on the VIF value, by default 5.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the VIF and tolerance values for each pair of variables, if `to_return` is True.

    """
    # select all columns except target
    explanatory_variables = [col for col in df.columns if col != target]

    df_multicolinearity = pd.DataFrame()

    # Iterate over each pair of columns without repetition
    for col1, col2 in itertools.combinations(explanatory_variables, 2):
        X = df[[col1, col2]]
        X = sm.add_constant(X)

        vif = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=X.columns,
        )

        tolerance = 1 / vif

        df_aux = pd.concat([vif, tolerance], axis=1, keys=["VIF", "Tolerance"])

        df_aux = df_aux.iloc[2:, :]
        df_aux = df_aux.rename(index={col2: col1 + "-" + col2})
        df_multicolinearity = pd.concat([df_multicolinearity, df_aux])

    for index, row in df_multicolinearity.iterrows():
        if row["VIF"] >= max_vif:
            print(
                f"Multicollinearity detected for variables: {index}. VIF: {row['VIF']:.1f}. VIF threshold: {max_vif}. Tolerance: {row['Tolerance']:0.3f}"
            )

    if to_return:
        return df_multicolinearity


def breusch_pagan_test(
    model: sm.regression.linear_model.RegressionResultsWrapper, alpha: float = 0.05
) -> tuple:
    """
    Performs the Breusch-Pagan test for heteroscedasticity.

    The Breusch-Pagan test checks if there is correlation between the residuals and one or more features. The null hypothesis
    is that the residuals are homoscedastic, while the alternative hypothesis is that the residuals are heteroscedastic,
    indicating the omission of a relevant variable.

    Breusch-Pagan Test Hypothesis:
    * H0: Absence of heterocedasticity, that is the residuals present homocedasticity.
    * H1: Presence of heterocedasticity in the residuals, that is, there is correlation between the residuals and one of more features. This indicates the omission of a relevant variable.

    Parameters
    ----------
    model : sm.regression.linear_model.RegressionResultsWrapper
        The fitted regression model.
    alpha: float, optional
        Significance level, by default 0.05.

    Returns
    -------
    tuple
        A tuple containing the test statistic and the p-value.

    """

    df = pd.DataFrame({"yhat": model.fittedvalues, "resid": model.resid})

    df["up"] = (np.square(df.resid)) / np.sum(((np.square(df.resid)) / df.shape[0]))

    modelo_aux = sm.OLS.from_formula("up ~ yhat", df).fit()

    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)

    anova_table["sum_sq"] = anova_table["sum_sq"] / 2

    chisq = anova_table["sum_sq"].iloc[0]

    p_value = stats.chi2.pdf(chisq, 1) * 2

    print(f"chisq: {chisq}")

    print(f"p-value: {p_value}")

    if p_value < 0.05:
        print("Reject-se H0: There is heteroscedasticity")
    else:
        print("Does not reject H0: There is no heteroscedasticity")
    return chisq, p_value


def inverse_boxcox_transformation(ybc: float, lmbda: float) -> float:
    """Transforms a variable y that has been Box-Cox transformed with parameter lambda
    back to its original scale.

    Parameters
    ----------
    ybc : float
        The Box-Cox transformed variable.
    lmbda : float
        The lambda parameter used in the Box-Cox transformation.

    Returns
    -------
    float
        The variable y transformed back to its original scale.
    """
    y = ((ybc * lmbda) + 1) ** (1 / lmbda)
    return y
