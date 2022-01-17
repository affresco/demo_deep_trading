import numpy as np
import scipy.stats as scipy_stats


# ####################################################################
# BLACK 1976 PRICING MODEL: D1 AND D2
# ####################################################################

def compute_d1(forward: np.ndarray, strike: np.ndarray, volatility: np.ndarray, ttm: np.ndarray, df: np.ndarray):
    num = np.log(np.divide(forward, strike)) + 0.5 * np.multiply(np.power(volatility, 2), ttm)
    den = np.multiply(volatility, np.sqrt(ttm))
    return np.multiply(np.divide(num, den), df)


def compute_d2(forward: np.ndarray, strike: np.ndarray, volatility: np.ndarray, ttm: np.ndarray, df: np.ndarray):
    num = np.log(np.divide(forward, strike)) - 0.5 * np.multiply(np.power(volatility, 2), ttm)
    den = np.multiply(volatility, np.sqrt(ttm))
    return np.multiply(np.divide(num, den), df)


def compute_d1_and_d2(forward: np.ndarray, strike: np.ndarray, volatility: np.ndarray, ttm: np.ndarray, df: np.ndarray):
    d1_ = compute_d1(forward=forward, strike=strike, volatility=volatility, ttm=ttm, df=df)
    return d1_, d1_ - np.multiply(volatility, np.sqrt(ttm))


# ####################################################################
# BLACK 1976 PRICING MODEL: PRICE
# ####################################################################

def price(option_type: float, forward: np.ndarray, strike: np.ndarray, df: np.ndarray, d1: np.ndarray, d2: np.ndarray):
    term_s = option_type * forward * scipy_stats.norm.cdf(np.multiply(option_type, d1))
    term_k = option_type * strike * scipy_stats.norm.cdf(np.multiply(option_type, d2))
    return np.multiply(df, (term_s - term_k))


def price_delta_proba(option_type: float, forward: np.ndarray, strike: np.ndarray, df: np.ndarray, d1: np.ndarray,
                      d2: np.ndarray):
    cdf_d1 = scipy_stats.norm.cdf(np.multiply(option_type, d1))
    cdf_d2 = scipy_stats.norm.cdf(np.multiply(option_type, d2))

    term_s = option_type * forward * cdf_d1
    term_k = option_type * strike * cdf_d2

    p = np.multiply(df, (term_s - term_k))
    d = option_type * cdf_d1

    return p, d, cdf_d2


def delta(option_sign: float, d1: np.ndarray):
    return option_sign * scipy_stats.norm.cdf(option_sign * d1)


# ####################################################################
# BLACK 1976 PRICING MODEL: GREEKS
# ####################################################################


def vega(strike: np.ndarray, ttm, d2: np.ndarray, df: np.ndarray):
    pdf_d2 = scipy_stats.norm.pdf(d2)
    return strike * np.multiply(np.multiply(df, pdf_d2), np.sqrt(ttm))


def gamma(forward: np.ndarray, strike: np.ndarray, volatility, ttm, df_backward=1.0, d2=None, pdf_d2=None,
          dividend_yield=0.0, ):
    pdf_d2 = scipy_stats.norm.pdf(d2)
    den_ = volatility * np.sqrt(ttm) * df_backward * forward ** 2
    return (strike * pdf_d2 * np.exp(-2. * ttm * dividend_yield)) / den_


def vega_and_gamma(forward: np.ndarray, strike: np.ndarray, volatility: np.ndarray, ttm: np.ndarray, d2: np.ndarray,
                   df: np.ndarray):
    pdf_d2 = scipy_stats.norm.pdf(d2)
    vega = strike * np.multiply(np.multiply(df, pdf_d2), np.sqrt(ttm))

    den_ = volatility * np.sqrt(ttm) * df * forward ** 2
    gamma = (strike * pdf_d2 * np.exp(-2. * ttm)) / den_

    return vega, gamma


# ####################################################################
# NOT COMPUTED FOR NOW
# ####################################################################

def rho(option_sign, forward: np.ndarray, strike: np.ndarray, volatility, ttm, d2=None, cdf_d2=None, df_backward=1.0,
        **kwargs):
    d2 = d2 or compute_d2(forward=forward, strike=strike, volatility=volatility, ttm=ttm)
    if cdf_d2 is None:
        cdf_d2 = scipy_stats.norm.cdf(option_sign * d2)
    return option_sign * strike * ttm * df_backward * cdf_d2


def theta(option_sign, forward: np.ndarray, strike: np.ndarray, volatility, ttm,
          d1=None, pdf_d1=None, cdf_d1=None, d2=None, pdf_d2=None, cdf_d2=None,
          df_backward=1.0, df_div_repo=1.0, **kwargs):
    int_rate = (-1. / ttm) * np.log(df_backward)
    div_repo_rate = (-1. / ttm) * np.log(df_div_repo)

    d1 = d1 or compute_d1(forward=forward, strike=strike, volatility=volatility, ttm=ttm)
    d2 = d2 or compute_d2(forward=forward, strike=strike, volatility=volatility, ttm=ttm)

    if pdf_d1 is None:
        pdf_d1 = scipy_stats.norm.pdf(d1)  # --> same sign for puts and calls

    if cdf_d1 is None:
        cdf_d1 = scipy_stats.norm.cdf(option_sign * d1)

    if cdf_d2 is None:
        cdf_d2 = scipy_stats.norm.cdf(option_sign * d2)

    theta_t1 = -0.5 * forward * df_backward * pdf_d1 * volatility / np.sqrt(ttm)
    theta_t2 = -option_sign * int_rate * strike * df_backward * cdf_d2
    theta_t3 = option_sign * div_repo_rate * forward * df_backward * cdf_d1

    return theta_t1 + theta_t2 + theta_t3
