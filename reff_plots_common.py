from datetime import datetime
from scipy.optimize import curve_fit
from scipy.signal import convolve
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

#Common functions used by the R_eff and SIR plots for Australian states and NZ


def whiten(color, f):
    """Mix a color with white where f is how much of the original colour to keep"""
    white = np.array(mcolors.to_rgb("white"))
    return (1 - f) * white + f * np.array(mcolors.to_rgb(color))


def th(n):
    """Ordinal of an integer, eg "1st", "2nd" etc"""
    return str(n) + (
        "th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )


def hksar_chp_case_data(start_date=np.datetime64('2022-01-01')):
    """Daily new cases from HK Government Data Web"""
    url = f'http://www.chp.gov.hk/files/misc/latest_situation_of_reported_cases_covid_19_eng.csv'
    df1 = pd.read_csv(url)

    df1 = df1[df1['Number of cases tested positive for SARS-CoV-2 virus'].notnull()]
    
    # Manually added cumulative case numbers
    df2 = pd.read_csv('data/manual.csv')
    df2 = df2[df2['Number of cases tested positive for SARS-CoV-2 virus'].notnull()]
    
    df2 = df2[df1.columns]

    df = pd.concat([df1, df2])
    df = df.drop_duplicates('As of date', keep='first')
    
    dates = np.array(
        [
            np.datetime64(datetime.strptime(date.strip(), "%d/%m/%Y"), 'D') - 1
            for date in df['As of date']
        ]
    )

    cases = -np.diff(np.array(df['Number of cases tested positive for SARS-CoV-2 virus'].astype(int))[::-1])
    cases = cases[::-1]
    dates = dates[1:]

    cases = cases[dates >= start_date]
    dates = dates[dates >= start_date]    

    return dates, cases

def covidlive_case_data(state, start_date=np.datetime64('2021-06-10')):
    """Daily net local cases from covidlive"""
    url = f'https://covidlive.com.au/report/daily-source-overseas/{state.lower()}'
    df = pd.read_html(url)[1]

    df = df[df['LOCAL'] != '-']

    dates = np.array(
        [
            np.datetime64(datetime.strptime(date, "%d %b %y"), 'D') - 1
            for date in df['DATE']
        ]
    )

    cases = np.array(df['LOCAL'].astype(int))[::-1]
    dates = dates[::-1]

    cases = np.diff(cases, prepend=0)[dates >= start_date]
    dates = dates[dates >= start_date]    

    return dates, cases


def covidlive_new_cases(state, start_date=np.datetime64('2021-06-10')):
    """Daily new cases from covidlive"""
    url = f'https://covidlive.com.au/report/daily-cases/{state.lower()}'
    df = pd.read_html(url)[1][:-1]

    df = df[df['NEW'] != '-']

    dates = np.array(
        [
            np.datetime64(datetime.strptime(date, "%d %b %y"), 'D') - 1
            for date in df['DATE']
        ]
    )

    cases = np.array(df['NEW'].astype(int))[::-1]
    dates = dates[::-1]

    cases = cases[dates >= start_date]
    dates = dates[dates >= start_date]    

    return dates, cases


def covidlive_doses_per_100(n, state, population):
    """return cumulative 1st + 2nd doses per 100 population for the last n days"""
    df = pd.read_html(
        f"https://covidlive.com.au/report/daily-vaccinations-first-doses/{state.lower()}"
    )[1]
    first = np.array(df['FIRST'][::-1])
    first_dates = np.array(
        [np.datetime64(datetime.strptime(d, '%d %b %y'), 'D') for d in df['DATE'][::-1]]
    )

    df = pd.read_html(
        f"https://covidlive.com.au/report/daily-vaccinations-people/{state.lower()}"
    )[1]
    second = np.array(df['SECOND'][::-1])
    second_dates = np.array(
        [np.datetime64(datetime.strptime(d, '%d %b %y'), 'D') for d in df['DATE'][::-1]]
    )

    first[np.isnan(first)] = 0
    second[np.isnan(second)] = 0
    maxlen = max(len(first), len(second))
    if len(first) < len(second):
        first = np.concatenate([np.zeros(maxlen - len(first)), first])
        dates = second_dates
    elif len(second) < len(first):
        second = np.concatenate([np.zeros(maxlen - len(second)), second])
        dates = second_dates
    else:
        dates = first_dates

    IX_CORRECTION = np.where(dates==np.datetime64('2021-07-29'))[0][0]

    if state.lower() in ['nt', 'act']:
        first[:IX_CORRECTION] += first[IX_CORRECTION] - first[IX_CORRECTION - 1]
        second[:IX_CORRECTION] += second[IX_CORRECTION] - second[IX_CORRECTION - 1]

    if first[-1] == first[-2]:
        first = first[:-1]
        dates = dates[:-1]
        second = second[:-1]

    daily_doses = np.diff(first + second, prepend=0)

    return 100 * daily_doses.cumsum()[-n:] / population


def gaussian_smoothing(data, pts):
    """Gaussian smooth an array by given number of points"""
    x = np.arange(-4 * pts, 4 * pts + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2))
    smoothed = convolve(data, kernel, mode='same')
    normalisation = convolve(np.ones_like(data), kernel, mode='same')
    return smoothed / normalisation


def log_gaussian_smoothing(data, pts, log_offset=10, vmin=0.1):
    """Take the log of the data, apply gaussian smoothing by given number of points,
    then exponentiate and return. Data is first offset by log_offset, which is then
    subtracted after smoothing. data is clipped to vmin from below before any
    processing, this is not subtracted off later.
    """
    pre = np.log(data.clip(vmin) + log_offset)
    smoothedlog = gaussian_smoothing(pre, pts)
    return np.exp(smoothedlog) - log_offset


def n_day_average(data, n):
    ret = np.cumsum(data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def partial_derivatives(function, x, params, u_params):
    model_at_center = function(x, *params)
    partial_derivatives = []
    for i, (param, u_param) in enumerate(zip(params, u_params)):
        d_param = u_param / 1e6
        params_with_partial_differential = np.zeros(len(params))
        params_with_partial_differential[:] = params[:]
        params_with_partial_differential[i] = param + d_param
        model_at_partial_differential = function(x, *params_with_partial_differential)
        partial_derivative = (model_at_partial_differential - model_at_center) / d_param
        partial_derivatives.append(partial_derivative)
    return partial_derivatives


def model_uncertainty(function, x, params, covariance):
    u_params = [np.sqrt(abs(covariance[i, i])) for i in range(len(params))]
    derivs = partial_derivatives(function, x, params, u_params)
    squared_model_uncertainty = sum(
        derivs[i] * derivs[j] * covariance[i, j]
        for i in range(len(params))
        for j in range(len(params))
    )
    return np.sqrt(squared_model_uncertainty)


def get_confidence_interval(data, confidence_interval=0.68, axis=0):
    """Return median (lower, upper) for a confidence interval of the data along the
    given axis"""
    n = data.shape[axis]
    ix_median = n // 2
    ix_lower = int((n * (1 - confidence_interval)) // 2)
    ix_upper = n - ix_lower
    sorted_data = np.sort(data, axis=axis)
    median = sorted_data.take(ix_median, axis=axis)
    lower = sorted_data.take(ix_lower, axis=axis)
    upper = sorted_data.take(ix_upper, axis=axis)
    return median, (lower, upper)


def stochastic_sir(
    initial_caseload,
    initial_cumulative_cases,
    initial_R_eff,
    tau,
    population_size,
    vaccine_immunity,
    n_days,
    n_trials=10000,
    cov_caseload_R_eff=None,
):
    """Run n trials of a stochastic SIR model, starting from an initial caseload and
    cumulative cases, for a population of the given size, an initial observed R_eff
    (i.e. the actual observed R_eff including the effects of the current level of
    immunity), a mean generation time tau, and an array `vaccine_immunity` for the
    fraction of the population that is immune over time. Must have length n_days, or can
    be a constant. Runs n_trials separate trials for n_days each. cov_caseload_R_eff, if
    given, can be a covariance matrix representing the uncertainty in the initial
    caseload and R_eff. It will be used to randomly draw an initial caseload and R_eff
    from a multivariate Gaussian distribution each trial. Returns the full dataset of
    daily infections, cumulative infections, and R_eff over time, with the first axis of
    each array being the trial number, and the second axis the day.

    The R_eff given to this function should be the growth factor every tau days, that
    is, an R_eff that assumes a generation distribution that is delta-distributed at tau
    days. It will be converted internally to the appropriate R_eff for use with an
    exponential generation that the SIR model assumes, in order to result in the same
    tau-day growth factor.
    """
    if not isinstance(vaccine_immunity, np.ndarray):
        vaccine_immunity = np.full(n_days, vaccine_immunity)
    # Our results dataset over all trials, will extract conficence intervals at the end.
    trials_infected_today = np.zeros((n_trials, n_days))
    trials_R_eff = np.zeros((n_trials, n_days))
    for i in range(n_trials):
        # print(f"trial {i}")
        # Randomly choose an R_eff and caseload from the distribution
        if cov_caseload_R_eff is not None:
            caseload, R_eff = np.random.multivariate_normal(
                [initial_caseload, initial_R_eff], cov_caseload_R_eff
            )
        else:
            caseload, R_eff = initial_caseload, initial_R_eff

        # We define R_eff as the 5-day growth factor in cases, implicitly assuming a
        # generation distribution that is a delta function at 5 days. However, the SIR
        # model is going to generate secondary cases using an exponential generation
        # distribution. So we need to produce an R_eff for use by the SIR model, which,
        # if fed to an exponential generation distribution, would result in the same
        # growth factor over the mean generation time, otherwise the SIR model will
        # predict unrealistically fast growth (the cases that transmit early have more
        # opportunity for further spread). This conversion derived from details in
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1766383/
        R_eff = 1 + np.log(R_eff)

        # Clipping:
        caseload = max(0, caseload)
        R_eff = max(0.1, R_eff)

        cumulative = initial_cumulative_cases
        # First we back out an R0 from the R_eff and existing immunity. In this context,
        # R0 is the rate of spread *including* the effects of restrictions and
        # behavioural change, which are assumed constant here, but excluding immunity
        # due to vaccines or previous infection.
        R0 = R_eff / ((1 - vaccine_immunity[0]) * (1 - cumulative / population_size))
        # Initial pops in each compartment
        infectious = int(round(caseload * tau / R_eff))
        recovered = cumulative - infectious
        for j, vax_immune in enumerate(vaccine_immunity):
            # vax_immune is as fraction of the population, recovered and infectious are
            # in absolute numbers so need to be normalised by population to get
            # susceptible fraction
            s = (1 - vax_immune) * (1 - (recovered + infectious) / population_size)
            s = max(0, s)
            R_eff = s * R0
            infected_today = np.random.poisson(infectious * R_eff / tau)
            recovered_today = np.random.binomial(infectious, 1 / tau)
            infectious += infected_today - recovered_today
            recovered += recovered_today
            cumulative += infected_today
            trials_infected_today[i, j] = infected_today
            trials_R_eff[i, j] = R_eff 

    cumulative_infected = trials_infected_today.cumsum(axis=1) + initial_cumulative_cases

    return trials_infected_today, cumulative_infected, trials_R_eff


def exponential(x, A, k):
    return A * np.exp(k * x)


# Exponential growth, but with the expected rate of decline in k due to vaccines.
def exponential_with_vax(x, A, k, dk_dt):
    return A * np.exp(k * x + 1 / 2 * dk_dt * x ** 2)


# Exponential growth, but with the expected rate of decline in k due to immunity.
def exponential_with_infection_immunity(
    x,
    A,
    k,
    cumulative_cases,
    tau,
    effective_population,
):
    # Susceptible population half a day in the future and half a day in the past:
    s_2 = (1 - (cumulative_cases + A / 2)/ effective_population)
    s_1 = (1 - (cumulative_cases - A / 2) / effective_population)

    dk_dt = 1 / tau * (s_2 / s_1 - 1)
    return A * np.exp(k * x + 1 / 2 * dk_dt * x ** 2)


def clip_params(params, R_max, caseload_max, tau):
    """Clip exponential fit params to be within a reasonable range to suppress when
    unlucky points lead us to an unrealistic exponential blowup. Modifies array
    in-place."""
    params[0] = min(params[0], caseload_max)
    params[1] = min(params[1], np.log(R_max ** (1 / tau)))


def determine_smoothed_cases_and_Reff(
    new,
    smoothing=4,
    padding=12,
    padding_model=exponential,
    fit_pts=20,
    x0=-14,
    delta_x=1,
    pre_fit_smoothing=None,
    tau=5,
    N_monte_carlo=1000,
    R_clip=50,
):
    """Most of the magic happens here. Smooth case numbers and estimate R_eff, and
    covariances. Most of this is concerned with fitting a model to recent case numbers,
    and using that model to pad the data, which is required before we can use a
    symmetric smoothing method on the dataset. This fit mostly determines what the
    latest R_eff estimate will be.

    Args:

    new: daily case numbers

    smoothing: Standard deviation, in days, of the Gaussian smoothing that will be used
        to smooth the log of daily case numbers.

    padding: How many days to pad with an extrapolation based on the model fit to recent
        data. Should be at least 3 × smoothing for good results.

    fit_pts: how many points to include in the fit to recent data. Should be bigger than
        -x0 by a few multiples of delta_x. This is because we're doing a weighted fit,
        with equal weights for recent data, then the weights starting to roll off at
        -x0, to zero by the time we're at -fit_pts.

    x0: index in array of new cases at which the weights in the weighted fit should drop
        off. e.g. for a 14-day fit set to -14.

    delta_x: width, in days, over which the fit weights decrease in the vicinity of x0.

    tau: mean generation time of the virus in days.

    N_monte_carlo: how many iterations to use in Monte-Carlo estimation of covariances
    
    R_clip: maximum value to clip R_eff to in fit results. Mostly to limit
        unrealistically huge uncertainties due to unlucky small numbers when caseloads
        are small.

    Returns:

    new_smoothed: smoothed daily case numbers
    
    u_new_smoothed: 1-sigma uncertainty in new_smoothed each day

    R: estimate of R_eff each day, defined as the growth factor in daily cases over the
        mean generation time tau.

    u_R: 1-sigma uncertainty in R each day

    R_exp: equivalent R_eff that if given to an SIR model with exponentially-distributed
        generation times, will result in a growth factor of R every tau days. 

    cov: covariance matrix for the latest smoothed caseload and R

    cov_exp: covariance matrix for latest smoothed caseload and R_exp

    shot_noise_factor : Measure of day-to-day case fluctuations, defined as the scaling
        factor of Poisson uncertainty required to obtain a reduced chi2 of 1.0 between
        daily cases and smoothed daily cases.

    """

    # Smoothing requires padding to give sensible results at the right edge. Compute a
    # model fit to daily cases over the last fortnight (or whatever x0 is), and pad the
    # data with the fit results prior to smoothing.
    fit_x = np.arange(-fit_pts, 0)
    fit_weights = 1 / (1 + np.exp(-(fit_x - x0) / delta_x))
    pad_x = np.arange(padding)

    params, _ = curve_fit(
        padding_model,
        fit_x,
        (
            n_day_average(new, pre_fit_smoothing)[-fit_pts:]
            if pre_fit_smoothing is not None
            else new[-fit_pts:]
        ),
        sigma=1 / fit_weights,
    )

    clip_params(
        params, R_max=R_clip, caseload_max=2 * new[-fit_pts:].max() + 1, tau=tau
    )

    fit = padding_model(pad_x, *params).clip(0.1, None)

    new_padded = np.zeros(len(new) + padding)
    new_padded[: -padding] = new
    new_padded[-padding:] = fit

    new_smoothed = log_gaussian_smoothing(new_padded, smoothing)[: -padding]
    R = (new_smoothed[1:] / new_smoothed[:-1]) ** tau

    # Arrays for variances and covariances:
    var_R = np.zeros_like(R)
    var_new_smoothed = np.zeros_like(new_smoothed)
    cov_R_new_smoothed = np.zeros_like(R)

    # Uncertainty in new cases is whatever multiple of Poisson noise puts them on
    # average 1 sigma away from the smoothed new cases curve. Only use data when
    # smoothed data > 1.0
    valid = new_smoothed > 1.0
    if valid.sum():
        shot_noise_factor = np.sqrt(
            ((new[valid] - new_smoothed[valid]) ** 2 / new_smoothed[valid]).mean()
        )
    else:
        shot_noise_factor = 1.0
    u_new = shot_noise_factor * np.sqrt(new.clip(1))

    # Monte-carlo of the above with noise to compute variance in R, new_smoothed,
    # and their covariance:

    all_Reffs = np.zeros(N_monte_carlo)

    for i in range(N_monte_carlo):
        new_with_noise = np.random.normal(new, u_new).clip(0.1, None)

        trial_params, trial_cov = curve_fit(
            padding_model,
            fit_x,
            (
                n_day_average(new_with_noise, pre_fit_smoothing)[-fit_pts:]
                if pre_fit_smoothing is not None
                else new_with_noise[-fit_pts:]
            ),
            sigma=1 / fit_weights,
            maxfev=20000,
        )
        if pre_fit_smoothing is not None:
            # Compensate for the decreased noise caused by the additional smoothing:
            trial_cov *= pre_fit_smoothing

        clip_params(
            trial_params,
            R_max=R_clip,
            caseload_max=2 * new_with_noise[-fit_pts:].max() + 1,
            tau=tau,
        )

        trial_params = np.random.multivariate_normal(trial_params, trial_cov)

        clip_params(
            trial_params,
            R_max=R_clip,
            caseload_max=2 * new_with_noise[-fit_pts:].max() + 1,
            tau=tau,
        )

        fit = padding_model(pad_x, *trial_params).clip(0.1, None)
        new_padded[:-padding] = new_with_noise
        new_padded[-padding:] = fit

        new_smoothed_noisy = log_gaussian_smoothing(new_padded, smoothing)[:-padding]
        var_new_smoothed += (new_smoothed_noisy - new_smoothed) ** 2 / N_monte_carlo
        R_noisy = (new_smoothed_noisy[1:] / new_smoothed_noisy[:-1]) ** tau
        var_R += (R_noisy - R) ** 2 / N_monte_carlo
        cov_R_new_smoothed += (
            (new_smoothed_noisy[1:] - new_smoothed[1:]) * (R_noisy - R) / N_monte_carlo
        )
        all_Reffs[i] = R_noisy[-1]

    print("R_eff:", get_confidence_interval(all_Reffs))
    # Construct a covariance matrix for the latest estimate in new_smoothed and R:
    cov = np.array(
        [
            [var_new_smoothed[-1], cov_R_new_smoothed[-1]],
            [cov_R_new_smoothed[-1], var_R[-1]],
        ]
    )

    u_new_smoothed = np.sqrt(var_new_smoothed)
    u_R = np.sqrt(var_R)

    return new_smoothed, u_new_smoothed, R, u_R, cov, shot_noise_factor 


def get_SIR_projection(
    current_caseload,
    cumulative_cases,
    R_eff,
    tau,
    population,
    test_detection_rate,
    vaccine_immunity,
    n_days,
    n_trials,
    cov,
):

    trials_infected_today, trials_cumulative, trials_R_eff = stochastic_sir(
        initial_caseload=current_caseload,
        initial_cumulative_cases=cumulative_cases,
        initial_R_eff=R_eff,
        tau=tau,
        population_size=population * test_detection_rate,
        vaccine_immunity=vaccine_immunity,
        n_days=n_days,
        n_trials=n_trials,
        cov_caseload_R_eff=cov,
    )

    new_projection, (
        new_projection_lower,
        new_projection_upper,
    ) = get_confidence_interval(trials_infected_today)

    cumulative_median, (cumulative_lower, cumulative_upper) = get_confidence_interval(
        trials_cumulative,
    )

    # Convert R_eff back to delta-generation-distribution-Reffs:
    trials_R_eff = np.exp(trials_R_eff - 1)

    R_eff_projection, (
        R_eff_projection_lower,
        R_eff_projection_upper,
    ) = get_confidence_interval(trials_R_eff)

    total_cases = cumulative_median[-1]
    total_cases_lower = cumulative_lower[-1]
    total_cases_upper = cumulative_upper[-1]

    return (
        new_projection,
        new_projection_lower,
        new_projection_upper,
        R_eff_projection,
        R_eff_projection_lower,
        R_eff_projection_upper,
        total_cases,
        total_cases_lower,
        total_cases_upper,
    )

def get_exp_projection(t_projection, current_caseload, R_eff, cov, tau):

    def log_projection_model(t, A, R):
        return np.log(A * R ** (t / tau))

    new_projection = np.exp(log_projection_model(t_projection, current_caseload, R_eff))
    log_new_projection_uncertainty = model_uncertainty(
        log_projection_model, t_projection, (current_caseload, R_eff), cov
    )
    new_projection_upper = np.exp(
        np.log(new_projection) + log_new_projection_uncertainty
    )
    new_projection_lower = np.exp(
        np.log(new_projection) - log_new_projection_uncertainty
    )

    return new_projection, new_projection_lower, new_projection_upper
