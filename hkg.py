#!/usr/bin/env python3

import sys
from datetime import datetime
from pytz import timezone
from pathlib import Path
import json

import urllib
import requests
import pandas as pd
from scipy.signal import convolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.units as munits
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib import font_manager

from reff_plots_common import (
    hksar_chp_case_data,
    exponential,
    determine_smoothed_cases_and_Reff,
    exponential_with_vax,
    exponential_with_infection_immunity,
    get_SIR_projection,
    get_exp_projection,
    whiten,
    th,
)

# Global Variables
# Population of Hong Kong
# https://www.censtatd.gov.hk/en/scode150.html
POP_OF_HKG = 7394700

# Vaccination rate 7 day average
# source: https://static.data.gov.hk/covid-vaccine/summary.json
def hkg_7day_vaccination_rate():
    with urllib.request.urlopen("https://static.data.gov.hk/covid-vaccine/summary.json") as url:
        data = json.loads(url.read().decode())
    return data['sevenDayAvg'] / POP_OF_HKG

# Global settings
# Our uncertainty calculations are stochastic. Make them reproducible, at least:
np.random.seed(0)

def hkg_doses_per_100(n):
    """return HKG cumulative doses per 100 population for the last n days"""
    url = "https://www.fhb.gov.hk/download/opendata/COVID19/vaccination-rates-over-time-by-age.csv"
    df = pd.read_csv(url)
    
    # Convert dates to np.datetime64
    df['Date'] = pd.to_datetime(df['Date'])
    
    dates = np.array(sorted(set([row['Date'] for rownum, row in df.iterrows()])))

    total_doses = {d: 0 for d in dates}

    for rownum, row in df.iterrows():
        date = row['Date']
        total_doses[date] += row['Sinovac 1st dose'] + row['Sinovac 2nd dose'] + row['Sinovac 3rd dose'] + row['BioNTech 1st dose'] + row['BioNTech 2nd dose'] + row['BioNTech 3rd dose'] 

    doses = np.array(list(total_doses.values())).cumsum()

    return 100 * doses[-n:] / POP_OF_HKG

def projected_vaccine_immune_population(t, historical_doses_per_100):
    """compute projected future susceptible population, given an array
    historical_doses_per_100 for cumulative doses doses per 100 population prior to and
    including today (length doesn't matter, so long as it goes back longer than
    VAX_ONSET_MU plus 3 * VAX_ONSET_SIGMA), and assuming a certain vaccine efficacy and
    rollout schedule"""

    # We assume vaccine effectiveness after each dose ramps up the integral of a Gaussian
    # with the following mean and stddev in days:
    VAX_ONSET_MU = 10.5 
    VAX_ONSET_SIGMA = 3.5

    # SEP = np.datetime64('2021-09-01').astype(int) - dates[-1].astype(int)
    # OCT = np.datetime64('2021-10-01').astype(int) - dates[-1].astype(int)

    doses_per_100 = np.zeros_like(t)
    doses_per_100[0] = historical_doses_per_100[-1]

    # History of previously projected rates, so I can remake old projections:
    # if dates[-1] >= np.datetime64('2021-10-21'):
    #     AUG_RATE = None
    #     SEP_RATE = None
    #     OCT_RATE = 0.1
    # elif dates[-1] >= np.datetime64('2021-10-30'):
    #     AUG_RATE = None
    #     SEP_RATE = None
    #     OCT_RATE = 0.5
    # elif dates[-1] >= np.datetime64('2021-10-10'):
    #     AUG_RATE = None
    #     SEP_RATE = None
    #     OCT_RATE = 1.3
    # else:
    #     AUG_RATE = 1.4
    #     SEP_RATE = 1.6
    #     OCT_RATE = 1.8
    
    VAX_RATE = hkg_7day_vaccination_rate()
    
    for i in range(1, len(doses_per_100)):
        # if i < SEP:
        #     doses_per_100[i] = doses_per_100[i - 1] + AUG_RATE
        # elif i < OCT:
        #     doses_per_100[i] = doses_per_100[i - 1] + SEP_RATE
        # else:
        #     doses_per_100[i] = doses_per_100[i - 1] + OCT_RATE
        doses_per_100[i] = doses_per_100[i - 1] + VAX_RATE
        
#     if dates[-1] >= np.datetime64('2021-11-21'):
#         MAX_DOSES_PER_100 = 2 * 84.0
#     else:
#         MAX_DOSES_PER_100 = 2 * 85.0
    
    # Cap max doses per 100 at 85%
    MAX_DOSES_PER_100 = 2*85.0
    
    doses_per_100 = np.clip(doses_per_100, 0, MAX_DOSES_PER_100)

    all_doses_per_100 = np.concatenate([historical_doses_per_100, doses_per_100])
    # The "prepend=0" makes it as if all the doses in the initial day were just
    # administered all at once, but as long as historical_doses_per_100 is long enough
    # for it to have taken full effect, it doesn't matter.
    daily = np.diff(all_doses_per_100, prepend=0)

    # convolve daily doses with a transfer function for delayed effectiveness of vaccines
    pts = int(VAX_ONSET_MU + 3 * VAX_ONSET_SIGMA)
    x = np.arange(-pts, pts + 1, 1)
    kernel = np.exp(-((x - VAX_ONSET_MU) ** 2) / (2 * VAX_ONSET_SIGMA ** 2))
    kernel /= kernel.sum()
    convolved = convolve(daily, kernel, mode='same')

    effective_doses_per_100 = convolved.cumsum()

    immune = 0.4 * effective_doses_per_100[len(historical_doses_per_100):] / 100

    return immune

converter = mdates.ConciseDateConverter()

munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime] = converter

VAX = 'vax' in sys.argv
OLD = 'old' in sys.argv

if not VAX and sys.argv[1:]:
    if len(sys.argv) == 2:
        LGA_IX = int(sys.argv[1])
    elif OLD and len(sys.argv) == 3:
        OLD_END_IX = int(sys.argv[2])
    else:
        raise ValueError(sys.argv[1:])

if OLD:
    VAX = True

#dates, new = covidlive_new_cases('HKG', start_date=np.datetime64('2021-05-10'))
dates, new = hksar_chp_case_data(start_date=np.datetime64('2022-01-01'))

# Use test detection rate of 1 in 5 as suggested by Prof Ben Cowling
if dates[-1] >= np.datetime64('2022-02-08'):
    TEST_DETECTION_RATE = 0.2
else:
    TEST_DETECTION_RATE = 0.27

START_VAX_PROJECTIONS = 1
all_dates = dates
all_new = new

# Current vaccination level:
doses_per_100 = hkg_doses_per_100(n=len(dates))


if OLD:
    dates = dates[:START_VAX_PROJECTIONS + OLD_END_IX]
    new = new[:START_VAX_PROJECTIONS + OLD_END_IX]
    doses_per_100 = doses_per_100[:START_VAX_PROJECTIONS + OLD_END_IX]

START_PLOT = np.datetime64('2022-01-01')
END_PLOT = np.datetime64('2022-05-01') if VAX else dates[-1] + 28

tau = 5  # reproductive time of the virus in days
R_clip = 50

immune = projected_vaccine_immune_population(np.arange(100), doses_per_100)
s = 1 - immune
dk_dt = 1 / tau * (s[1] / s[0] - 1)

# Keep the old methodology for old plots:
if dates[-1] >= np.datetime64('2022-01-04'):
    padding_model = lambda x, A, k: exponential_with_infection_immunity(
        x,
        A,
        k,
        cumulative_cases=new.sum(),
        tau=tau,
        effective_population=TEST_DETECTION_RATE * POP_OF_HKG,
    )
elif dates[-1] >= np.datetime64('2021-10-27'):
    padding_model = lambda x, A, k: exponential_with_vax(x, A, k, dk_dt)
else:
    padding_model = exponential

# Whether or not to do a 5dma of data prior to the fit. Change of methodology as of
# 2021-11-19, so keep old methodology for remaking plots prior to then. Changed
# methodology back on 2021-12-11.
if dates[-1] > np.datetime64('2021-12-10'):
    PRE_FIT_SMOOTHING = None
elif dates[-1] > np.datetime64('2021-11-18'):
    PRE_FIT_SMOOTHING = 5
else:
    PRE_FIT_SMOOTHING = None

    
# Where the magic happens, estimate everything:
(
    new_smoothed,
    u_new_smoothed,
    R,
    u_R,
    cov,
    shot_noise_factor,
) = determine_smoothed_cases_and_Reff(
    new,
    fit_pts=min(20, len(dates[dates >= START_PLOT])),
    pre_fit_smoothing=PRE_FIT_SMOOTHING,
    padding_model=padding_model,
    R_clip=R_clip,
    tau=tau,
)

# Fudge what would happen with a different R_eff:
# cov_R_new_smoothed[-1] *= 0.05 / np.sqrt(variance_R[-1])
# R[-1] = 0.75
# variance_R[-1] = 0.05**2

R = R.clip(0, None)
R_upper = (R + u_R).clip(0, R_clip)
R_lower = (R - u_R).clip(0, R_clip)

new_smoothed = new_smoothed.clip(0, None)
new_smoothed_upper = (new_smoothed + u_new_smoothed).clip(0, None)
new_smoothed_lower = (new_smoothed - u_new_smoothed).clip(0, None)


# Projection of daily case numbers:
days_projection = (END_PLOT - dates[-1]).astype(int)
t_projection = np.linspace(0, days_projection, days_projection + 1)


if VAX:
    # Fancy stochastic SIR model
    (
        new_projection,
        new_projection_lower,
        new_projection_upper,
        R_eff_projection,
        R_eff_projection_lower,
        R_eff_projection_upper,
        total_cases,
        total_cases_lower,
        total_cases_upper,
    ) = get_SIR_projection(
        current_caseload=new_smoothed[-1],
        cumulative_cases=new.sum(),
        R_eff=R[-1],
        tau=tau,
        population=POP_OF_HKG,
        test_detection_rate=TEST_DETECTION_RATE,
        vaccine_immunity=projected_vaccine_immune_population(
            t_projection, doses_per_100
        ),
        n_days=days_projection + 1,
        n_trials=1000 if OLD else 10000,  # just save some time if we're animating
        cov=cov,
    )

else:
    # Simple model, no vaccines or community immunity
    new_projection, new_projection_lower, new_projection_upper = get_exp_projection(
        t_projection=t_projection,
        current_caseload=new_smoothed[-1],
        R_eff=R[-1],
        cov=cov,
        tau=tau,
    )


NIGHT_RESTAURANT_BAN = np.datetime64('2022-01-14')
CAP599F_ADDITIONAL_PREMISES = np.datetime64('2022-02-10')
VACCINE_BUBBLE = np.datetime64('2022-02-24')
# END_LOCKDOWN = np.datetime64('2021-10-15')
# FURTHER_EASING = np.datetime64('2021-10-22')
# EASING_95  = np.datetime64('2021-11-12')

fig1 = plt.figure(figsize=(10, 6))
ax1 = plt.axes()


#ax1.fill_betweenx(
#    [-10, 10],
#    [LOCKDOWN, LOCKDOWN],
#    [END_LOCKDOWN, END_LOCKDOWN],
#    color=whiten("red", 0.45),
#    linewidth=0,
#    label="Lockdown",
#)
#

ax1.fill_betweenx(
   [-10, 10],
   [NIGHT_RESTAURANT_BAN, NIGHT_RESTAURANT_BAN],
   [CAP599F_ADDITIONAL_PREMISES, CAP599F_ADDITIONAL_PREMISES],
   color=whiten("yellow", 0.25),
   linewidth=0,
   label="6pm Dining Curfew 晚市禁堂食",
)

ax1.fill_betweenx(
   [-10, 10],
   [CAP599F_ADDITIONAL_PREMISES, CAP599F_ADDITIONAL_PREMISES],
   [VACCINE_BUBBLE, VACCINE_BUBBLE],
   color=whiten("yellow", 0.5),
   linewidth=0,
   label="2022-02-10 Restrictions\n限聚令收緊",
)

ax1.fill_betweenx(
   [-10, 10],
   [VACCINE_BUBBLE, VACCINE_BUBBLE],
   [END_PLOT, END_PLOT],
   color=whiten("green", 0.25),
   linewidth=0,
   label="Vaccine Pass+Restrictions\n疫苗通行證+限聚令收緊",
)


ax1.fill_between(
    dates[1:] + 1,
    R,
    label=R"$R_\mathrm{eff}$ 有效傳染數",
    step='pre',
    color='C0',
)

if VAX:
    ax1.fill_between(
        np.concatenate([dates[1:].astype(int), dates[-1].astype(int) + t_projection]) + 1,
        np.concatenate([R_lower, R_eff_projection_lower]),
        np.concatenate([R_upper, R_eff_projection_upper]),
        label=f"$R_\\mathrm{{eff}}$/projection uncertainty\n推算不確定性",
        color='cyan',
        edgecolor='blue',
        alpha=0.2,
        step='pre',
        zorder=2,
        hatch="////",
    )
    ax1.fill_between(
        dates[-1].astype(int) + t_projection + 1,
        R_eff_projection,
        label=f"$R_\\mathrm{{eff}}$ (projection預測) ",
        step='pre',
        color='C0',
        linewidth=0,
        alpha=0.75
    )
else:
    ax1.fill_between(
        dates[1:] + 1,
        R_lower,
        R_upper,
        label=R"$R_\mathrm{eff}$ uncertainty 不確定性",
        color='cyan',
        edgecolor='blue',
        alpha=0.2,
        step='pre',
        zorder=2,
        hatch="////",
    )


ax1.axhline(1.0, color='k', linewidth=1)
ax1.axis(xmin=START_PLOT, xmax=END_PLOT, ymin=0, ymax=5)
ax1.grid(True, linestyle=":", color='k', alpha=0.5)

ax1.set_ylabel(R"$R_\mathrm{eff}$ 有效傳染數")

u_R_latest = (R_upper[-1] - R_lower[-1]) / 2

R_eff_string = fR"$R_\mathrm{{eff}}={R[-1]:.02f} \pm {u_R_latest:.02f}$"

latest_update_day = datetime.fromisoformat(str(dates[-1] + 1))
latest_update_day = f'{latest_update_day.strftime("%B")} {th(latest_update_day.day)}'
latest_update_day_iso = datetime.fromisoformat(str(dates[-1] + 1)).strftime("%Y-%m-%d")

if VAX:
    title_lines = [
        f"香港冠狀病毒病數學模型推算 SIR model of the HKSAR as of {latest_update_day_iso}",
        f"Starting from currently estimated {R_eff_string} 開始",
    ]
else:
    region = "the HKSAR"
    title_lines = [
        f"香港冠狀病毒病的有效傳染數 $R_\\mathrm{{eff}}$ in {region} as of {latest_update_day_iso}",
        f"最新估計 Latest estimate: {R_eff_string}",
    ]
    
ax1.set_title('\n'.join(title_lines))

ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
ax2 = ax1.twinx()
if OLD:
    ax2.step(all_dates + 1, all_new + 0.02, color='purple', alpha=0.5)
ax2.step(dates + 1, new + 0.02, color='purple', label='Daily cases\n每日個案')
ax2.plot(
    dates.astype(int) + 0.5,
    new_smoothed,
    color='magenta',
    label='Daily cases(smoothed)\n每日個案(平滑）',
)

ax2.fill_between(
    dates.astype(int) + 0.5,
    new_smoothed_lower,
    new_smoothed_upper,
    color='magenta',
    alpha=0.3,
    linewidth=0,
    zorder=10,
    label=f'Smoothing {"projection" if VAX else "trend"} uncertainty\n平滑{"推算" if VAX else "趨勢"}不確定性',
)
ax2.plot(
    dates[-1].astype(int) + 0.5 + t_projection,
    new_projection.clip(0, 1e6),  # seen SVG rendering issues when this is big
    color='magenta',
    linestyle='--',
    label=f'Daily cases ({"SIR projection" if VAX else "exponential trend"})\n每日個案({"SIR 推算" if VAX else "趨勢"})',
)
ax2.fill_between(
    dates[-1].astype(int) + 0.5 + t_projection,
    new_projection_lower.clip(0, 1e6),  # seen SVG rendering issues when this is big
    new_projection_upper.clip(0, 1e6),
    color='magenta',
    alpha=0.3,
    linewidth=0,
)

ax2.set_ylabel("Daily cases(log scale)\n每日個案(對數刻度)")

ax2.set_yscale('log')
ax2.axis(ymin=1, ymax=100_000)
fig1.tight_layout(pad=1.8)

handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

handles += handles2
labels += labels2

if VAX:
    # order = [4, 6, 5, 7, 8, 10, 9, 0, 1, 2, 3]
    order = [3, 5, 4, 6, 7, 9, 8, 0, 1, 2]
else:
    # order = [4, 5, 6, 7, 9, 8, 0, 1, 2, 3]
    order = [3, 4, 5, 6, 8, 7, 0, 1, 2]
    
fontP = font_manager.FontProperties()
fontP.set_family('SimHei')
fontP.set_size(8)

plt.rcParams['font.sans-serif']=['SimHei']

ax2.legend(
    # handles,
    # labels,
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='upper left',
    ncol=1,
    prop=fontP,
)


ax2.yaxis.set_major_formatter(mticker.EngFormatter())
ax2.yaxis.set_minor_formatter(mticker.EngFormatter())
ax2.tick_params(axis='y', which='minor', labelsize='x-small')
plt.setp(ax2.get_yminorticklabels()[1::2], visible=False)
locator = mdates.DayLocator([1, 15])
ax1.xaxis.set_major_locator(locator)
formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
ax1.xaxis.set_major_formatter(formatter)

ax2.tick_params(axis='y', colors='purple', which='both')
ax1.spines['right'].set_color('purple')
ax2.spines['right'].set_color('purple')
ax2.yaxis.label.set_color('purple')

ax1.tick_params(axis='y', colors='C0', which='both')
ax1.spines['left'].set_color('C0')
ax2.spines['left'].set_color('C0')
ax1.yaxis.label.set_color('C0')

axpos = ax1.get_position()

text = fig1.text(
    0.99,
    0.02,
    "@arthurhsliu | based on work by @chrisbilbo",
    size=8,
    alpha=0.5,
    color=(0, 0, 0.25),
    fontfamily="monospace",
    horizontalalignment="right"
)
text.set_bbox(dict(facecolor='white', alpha=0.8, linewidth=0))

if VAX:
    total_cases_range = f"{total_cases_lower/1000:.1f}k—{total_cases_upper/1000:.1f}k"
    text = fig1.text(
        0.70,
        0.80,
        "\n".join(
            [
                f"Projected total cases in outbreak",
                f"推算爆發的總數:  {total_cases/1000:.1f}k",
                f"68% range範圍:  {total_cases_range}",
            ]
        ),
        fontsize='small',
    )
    text.set_bbox(dict(facecolor='white', alpha=0.8, linewidth=0))

    suffix = '_vax'
else:
    suffix = ''

if OLD:
    fig1.savefig(f'hkg_animated/{OLD_END_IX:04d}.png', dpi=133)
else:
    fig1.savefig(f'COVID_HKG{suffix}.svg')
    fig1.savefig(f'COVID_HKG{suffix}.png', dpi=133)
if True: # Just to keep the diff with hkg.py sensible here
    ax2.set_yscale('linear')
    if OLD and dates[-1] < np.datetime64('2021-12-15'):
        ymax = 100
    elif VAX:
        ymax = 10_000
    else:
        ymax = 10_000
    ax2.axis(ymin=0, ymax=ymax)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(ymax / 10))
    ax2.yaxis.set_major_formatter(mticker.EngFormatter())
    ax2.set_ylabel("Daily confirmed cases(linear scale)\n每日個案(線性刻度)")
    if OLD:
        fig1.savefig(f'hkg_animated_linear/{OLD_END_IX:04d}.png', dpi=133)
    else:
        fig1.savefig(f'COVID_HKG{suffix}_linear.svg')
        fig1.savefig(f'COVID_HKG{suffix}_linear.png', dpi=133)

# Save some deets to a file for the auto reddit posting to use:
try:
    # Add to existing file if already present
    stats = json.loads(Path("latest_hkg_stats.json").read_text())
except FileNotFoundError:
    stats = {}

if True: # keep the diff simple
    stats['R_eff'] = R[-1] 
    stats['u_R_eff'] = u_R_latest
    stats['today'] = str(np.datetime64(datetime.now(), 'D'))

if VAX:
    # Case number predictions
    stats['projection'] = []
    # in case I ever want to get the orig projection range not expanded - like to
    # compare past projections:
    stats['SHOT_NOISE_FACTOR'] = shot_noise_factor 
    for i, cases in enumerate(new_projection):
        date = dates[-1] + i
        lower = new_projection_lower[i]
        upper = new_projection_upper[i]
        lower = lower - shot_noise_factor * np.sqrt(lower)
        upper = upper + shot_noise_factor * np.sqrt(upper)
        lower = max(lower, 0)
        stats['projection'].append(
            {'date': str(date), 'cases': cases, 'upper': upper, 'lower': lower}
        )
        if i < 8:
            print(f"{cases:.0f} {lower:.0f}—{upper:.0f}")

if not OLD:
    # Only save data if this isn't a re-run on old data
    Path("latest_hkg_stats.json").write_text(json.dumps(stats, indent=4))

    # Update the date in the HTML
    html_file = 'COVID_HKG.html'
    html_lines = Path(html_file).read_text().splitlines()
    now = datetime.now(timezone('Asia/Hong_Kong')).strftime('%Y-%m-%d %H:%M')
    for i, line in enumerate(html_lines):
        if 'Last updated' in line:
            html_lines[i] = f'    Last updated: {now} Hong Kong time'
    Path(html_file).write_text('\n'.join(html_lines) + '\n')
    plt.show()
