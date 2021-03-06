#!/bin/bash

DATE=$(date +%Y%m%d)

curl -L https://www.fhb.gov.hk/download/opendata/COVID19/vaccination-rates-over-time-by-age.csv > data/vaccination-rates-over-time-by-age.csv.$DATE
curl -L http://www.chp.gov.hk/files/misc/latest_situation_of_reported_cases_covid_19_eng.csv > data/latest_situation_of_reported_cases_covid_19_eng.csv.$DATE
curl -L http://www.chp.gov.hk/files/misc/enhanced_sur_covid_19_eng.csv > data/enhanced_sur_covid_19_eng.csv.$DATE
curl -L http://www.chp.gov.hk/files/misc/statistics_on_covid_19_testing_cumulative.csv > data/statistics_on_covid_19_testing_cumulative.csv.$DATE
