# Flood Forecasting System in Lithuania

## Workflow
### System input:
System takes in precipitation values for the last 3 days for each water station. Then, the following values are calculated
1. Precipitation sum in the last 12h
2. Precipitation sum in the last 24h
3. Precipitation sum in the last 48h
4. Precipitation sum in the last 72h
Additionally, average water level for the previous day and date is used
### System output:
The system output is a water elvel for the next day


### Example
Input:
1. 2022-01-01,252,0.1,0.2
2. 2022-01-02,332,15.7,17.4
3. 2022-01-03,416,4.5,6.2

Output:
1. 2022-01-04,475.905859
