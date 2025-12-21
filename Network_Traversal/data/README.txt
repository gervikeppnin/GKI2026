README — Hydraulic Network Data Files

This folder contains the CSV files used to construct the water distribution network model for Veitur's problem submission.

1. junctions_csv.csv
Defines all junction nodes.

Columns:
- name — unique node ID
- elevation — ground elevation (m)
- demand — base demand (L/sec)

2. reservoir_csv.csv
Defines fixed hydraulic head boundary nodes.

Columns:
- name — reservoir ID
- head — hydraulic head (m)

3. pipe_csv_check.csv
Defines all pipes (links) in the network.

Columns:
- start — upstream node name
- end — downstream node name
- length — pipe length (m)
- diameter — inside diameter (mm)
- roughness — Darcy-Weisbach roughness coefficient (mm)
- minorLoss — minor loss coefficient (K), possibly unused but we can leave it open for participants to mess with
- status — OPEN, CLOSED, or CV (check valve, signals one-way flow)
- tags — this field contains the pipe material
- year — installation year

4. pumps_csv.csv
Defines pumps.

Columns:
- name — pump ID
- start — suction node
- end — discharge node
- curve — pump curve name

5. pump_curves_csv.csv
Defines pump performance curves, mandatory for EPANET pump modeling.

Columns:
- name — curve ID
- flow — flow rate (L/sec)
- head — head added by the pump (m)

6. valves_csv.csv
Defines valves.

Columns:
- name — valve ID
- start — upstream node
- end — downstream node
- type — valve type
- setting — valve control setting

7. boundary_flow.csv
Boundary flow time series.

Columns:
- hour — time index
- flow — flow rate (L/sec)

8. sensor_measurements.csv
Hourly sensor pressure measurements.

Columns:
- hour — time index
- pressure_avg — hourly average pressure (bar)
- sensor — sensor junction identifier

All datasets are internally consistent, with no missing node references or duplicate pipes.
