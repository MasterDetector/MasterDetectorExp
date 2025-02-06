# MasterDetector

Our study on anomaly detection with error repairing by master data in multivariate time series.

- `code/` includes the codes for experiments.
- `data/` includes the trajectory dataset.
  - `time_series_data_25168` includes the time series data.
  - `master_data_1180` includes the master data of the trajectory data.
- `Appendix.pdf` includes the experiment results on three other datasets.

To reproduce the experiments, please run the code in `src/main/java/Experiment.java`, where
- `main_td_scale()` gives the results of varying time series data size. 
- `main_md_scale()` gives the results of varying master data data size.
- `main_error_rate()` gives the results of varying error rate.
- `main_error_range()` gives the results of varying error range.
- `main_error_length()` gives the results of varying error length.
- `main_parameters(true, true, true, true)` gives the results of varying different parameters.
- `whole_data_set(4)` gives the results of the whole dataset.
- `varyingModel(5)` gives the results of varying different auto-regression models.

The proposed methods have been deployed and included as a function in an open-source time series database Apache IoTDB, for anomaly detection with data repairing. 
The introduction document is available in the [system website](https://iotdb.apache.org/UserGuide/latest/SQL-Manual/UDF-Libraries_apache.html#masterdetect), and the code of our proposal is included in the [GitHub repository](https://github.com/apache/iotdb/tree/research/master-detector) of the system.

