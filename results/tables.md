# Results (test subjects, falls at impact positions 1-3 s)

## Fall detection: all models

                                                       Model Params  F1 (%)  Precision (%)  Recall (%)  AUC (%)  Model energy (nJ)  Input energy (nJ)  Total energy (nJ)
                                                   SVM (RBF)  20.4K   99.25          99.12       99.39    99.97              94.28              76.32             170.60
                                                        SCNN  49.7K   98.77          99.11       98.43    99.90             211.17              65.98             277.15
                                                    ANN twin  49.7K   98.74          98.64       98.84    99.92            4461.17              65.98            4527.15
                                                        k-NN 330.2K   98.39          98.96       97.82    99.60            1529.01              76.32            1605.33
                                               Random Forest 158.4K   97.65          98.88       96.45    99.80               3.38              76.32              79.70
                               Threshold (acc. + gyro peaks)      2   88.94          82.41       96.59    91.17               0.00              76.32              76.32
                                       Threshold (acc. peak)      1   86.23          77.88       96.59    93.00               0.00              76.32              76.32
SCNN, original pipeline (aligned windows, zero-phase filter)  49.7K   68.20          70.86       65.73    75.34                 --                 --                 --


## Fall direction: all models

                Model Params  Macro-F1 (%)  Precision (%)  Recall (%)  AUC (%)  Model energy (nJ)  Input energy (nJ)  Total energy (nJ)
            SVM (RBF) 100.4K         76.84          76.95       76.86    92.64             464.20              76.32             540.52
        Random Forest 256.0K         73.02          75.33       72.20    93.18               3.50              76.32              79.82
             ANN twin  50.0K         72.74          72.56       73.80    91.78            4484.76              65.98            4550.74
                 SCNN  50.0K         70.25          70.00       71.05    91.40             209.48              65.98             275.46
                 k-NN 167.2K         68.33          68.67       68.64    85.94             774.23              76.32             850.55
Impact-direction rule     24         54.29          55.16       54.91    78.13               0.11              76.32              76.43


## Fall detection: SNN efficiency variants

Model Params  F1 (%)  Precision (%)  Recall (%)  AUC (%)  Model energy (nJ)  Input energy (nJ)  Total energy (nJ)
 SCNN  49.7K   98.77          99.11       98.43    99.90             211.17              65.98             277.15


## Fall detection: encoding-stream ablation

Model Params  F1 (%)  Precision (%)  Recall (%)  AUC (%)  Model energy (nJ)  Input energy (nJ)  Total energy (nJ)
 SCNN  49.7K   98.77          99.11       98.43    99.90             211.17              65.98             277.15

