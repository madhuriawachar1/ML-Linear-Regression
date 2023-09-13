




| Name  | RMSE  UnR    |   RMSE REl1 | RMSE REl2    |t1     |t2     |t3 
| --    | ---          | ------      |   ------     |------ |------ |------
| manual| 0.9273197590 |             | 0.927459574  |0.00239|       |0.00148
| jax   | 0.9273197608 | 0.931603138 | 0.927459576  |4.89153|0.29175|0.35917
   
jax is better than manual.
  


|Name                            | RMSE                  | time
| SGD with ridge                 | 0.864748797295971     | 0.0189259052276611    
| mini-batch SGD with ridge      | 0.8505949838692853    | 0.0047283172607421 
| SGD with momentum and ridge    | 0.8861044511681929    | 0.0973663330078125  


SGD with momentum and ridge  gradient gives the best result among sgd.
