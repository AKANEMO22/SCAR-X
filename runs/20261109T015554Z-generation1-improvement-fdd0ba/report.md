# G1 local capability comparison

| Seed | Condition | Dev text | Confirm text | Digits |
|---|---|---:|---:|---:|
| 17 | original_full | 0.188 | 0.125 | 0.881 |
| 17 | lexical_full | 0.125 | 0.094 | 0.892 |
| 17 | lexical_readout | 0.125 | 0.000 | 0.897 |
| 29 | original_full | 0.188 | 0.094 | 0.889 |
| 29 | lexical_full | 0.125 | 0.062 | 0.839 |
| 29 | lexical_readout | 0.062 | 0.031 | 0.881 |
| 43 | original_full | 0.188 | 0.094 | 0.878 |
| 43 | lexical_full | 0.062 | 0.062 | 0.856 |
| 43 | lexical_readout | 0.000 | 0.000 | 0.858 |

{
  "lexical_vs_original": {
    "mean": -0.03125,
    "lower95": -0.03125,
    "upper95": -0.03125,
    "independent_seeds": 3,
    "method": "paired seed-cluster bootstrap; exploratory with only three clusters"
  },
  "full_vs_readout": {
    "mean": 0.0625,
    "lower95": 0.03125,
    "upper95": 0.09375,
    "independent_seeds": 3,
    "method": "paired seed-cluster bootstrap; exploratory with only three clusters"
  }
}

Three independent seeds, exploratory uncertainty; familiar8families. No representation/retention/scaling success claim.
