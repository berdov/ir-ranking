| Variant | Tune nDCG@10 | Difference vs BM25F |
| --- | ---: | ---: |
| baseline_bm25f | 0.595958 | +0.000000 |
| lexical_lightgbm_raw | 0.610568 | +0.014610 |
| lexical_lightgbm_rank | 0.612941 | +0.016983 |
| lexical_catboost_raw | 0.610649 | +0.014691 |
| lexical_catboost_rank | 0.614246 | +0.018288 |
| blend_lexical_catboost_rank_lexical_lightgbm_rank_cat0.25 | 0.613484 | +0.017526 |
| blend_lexical_catboost_rank_lexical_lightgbm_rank_cat0.5 | 0.613039 | +0.017080 |
| blend_lexical_catboost_rank_lexical_lightgbm_rank_cat0.75 | 0.613793 | +0.017835 |
