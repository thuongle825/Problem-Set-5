# Problem-Set-5
# AI Transcript
**Model:** Claude ~ Opus 5.5 ; **Date:** 11/24

***Prompt:*** 

[Prep] Act as a model risk analyst comparing two candidate credit
models before one goes into production.

[Request] I have two fitted classifiers scored on my test set. The
random forest got 0.9?? accuracy and the penalised logistic model got
0.9??. Write Python that tells me whether the forest is significantly
better, and give me a p-value.

[Iterate] numpy, pandas and scipy only. Reuse my y_test, my two
prediction arrays, and nothing else.

[Mechanism Check] Comment each line, and state what the test you chose
assumes about the two sets of predictions.

[Evaluate] One sentence I can put in front of the head of risk.

***Reply:***

The test is McNemar's exact test. Both models were scored on the same applicants, so their predictions are paired. Comparing the two accuracies as if they came from independent samples would overstate the uncertainty, and a z-test on 0.9x vs 0.9y would be wrong. McNemar looks only at the cases where the models disagree. If the models are equally good, each disagreement is a coin flip between "forest right" and "logistic right." The script uses the exact binomial version, which stays valid when there are few disagreements. It reports a two-sided p-value as the primary result and a one-sided p-value for "forest is better." You should commit to one of these before you look at the numbers.

What the test assumes about the two sets of predictions

They are paired row by row. Prediction i from each model refers to the same applicant, in the same order as y_test. The assert checks length but cannot check order.
The models may be correlated, and that's fine. McNemar is designed for two models that tend to get the same cases right or wrong. It does not assume their errors are independent of each other.
Test rows are independent of one another. It assumes one row per applicant, with no duplicate customers and no time-clustered defaults. If the same borrower appears repeatedly, the p-value will be too small.
The null is equal error rates, meaning disagreements are equally likely to go either way. Agreements contribute nothing to the test.
The test set is untouched. If the threshold or hyperparameters were tuned on this data, the p-value is optimistic.
It tests accuracy at the fixed cutoff only. It says nothing about ranking (AUC), calibration, or cost-weighted errors. It also ignores how much the forest would vary if retrained.

import numpy as np                                    # array maths
import pandas as pd                                   # contingency table display
from scipy.stats import binomtest                     # exact binomial test (McNemar's exact form)

# --- The two prediction arrays, built from the models you already fitted ---
pred_rf = rf.predict(X_test)                          # forest was fitted on unscaled X, so predict on X_test
pred_lr = lasso.predict(X_test_s)                     # penalised (L1) logistic was fitted on scaled X, so use X_test_s

# --- Flat arrays under new names, so nothing in your notebook (rf, y, c) gets overwritten ---
y_true_arr = np.asarray(y_test).ravel()               # true labels; drops the shuffled pandas index
rf_arr = np.asarray(pred_rf).ravel()                  # forest 0/1 predictions (0.5 cutoff)
lr_arr = np.asarray(pred_lr).ravel()                  # LASSO-logistic 0/1 predictions (0.5 cutoff)

assert y_true_arr.shape == rf_arr.shape == lr_arr.shape, "Arrays must be the same length"  # pairing check

rf_ok = rf_arr == y_true_arr                          # True where the forest is correct
lr_ok = lr_arr == y_true_arr                          # True where the logistic model is correct

print(f"Forest accuracy:   {rf_ok.mean():.4f}")       # should match your earlier forest test accuracy
print(f"Logistic accuracy: {lr_ok.mean():.4f}")       # should match your earlier LASSO accuracy

table = (pd.crosstab(pd.Series(rf_ok, name="Forest correct"),            # 2x2 table of agreement
                     pd.Series(lr_ok, name="Logistic correct"))
           .reindex(index=[True, False], columns=[True, False], fill_value=0))  # keep all four cells
print(table, "\n")                                    # show the full paired picture

n_rf_wins = int(np.sum(rf_ok & ~lr_ok))               # forest right, logistic wrong
n_lr_wins = int(np.sum(~rf_ok & lr_ok))               # logistic right, forest wrong
n_disc = n_rf_wins + n_lr_wins                        # only disagreements carry information

if n_disc == 0:                                       # identical correctness on every row
    print("No discordant cases: the models cannot be distinguished on this test set.")
else:
    p_two = binomtest(n_rf_wins, n_disc, 0.5, alternative="two-sided").pvalue  # H1: error rates differ
    p_one = binomtest(n_rf_wins, n_disc, 0.5, alternative="greater").pvalue    # H1: forest is better
    print(f"Discordant cases: forest wins {n_rf_wins}, logistic wins {n_lr_wins} (of {n_disc})")
    print(f"McNemar exact p-value, two-sided:           {p_two:.4g}")
    print(f"McNemar exact p-value, one-sided (RF > LR): {p_one:.4g}")
    verdict = "is" if p_two < 0.05 else "is not"      # pre-specified 5% two-sided decision rule
    better = "forest" if n_rf_wins > n_lr_wins else "logistic model" if n_lr_wins > n_rf_wins else "neither model"  # direction
    print(f"\nHead-of-risk line: On the same {len(y_true_arr):,} holdout cases, the forest was right "
          f"where the logistic model was wrong in {n_rf_wins} case(s) and the reverse happened in {n_lr_wins}; "
          f"McNemar's exact test gives p = {p_two:.3g}, so the accuracy difference (in favour of the {better}) "
          f"{verdict} statistically significant at the 5% level.")
