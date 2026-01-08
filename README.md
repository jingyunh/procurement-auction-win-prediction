\# Predicting Procurement Auction Wins (PM)



This repository contains a small, self contained code sample that predicts whether a bidder wins a PM procurement auction using ex ante bidder and project characteristics.



The purpose of this project is not to make causal claims. Instead, it demonstrates a careful and reproducible predictive modeling workflow using real procurement auction data, with particular attention to leakage control, transparent baselines, and interpretable evaluation.



---



\## Data



The input data are \*\*not included\*\* in this repository.



The script expects a CSV file located at: "data/raw/pm\_bidder\_level.csv".





\### Dataset description



The dataset is a bidder level panel constructed from procurement auction records. Each row corresponds to a \*\*bidder × PM auction\*\* observation.



The data include:

\- bidder characteristics (for example, firm size indicators and capacity proxies),

\- project and auction characteristics (such as engineer estimates and competition intensity),

\- and auction outcomes.



Only \*\*ex ante information\*\* is used as model input. Bid values and other post bid outcomes are intentionally excluded from the feature set.



\### Required columns



The script expects the following variables to be present:



\- `win`

  Binary indicator equal to 1 if the bidder wins the PM auction.



\- `jpnumber`

  Auction identifier. This variable is used to perform group based train test splits at the auction level.



\- Ex ante predictors used in the models:

  - `is\\\_large\\\_firm`

  - `lengest` (log engineer estimate)

  - `lnumb` (log number of bidders)

  - `util` (capacity or utilization proxy)

  - `cross\\\_type\\\_count` (cross type competition measure)



Observations with missing or invalid values of `win` are dropped prior to estimation.



---



\## Modeling task



\*\*Task:\*\* Predict the probability that a bidder wins a PM auction.



This is a \*\*predictive classification problem\*\*, not a causal analysis.



Three models are estimated:

1\. \*\*Logistic regression (baseline)\*\*

   Uses a small set of common and interpretable predictors.



2\. \*\*Logistic regression (full)\*\*

   Extends the baseline by adding capacity and cross type competition measures.



3\. \*\*Tree based benchmark\*\*

   A histogram based gradient boosting classifier estimated on numeric features only, included as a nonlinear comparison.



---



\## Evaluation design



\### Leakage control



To avoid information leakage across bidders within the same auction, the train test split is performed at the \*\*auction level\*\* using a group based split on `jpnumber`. As a result, no auction appears in both the training and test sets.



\### Metrics



Model performance is evaluated using:

\- ROC AUC (ranking performance),

\- Brier score (probability calibration).



\### Stability



Five fold cross validation AUC is reported on the training set to assess stability across random splits.



---



\## Outputs



Running the training script produces the following outputs:



\- \*\*Console output\*\*

  Prints cross validation and test set performance for all models.



\- \*\*Saved metrics\*\*

"reports/metrics.txt"



A text summary of test set performance and model comparisons.



\- \*\*Figures\*\*

"reports/calibration\_curve.png", "reports/roc\_curves.png"



Calibration and ROC plots evaluated on the test set.



---



\## How to run



From the repository root:



```bash

python -m venv .venv

\\# Windows PowerShell

.\\\\.venv\\\\Scripts\\\\activate



pip install -r requirements.txt

python src/train\\\_win\\\_model.py





The script will print evaluation results to the console and save metrics and figures under the reports/ directory.








