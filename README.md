\# California Housing Price Prediction



Predicting the median house value of California districts from census data,

using scikit-learn. The project covers data exploration, baseline models,

gradient boosting, error analysis and feature importance.



> Note: the target is the \*\*median house value of a district\*\* (1990 census),

> not the price of an individual house or apartment.



\## Results



Test set: 4,128 districts (20% of the data), identical split for every model.



| Model | MAE | RMSE | R² |

|---|---|---|---|

| Always the mean (baseline) | $90,607 | $114,486 | 0.000 |

| Linear regression | $53,320 | $74,558 | 0.576 |

| Random Forest | $32,679 | $50,429 | 0.806 |

| \*\*Gradient Boosting\*\* | \*\*$30,154\*\* | \*\*$45,407\*\* | \*\*0.843\*\* |



The best model predicts the district value within $50,000 of the truth

for \*\*82.7%\*\* of test districts.



\## Data exploration



!\[Price map](assets/price\_map.png)



Expensive districts concentrate along the coast (Bay Area, Los Angeles),

so location matters a lot.



!\[Correlations](assets/correlations.png)



\## Model analysis



!\[Predicted vs actual](assets/predicted\_vs\_actual.png)



!\[Feature importance](assets/feature\_importance.png)



Permutation importance shows that \*\*latitude and longitude\*\* are the most

important features, followed by median income. Population and the number of

bedrooms add almost nothing.



!\[Error distribution](assets/error\_distribution.png)



\## Project structure



```

src/

├── data.py       # dataset loading

├── explore.py    # exploratory analysis and plots

├── split.py      # train/test split (fixed seed)

├── baseline.py   # mean and linear regression baselines

├── train.py      # trains models, saves the best one

└── evaluate.py   # metrics, error analysis, plots

```



\## Installation



```bash

git clone https://github.com/om-mmdv/california-housing.git

cd california-housing

python -m venv venv

venv\\Scripts\\activate        # Windows

source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

```



\## Usage



```bash

python -m src.explore     # data exploration and plots

python -m src.baseline    # baseline scores

python -m src.train       # train models, save the best

python -m src.evaluate    # evaluate the best model

```



The dataset is downloaded automatically on the first run (about 1 MB).



\## Limitations



\- House values above $500,000 are capped at 5.0 in the data (about 5% of

&#x20; districts), so the model cannot learn real prices above that level. Three of

&#x20; the five largest errors are such capped districts.

\- The data is from the 1990 census: the model reflects prices of that time,

&#x20; not today's market.

\- Targets are district medians, not individual properties.

\- No hyperparameter tuning was done; the models use reasonable defaults.



\## Possible improvements



\- Tune hyperparameters (cross-validation, grid or random search)

\- Engineer features, e.g. distance to the coast or to big cities

\- Try other models (XGBoost, LightGBM)

