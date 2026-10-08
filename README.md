# Used Car Price Prediction (India)

Regression project that predicts the listing price of used cars from CarDekho/Gaadi-style listing data for six Indian cities. The whole workflow lives in one notebook, [Regression Task.ipynb](Regression%20Task.ipynb): parsing the raw nested data, cleaning, feature engineering, feature selection, model training and evaluation.

## Dataset

Six Excel files, one per city, each with one row per listing:

| City | Listings |
|---|---|
| Bangalore | 1,481 |
| Chennai | 1,419 |
| Delhi | 1,485 |
| Hyderabad | 1,483 |
| Jaipur | 1,120 |
| Kolkata | 1,381 |
| **Total** | **8,369** (8,281 after removing 88 duplicates) |

Every file has the same five columns. Each cell holds a stringified Python dict or list:

- `new_car_detail`: fuel type, body type, km driven, transmission, owner number, make, model, model year, variant and price.
- `new_car_overview`: registration year, insurance, seats, RTO, ownership and engine displacement.
- `new_car_feature`: a list of feature names grouped by section (comfort, safety, interior, exterior and so on).
- `new_car_specs`: engine, power, torque, dimensions, brakes, tyres, mileage and similar specs.

The target is `price`, in INR.


## Workflow

1. **Parse nested data.** Each column is parsed with `ast.literal_eval`.
   - `new_car_detail` is flattened with `json_normalize`.
   - The overview becomes a table of key/value pairs.
   - Features become a 0/1 matrix, about 170 columns, one per feature name.
   - Specs are flattened into columns.
   - The four parts are concatenated per city. The six city frames are then stacked with a `city` column, giving 8,369 rows × 243 columns.
2. **Clean up.**
   - Dropped constant columns, ID-like columns (`centralVariantId`, `priceActual`) and exact duplicate columns (`Transmission`, `Alloy Wheel Size`).
   - Dropped columns with more than 70% missing values (`BoreX Stroke`, `Compression Ratio`, `Ground Clearance Unladen`).
   - Converted price strings such as `₹ 8.11 Lakh` to numbers.
   - Extracted 4-digit years from the year columns.
   - Stripped units from numeric columns such as km, mileage, power, torque and dimensions.
   - Cast the binary feature flags to `Int8`.
3. **Missing values.**
   - Missing flags are set to 0 for three binary features.
   - Integer columns use the median.
   - Float columns use the mean when roughly symmetric (|skew| ≤ 0.5), otherwise the median.
   - Low-cardinality categoricals use the mode.
   - Sparse categoricals get `"Unknown"`.
4. **Outliers.** IQR-based capping (winsorising) on `km`, `Kms Driven`, `Mileage`, `Turning Radius` and `Cargo Volumn`. The target is not capped.
5. **EDA.** Charts for top manufacturers, fuel type split, transmission split, average price by make and by city, and scatter plots of price against mileage, engine, power, torque and model year.
6. **Feature engineering.**
   - Group scores, each the count of features present in a group: `Safety_Score` (36 flags), `Comfort_Score` (39), `Infotainment_Score` (33), `Exterior_Score` (38) and `Driving_Technology_Score` (14).
   - `Power_Per_CC` and `Torque_Per_CC`.
   - `Car_Volume`, which is length × width × height in cubic metres.
   - Several columns turned out to be exact copies of each other (`Engine` = `Engine Displacement` = `Displacement`, `Torque` = `Max Torque`, and the three seat columns). This is noted in the notebook.
7. **Encoding.** Fitted on the training split only.
   - Frequency encoding for `oem` and `variantName`.
   - `TargetEncoder` for the other categoricals.
8. **Feature selection.** Three signals were combined into one average rank: mutual information, Random Forest importance and permutation importance on the test set. These were then compared with RFECV.
   - Highly correlated features (|r| ≥ 0.85) were pruned: `Front Tread`, `Width`, `Wheel Base`, `Length`.
   - A 20-feature set, scored by 5-fold CV with a Random Forest, matched or beat the full 51-feature set (CV R² 0.878 vs 0.871). Swapping `Height` for `Mileage` improved it slightly (0.8815).
   - The final model uses **25 features**: the revised 20 plus `ft`, `bt`, `transmission`, `ownerNo` and `Seats`.
9. **Modelling.** 80/20 train/test split with `random_state=42`. Models are trained on `log1p(price)` and predictions are converted back with `expm1` for the final metrics. Linear Regression also uses standardised inputs.

## Results

Test set, 1,657 listings.

| Model | R² (log price) | R² (price) | MAE (₹) | RMSE (₹) |
|---|---|---|---|---|
| Linear Regression | 0.918 | 0.798 | 162,386 | 579,453 |
| Random Forest | 0.935 | 0.889 | 131,136 | 430,121 |
| Gradient Boosting | 0.935 | 0.830 | 149,482 | 532,645 |
| XGBoost (n_estimators=100, learning_rate=0.1, max_depth=6) | 0.938 | **0.915** | **127,778** | **377,290** |
| XGBoost, tuned with `RandomizedSearchCV` | **0.941** | 0.897 | 129,740 | 414,366 |

The tuned XGBoost used 30 sampled configurations with 5-fold CV. Its best parameters were `n_estimators=500`, `learning_rate=0.05`, `max_depth=4`, `min_child_weight=5`, `subsample=0.9` and `colsample_bytree=0.8`, with a CV R² of 0.933 on the log target.

The notebook ends with an actual-vs-predicted table for the tuned XGBoost.

**Reading the results.** The tuned XGBoost is best on the log scale and has the smallest train/test gap (0.026). The untuned XGBoost scores higher once predictions are converted back to rupees. Rupee-scale metrics weight errors on expensive cars much more heavily than log-scale metrics do, so choose between the two on which metric matters for your use case. Random Forest overfits the most (train R² 0.990 vs test 0.935 on the log scale).

## Requirements

Python 3 with: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (1.3 or later, for `TargetEncoder`), `xgboost`, `openpyxl`. Run the notebook in Jupyter or VS Code.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl
```