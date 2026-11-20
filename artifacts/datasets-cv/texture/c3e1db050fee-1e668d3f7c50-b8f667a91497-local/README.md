# Ten-fold cross-validation benchmark: texture
Target column: `Class`. Task: classification. Metric: balanced_accuracy (higher is better).

`data.csv` is the exact AGE-ML exported CSV. `cv_contract.json` fixes row positions
and all ten train/validation folds. There is NO independent test set. Do not
shuffle, split, filter, reorder or sample the dataset. Never add row IDs as features.

For EVERY candidate, return a self-contained Python factory module. The harness
writes it as `model_factory.py` in the node output folder. The module must provide `build_model(X_train, y_train)` returning a NEW, unfitted model with
`fit(X,y)` and `predict(X)` methods. All preprocessing must be inside that model,
fitted on the supplied fold training rows only. AutoGluon and other tools remain
allowed through a fit/predict wrapper. Do not tune on a fold's validation labels,
refit on all data, or fit in the factory. Use random_state/seed 42 for stochastic
models, including wrappers. Do not call another LLM, service or external dataset.

Return a SELF-CONTAINED Python MODULE defining build_model(X_train, y_train)
at module level. The harness writes this generated module to model_factory.py,
then executes the supplied evaluator in your node's conda_env after installation.
Your Bash script must create conda_env with required dependencies (numpy, pandas,
scikit-learn and your selected ML library); it need not train or print scores.
Do NOT run fitting, dataset reading, holdout splitting, prediction, or the
evaluator at module import time. Normal Python initialization (including scorer
construction, logging and computed constants) is allowed. Put fit/predict logic
in the returned model.
Do NOT run a separate holdout/full-data fit. Do NOT generate your own evaluation
or claim a validation score. The harness alone computes the mean of ten fold
scores, injects it into the existing MCTS reward (negative RMSE for regression),
and rejects candidates without completed valid CV exports. Earlier tutorial
examples using holdouts/results.csv do not apply to this benchmark. No final
full-data refit is required. See the mandatory node-level CV contract in prompts.
