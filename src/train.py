"""Grouped holdout and grouped CV; no test-driven hyperparameter selection."""
import hashlib
import json
import platform
import sys
from importlib.metadata import version
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import ROOT, FEATURES, TARGET, SEED

def metrics(y, p):
    return {'mae': float(mean_absolute_error(y, p)),
            'rmse': float(np.sqrt(mean_squared_error(y, p))),
            'r2': float(r2_score(y, p))}

def main():
    source = ROOT / 'data/concrete.csv'
    raw = pd.read_csv(source)
    assert list(raw.columns) == FEATURES + [TARGET]
    assert np.isfinite(raw.to_numpy()).all()
    # Identical full rows may be replicated database entries. Keep the first.
    df = raw.drop_duplicates().copy()
    df['source_row'] = df.index
    df = df.reset_index(drop=True)
    X, y = df[FEATURES], df[TARGET]
    # Same seven quantities, irrespective of age, define one exact mix group.
    groups = pd.factorize(pd.MultiIndex.from_frame(X[FEATURES[:7]]), sort=True)[0]
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.20,
                                        random_state=SEED).split(X, y, groups))
    assert set(groups[train]).isdisjoint(groups[test])
    Xtr, Xte, ytr, yte = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    cv = list(GroupKFold(n_splits=5).split(Xtr, ytr, groups[train]))
    for a, b in cv:
        assert set(groups[train][a]).isdisjoint(groups[train][b])
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    models_dir = ROOT / 'models'
    models_dir.mkdir(exist_ok=True)
    models = {'Mean baseline': DummyRegressor(strategy='mean'),
              'Linear regression': make_pipeline(StandardScaler(), LinearRegression())}
    rows, fitted = [], {}
    for name, model in models.items():
        scores = -cross_val_score(model, Xtr, ytr, cv=cv,
                                 scoring='neg_mean_absolute_error')
        fitted[name] = clone(model).fit(Xtr, ytr)
        rows.append({'model': name, 'cv_mae_mean': float(scores.mean()),
                     'cv_mae_std': float(scores.std()),
                     **metrics(yte, fitted[name].predict(Xte))})
    search = GridSearchCV(
        RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=2),
        {'max_depth': [None, 10], 'min_samples_leaf': [1, 3, 5],
         'max_features': [1.0, 0.7]},
        scoring='neg_mean_absolute_error', cv=cv, n_jobs=1, refit=True)
    search.fit(Xtr, ytr)
    fitted['Random forest'] = search.best_estimator_
    best = search.best_index_
    rows.append({'model': 'Random forest', 'cv_mae_mean': float(-search.best_score_),
                 'cv_mae_std': float(search.cv_results_['std_test_score'][best]),
                 **metrics(yte, search.predict(Xte))})
    results = pd.DataFrame(rows)
    results.to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(search.cv_results_).to_csv(out / 'rf_cv_search.csv', index=False)
    selected = min(rows, key=lambda r: r['cv_mae_mean'])['model']
    model = fitted[selected]
    joblib.dump(model, models_dir / 'model.joblib', compress=3)
    metadata = {'selected_model': selected, 'features': FEATURES,
                'training_min': Xtr.min().to_dict(), 'training_max': Xtr.max().to_dict()}
    (models_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    assignment = df[['source_row']].copy()
    assignment['mix_group'] = groups
    assignment['split'] = 'train'
    assignment.loc[test, 'split'] = 'test'
    assignment['cv_fold'] = -1
    for fold, (_, valid) in enumerate(cv):
        assignment.loc[train[valid], 'cv_fold'] = fold
    assignment.to_csv(out / 'split_assignments.csv', index=False)
    pred = df.iloc[test][['source_row'] + FEATURES + [TARGET]].copy()
    for name, m in fitted.items():
        pred[name] = m.predict(Xte)
    pred.to_csv(out / 'test_predictions.csv', index=False)
    examples = ROOT / 'examples'
    examples.mkdir(exist_ok=True)
    Xte.iloc[:5].to_csv(examples / 'inputs.csv', index=False)
    yte.iloc[:5].to_csv(examples / 'expected_strengths.csv', index=False)
    # The saved model remains fitted to training data only.
    assert np.allclose(joblib.load(models_dir / 'model.joblib').predict(Xte), model.predict(Xte))
    summary = {'seed': SEED, 'raw_rows': len(raw), 'duplicates_removed': len(raw)-len(df),
               'rows': len(df), 'mix_groups': int(len(np.unique(groups))),
               'train_rows': len(train), 'test_rows': len(test),
               'train_groups': int(len(np.unique(groups[train]))),
               'test_groups': int(len(np.unique(groups[test]))),
               'group_overlap': 0, 'rf_best_params': search.best_params_,
               'selected_model': selected,
               'dataset_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'python': sys.version, 'platform': platform.platform(),
               'packages': {p: version(p) for p in ['numpy','pandas','scipy','scikit-learn',
                                                  'matplotlib','joblib','xlrd']},
               'metrics': rows}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    for ax, name, color in zip(axes, ['Linear regression','Random forest'], ['#64748b','#087f8c']):
        p = fitted[name].predict(Xte)
        ax.scatter(yte, p, s=16, alpha=.65, color=color, edgecolors='none')
        lim = [min(0, p.min()-2), max(yte.max(), p.max())+3]
        ax.plot(lim, lim, '--', c='#333333', lw=1)
        ax.set(xlim=lim, ylim=lim, xlabel='Measured strength (MPa)',
               ylabel='Predicted strength (MPa)', title=name)
        ax.grid(alpha=.15)
    fig.savefig(out / 'predictions.png', dpi=220)
    fig.savefig(out / 'predictions.pdf')
    plt.close(fig)
    print(results.to_string(index=False))
    print(json.dumps({k: summary[k] for k in ['rows','mix_groups','train_rows','test_rows',
                                             'rf_best_params','selected_model']}, indent=2))

if __name__ == '__main__':
    main()
