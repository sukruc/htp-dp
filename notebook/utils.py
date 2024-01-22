from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
from sklearn import preprocessing, compose, pipeline, model_selection, neural_network, metrics, tree, ensemble, decomposition, cluster, impute, linear_model, neighbors, mixture, feature_selection
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection._split import check_array
import seaborn as sns
from itertools import product
from IPython.display import Markdown, Latex
import matplotlib.lines as mlines
from typing import Dict, Any, Union, TypeVar, Callable
from sklearn import covariance


class ArbitraryStratifiedKFold(model_selection.StratifiedKFold):
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups=None):
        if groups is None:
            groups = y
        groups = check_array(groups, input_name="y", ensure_2d=False, dtype=None)
        return super().split(X, groups, groups)


def getDataset(df):
    df = df.copy()
    df.columns = df.iloc[0].str.strip().str.replace("  ", " ")
    dfc = df.iloc[3:, [3, 4, 6, 7, 8, 35]].copy().astype(float)
    unitless = df.iloc[3:, [20, 24, 26, 28, 30]].copy().astype(float)
    a = pd.DataFrame(dfc.iloc[:, :5].values, columns=pd.MultiIndex.from_product([['Input Raw'], dfc.columns[:5]]))
    b = pd.DataFrame(unitless.values, columns=pd.MultiIndex.from_product([['Input Calculated'], unitless.columns]))
    c = pd.DataFrame(dfc.iloc[:, [-1]].values, columns=pd.MultiIndex.from_product([['Output'], dfc.columns[[-1]]]))
    d = pd.concat([a, b, c], axis=1)
    # compress multiindex
    e = d.copy()
    # e.columns = ['_'.join(col).strip() for col in e.columns.values]
    return e


class LocallyWeightedRegressor(neighbors.KNeighborsRegressor):
    def __init__(self, n_neighbors=2, gamma=None):
        super().__init__(n_neighbors=n_neighbors)
        if gamma is None:
            gamma = 1.0
        self.gamma = gamma

    def predict(self, X):
        y_preds = []
        dist, inds = self.kneighbors(X)
        for i in range(X.shape[0]):
            model_ftp = linear_model.LinearRegression()
            weights = np.exp(-self.gamma * dist[i])
            model_ftp.fit(self._fit_X[inds[i]], self._y[inds[i]], sample_weight=weights)
            y_preds.append(model_ftp.predict(X[i:i+1]))
        res = np.array(y_preds).squeeze()
        if res.ndim == 0:
            res = res.reshape(-1, 1)
        return res
    
    
def pearson(y_true, y_pred):
    def ms(x):
        return x - x.mean()
    a = ms(y_true)
    b = ms(y_pred)
    s = (a * b).sum()**2 / (a**2).sum() / (b**2).sum()
    return float(np.array(s).squeeze())


def bias(y_true, y_pred):
    return (-y_true + y_pred).sum() / y_true.sum()

def renameScores(x):
    return x.replace("test_", "Test ").replace("train_", "Train ")


NAME_KEYS = {
    "Mass flux": "$G$",
    "Saturation pressure": "$P_{sat}$",
    "Heat flux": "$q$",
    "Quality": "$x$",
    "Pressure drop": "$\Delta P_{sat}$",
    "Reynolds number": "$Re_{t}$",
    "Two-phase multiplier": "$X_{tt}$",
    "Froude number": "$Fr_{l}$",
    "Weber number": "$We_{L}$",
    "Bond number": "$Bo$",
    "Heat transfer coefficient": "$h_{TP}$"
}

H_NAMES = pd.Series({
        0: 'Mass flux',
        1: 'Saturation pressure',
        2: 'Heat flux',
        3: 'Quality',
        4: 'Pressure drop',
#         4: 'Heat transfer coefficient',
        5: 'Reynolds number',
        6: 'Two-phase multiplier',
        7: 'Froude number',
        8: 'Weber number',
        9: 'Bond number',
        10: "Heat transfer coefficient",
        11: "Tube type",
    })

DP_NAMES = H_NAMES.copy()
DP_NAMES[4] = H_NAMES[10]
DP_NAMES[10] = H_NAMES[4]

def standardizePred(y_pred: Union[pd.DataFrame, np.ndarray, pd.Series]):
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values[:, -1]
    elif isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    elif isinstance(y_pred, np.ndarray):
        y_pred = y_pred.squeeze()
    else:
        raise TypeError("y_pred must be a pandas DataFrame, Series or numpy array.")
    return y_pred


def renameH(df):
    df = df.copy()
    df.columns = H_NAMES
    return df

def renameDP(df):
    df = df.copy()
    df.columns = DP_NAMES
    return df

def getDecPipeline():
    return pipeline.Pipeline([
        ("impute", impute.SimpleImputer()),
        ('scale', preprocessing.StandardScaler()),
        ('pca', decomposition.PCA(n_components=2)),
    ])


def getClusterer():
    return mixture.GaussianMixture(n_components=3, n_init=50, init_params='k-means++', max_iter=1000, tol=1e-6, )

def process_ann_results(cv, config_col_name='Selected ANN config', attributes=('hidden_layer_sizes', 'activation')):
    tablo_ann = (
        pd.DataFrame(cv)
        .sort_values(by='mean_test_R2')
        .set_index('params')
        .rename_axis(index=config_col_name)
        # .rename(lambda x: ((n:=x.best_estimator_[-1].regressor_).hidden_layer_sizes ,n.activation))
        .rename(lambda x: tuple((" ".join(k.split("__")[-2:]), v) for k, v in x.items()))
    )
    tablo_ann = tablo_ann.drop([c for c in tablo_ann.columns if any([i in c for i in ['std_', 'rank_', 'split', 'param_']])], axis=1)
    tablo_ann = tablo_ann.rename(lambda x: x.replace("mean_", ""), axis=1)
    

    topten = tablo_ann[
        [col for col in tablo_ann.columns if 'test' in col]
        + [col for col in tablo_ann.columns if 'train' in col]
    ].sort_values('test_Pearson-R', ascending=False).reset_index().rename(renameScores , axis=1)

    ort_skor = tablo_ann.describe().pipe(lambda x: x[[c for c in x if "test" in c]]) \
        .T.drop("count", axis=1).round(4).rename(renameScores) \
        .style.set_caption("Scoring statistics for different models")

    return tablo_ann, topten, ort_skor

SCORING={
    "R2": metrics.make_scorer(metrics.r2_score),
    "RMSE": metrics.make_scorer(metrics.mean_squared_error, greater_is_better=False, squared=False),
    "MAE": metrics.make_scorer(metrics.mean_absolute_error, greater_is_better=False),
    "MAPE": metrics.make_scorer(metrics.mean_absolute_percentage_error, greater_is_better=False),
    "Bias%": metrics.make_scorer(bias, greater_is_better=False),
    "WAPE%": metrics.make_scorer(lambda x, y: metrics.mean_absolute_percentage_error(x, y, sample_weight=y), greater_is_better=False),
    "Pearson-R": metrics.make_scorer(pearson)
}

scoring = SCORING.copy()


T = TypeVar('T')
Prediction = Union[pd.DataFrame, np.ndarray, pd.Series]

class Dataset:
    def __init__(self, data, X, y):
        self.data = data
        self.X = X
        self.y = y
        self.data_train: pd.DataFrame = None
        self.data_test: pd.DataFrame = None
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.y_test: pd.Series = None
        self.c_train: pd.Series = None
        self.c_test: pd.Series = None
        self.sample_category: pd.Series = None
        self.cats = None
        self.decpipeline: pipeline.Pipeline = None
        self.clusterer: pipeline.Pipeline = None
        self.clusters: pd.Series = None
        self.scores: Dict[str, Any] = None
        self.models: Dict[str, pipeline.Pipeline] = {}
        self.models_cv: Dict[str, Dict[Any, Any]] = {}
        self.renameFunc = lambda x: x
        self.y_pred: Dict[str, Prediction] = {}
        self.y_pred_train: Dict[str, Prediction] = {}
        self.y_pred_test: Dict[str, Prediction] = {}
        self.scoring: Dict[str, Any] = {}
        
    def setDec(self, decpipeline):
        self.decpipeline = decpipeline
        return self
    
    def setClusterer(self, clusterer):
        self.clusterer = clusterer
        return self
    
    def decompose(self):
        self.Xdec = self.decpipeline.fit_transform(self.X, self.y)
        return self

    def setCategories(self):
        self.clusters = pd.Series(self.clusterer.fit_predict(self.Xdec[:, :2]))

    def split(self, test_size=0.2, random_state=None):
        self.data_train, self.data_test, self.X_train, self.X_test, self.y_train, self.y_test, self.c_train, self.c_test = \
            model_selection.train_test_split(self.data, self.X, self.y, self.clusters, test_size=test_size, random_state=random_state, stratify=self.sample_category)
        self.y_train = pd.Series(self.y_train.iloc[:, -1], index=self.y_train.index, name="Output")
        self.y_test = pd.Series(self.y_test.iloc[:, -1], index=self.y_test.index, name="Output")
        return self
    
    def displayEDA(self, stylized=True):
        data = self.data.groupby('dataset').agg(['mean']).T \
            .assign(Deviation=lambda x: x.diff(axis=1).iloc[:,1].abs() / x.iloc[:,1]) \
            .assign(Variable=self.col_names.drop(11).tolist())
        
        if stylized:
            return data.style.background_gradient(subset=['Deviation'], cmap='Reds') \
                .set_caption("Table: Input deviation statistics by tube type")
        
        else:
            return data
        
    @property
    def symbolized(self):
        return self.data.pipe(self.renameFunc).rename(NAME_KEYS, axis=1)
        
    def displayPairs(self, target_var, row_len=5):
        data = self.symbolized
        xcols = set(data.columns) - set([NAME_KEYS[target_var]]) - set(['Tube type'])
        ycols = set([NAME_KEYS[target_var]])

        # In each iteration, only draw pairplot for row_len variables from xcols

        for i in range(0, len(xcols), row_len):
            sns.pairplot(data, x_vars=list(xcols)[i:i+row_len], y_vars=ycols, hue='Tube type')
            plt.show()

    def displayHistogram(self, target_var):
        data = self.symbolized

        sns.displot(data, x=NAME_KEYS[target_var], hue='Tube type', kind='kde', fill=True)


            


    def defineCats(self):
        self.sample_category = self.data['dataset'] + " c:" + pd.Series(self.clusters, index=self.data.index).astype(str)
        self.cats = self.data_train['dataset'] + "_" + pd.Series(self.c_train, index=self.data_train.index).map({0: 'A', 1: 'B', 2: 'C'}).astype('category').cat.codes.astype(str)
        return self

    def getPartitionedDataByDataset(self):
        return [self.data['dataset'] == tip  for tip in self.data['dataset'].unique()]
    
    def setRenameFunc(self, f):
        self.renameFunc = f
        return self
    
    def displayStatsByCat(self):
        return self.data_train.pipe(self.renameFunc).assign(Category=self.cats).groupby("Category").mean() \
            .style \
            .background_gradient(axis=0, cmap='Greens') \
            .set_caption("Input variable statistics by sample category")
    

    def setColNames(self, col_names, copy=True):
        """Col names 10 and 11 are reserved for output and data type."""
        if copy:
            self.col_names = col_names.copy()
        else:
            self.col_names = col_names
        return self

    def addModel(self, model: pipeline.Pipeline, name: str, scoring: Dict[str, Any] = None):
        scoring = scoring or self.scoring
        self.models_cv[name] = self.getCv(model, scoring)
        self.models[name] = model.fit(self.X_train, self.y_train)
        self.y_pred[name] = pd.Series(standardizePred(self.models[name].predict(self.X_test)), index=self.y_test.index, name="Prediction")
        self.y_pred_train[name] = pd.Series(standardizePred(self.models[name].predict(self.X_train)), index=self.y_train.index, name="Prediction")
        return self

    def getCv(self, model, scoring=None):
        model.fit(self.X_train, self.y_train)
        return model.cv_results_
        # cv = model_selection.cross_validate(
        #     model,
        #     self.X_train,
        #     self.y_train,
        #     groups=self.c_train,
        #     cv=ArbitraryStratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        #     scoring=scoring or self.scoring,
        #     return_train_score=True,
        #     n_jobs=-1,
        #     verbose=2,
        #     return_estimator=True
        # )
        return cv
    
    def setScoring(self, scoring):
        self.scoring = scoring
        return self

    def displayModelCvStats(self, name, config_col_name='Selected ANN config', attributes=('hidden_layer_sizes', 'activation')):
        return process_ann_results(self.models_cv[name], config_col_name=name, attributes=attributes)
    
    def displayDecompositionResults(self, stylized=True):
        data = pd.DataFrame(self.decpipeline[-1].components_.T, columns=['PCA1', 'PCA2'], index=self.col_names.drop([10, 11]))

        if stylized:
            return data.style.background_gradient(cmap='jet', vmin=-1, vmax=1) \
                .set_caption("Projection axes by input variables")
        else:
            return data

    def displayAllModelTestResults(self, scoring: Dict[str, Any] = None):
        scoring = scoring or self.scoring
        if self.scores is None:
            self.scores = {}
        for name in self.models.keys():
            for k, v in scoring.items():
                dkt = self.scores.setdefault(name, {})
                dkt[k] = v._score_func(self.y_test, self.y_pred[name], **v._kwargs)
        return pd.DataFrame(self.scores)
    
    def visualizeClusters(self, fil1='Plain tube h', fil2='Microfin tube h'):
        f1 = self.data['dataset'] == fil1
        f2 = self.data['dataset'] == fil2
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot()
        ax.scatter(self.Xdec[f1, 0], self.Xdec[f1, 1], 
            c=self.clusters[f1], 
            # color='red',
            cmap='viridis', linewidth=0.5, marker='^', edgecolors='black')
        ax.scatter(self.Xdec[f2, 0], self.Xdec[f2, 1], 
                c=self.clusters[f2], 
                #    color='blue', 
                cmap='viridis', linewidth=0.5, marker='o', edgecolors='black')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.title("Clusters based on principal components")

        # add legend
        duz_boru = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                                    markersize=10, label='Plain Tube')
        mikrokanatli_boru = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                                    markersize=10, label='Finned Tube')
        ax.legend(handles=[duz_boru, mikrokanatli_boru]);
        plt.grid()
        plt.show()

        # repeat plot for h colors

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot()
        ax.scatter(self.Xdec[f1, 0], self.Xdec[f1, 1], 
            # c=clusters[f1], 
            color='red',
            cmap='viridis', linewidth=0.5, marker='^', edgecolors='black')
        ax.scatter(self.Xdec[f2, 0], self.Xdec[f2, 1], 
                #    c=clusters[f2], 
                color='blue', 
                cmap='viridis', linewidth=0.5, marker='o', edgecolors='black')
        # plt.title("Finned vs plain tube in pricincipal component space")
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.grid()
        # add legend
        duz_boru = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                                    markersize=10, label='Plain Tube')
        mikrokanatli_boru = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                                    markersize=10, label='Finned Tube')
        ax.legend(handles=[duz_boru, mikrokanatli_boru])
        plt.show()

    def plot_predictions(self, target_label='h_{TP}'):
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        markers = ['o', 's', 'D', 'v', '^', 'p', '*', 'h']
        

        data_h = self.data
        data_test = self.data_test
        y_test = self.y_test
        ypreds = []
        model_names = []
        for name, pred in self.y_pred.items():
            ypreds.append(pred)
            model_names.append(name)

        dataset_types = data_h['dataset'].unique()
        num_datasets = len(dataset_types)
        
        fig, axs = plt.subplots(nrows=num_datasets, ncols=len(ypreds), figsize=(20, 8*num_datasets), sharey=True)
        
        for row, dataset_type in enumerate(dataset_types):
            f = data_test['dataset'] == dataset_type
            
            for col, ypred in enumerate(ypreds):
                axs[row, col].scatter(y_test[f], ypred[f], color=colors[col], marker=markers[col], label='Prediction', alpha=0.5)
            
                xlims = tuple(axs[row, col].get_xlim())
                ylims = tuple(axs[row, col].get_ylim())

                ori = y_test[f].iloc[0]

                axs[row, col].grid()
                axs[row, col].axline((ori, ori), slope=1, label='x=y')
                axs[row, col].axline((ori, ori*1.1), slope=1.1, color='black', linestyle=':', label="+-10%")
                axs[row, col].axline((ori, ori*1.3), slope=1.3, color='gray', linestyle='--', label="+-30%")
                axs[row, col].axline((ori, ori*0.9), slope=0.9, color='black', linestyle=':')
                axs[row, col].axline((ori, ori*0.7), slope=0.7, color='gray', linestyle='--')
                if col == 0:
                    axs[row, col].legend()
                    axs[row, col].set_ylabel(f"$\hat{{{target_label}}}$")
                axs[row, col].set_xlabel(f"${target_label}$")
                axs[row, col].set_xlim(xlims)
                axs[row, col].set_ylim(ylims)
                axs[row, col].axis("equal")
                axs[row, col].set_title(f"{model_names[col]} - {dataset_type}")  # Set model name as title
        
            # fig.suptitle(f"Dataset Type: {dataset_type}", fontsize=16, y=0.92)
        
        plt.tight_layout()


class OutlierDataset(Dataset):
    def split(self, test_size=0.2, random_state=None):
        h_gm = mixture.GaussianMixture(n_components=3, n_init=100, init_params='k-means++', max_iter=1000, tol=1e-6, )
        h_pca = pipeline.Pipeline([
            ('scaler', impute.SimpleImputer(strategy='mean')),
            ('imputer', preprocessing.StandardScaler()),
            ('pca', decomposition.PCA(n_components=3))
        ])
        h_cluster_pipeline = pipeline.Pipeline([
            ("pca", h_pca),
            ("gm", h_gm)
        ])

        Xh = self.X
        yh = self.y
        data_h = self.data

        gm_h = h_cluster_pipeline.fit_predict(Xh)
        pca_h = h_cluster_pipeline[:1].fit_transform(Xh)[:, :2]

        outlier_indices = []
        for c in range(3):
            f = Xh[gm_h == c]
            zz = h_pca[:2].transform(f)
            maho = covariance.LedoitWolf().fit(zz).mahalanobis(zz)
            outlier_indices += pd.Series(maho, index=f.index).sort_values(ascending=False).head(10).index.tolist()

        self.X_train = Xh.drop(outlier_indices)
        self.X_test = Xh.loc[outlier_indices]
        self.y_train = yh.drop(outlier_indices)
        self.y_test = yh.loc[outlier_indices]
        self.data_train = data_h.drop(outlier_indices)
        self.data_test = data_h.loc[outlier_indices]
        self.outlier_indices = outlier_indices


        self.y_train = pd.Series(self.y_train.iloc[:, -1], index=self.y_train.index, name="Output")
        self.y_test = pd.Series(self.y_test.iloc[:, -1], index=self.y_test.index, name="Output")

        self.h_cluster_pipeline = h_cluster_pipeline
        return self
    
    def visualizeOutliers(self):
        plt.figure(figsize=(12, 6))
        plt.scatter(*self.h_cluster_pipeline[:1].transform(self.X_train)[:, :2].T, label='Training samples')
        plt.scatter(*self.h_cluster_pipeline[:1].transform(self.X_test)[:, :2].T, color='red', label='Extrapolation samples')
        plt.xlabel("$PCA_1$")
        plt.ylabel("$PCA_2$")
        plt.grid()
        plt.legend()
        plt.title("Input variables in PCA space");



def getANN():
    olcekli_ann = compose.TransformedTargetRegressor(
        regressor=neural_network.MLPRegressor(hidden_layer_sizes=(40, 10), max_iter=4000, random_state=42),
        transformer=preprocessing.StandardScaler()
    )

    union = pipeline.FeatureUnion([
        ("log", preprocessing.FunctionTransformer(func=np.log1p, inverse_func=np.expm1))
    ])

    comp = compose.ColumnTransformer([
        ], remainder='drop')


    pipe = pipeline.Pipeline([
        ("subselect", preprocessing.FunctionTransformer(lambda x: x['Input Raw'])),
        ('imputer', impute.SimpleImputer(strategy='mean')),
        ("rescale", preprocessing.StandardScaler()),
        ('mlp', olcekli_ann)
    ])
    gs = model_selection.GridSearchCV(
        pipe,
        param_grid={
            'mlp__regressor__hidden_layer_sizes': [
                (20, 20),
                (10, 10),
                (128, 64, 32, 16, 8,),
                (50, 50), 
                ],
            'mlp__regressor__activation': [
                'relu',
                'logistic'
            ],
            'mlp__regressor__tol': [1e-5],
        },
        cv=model_selection.KFold(n_splits=3, shuffle=True, random_state=42),
        scoring=scoring,
        refit="R2",
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    return gs

def getLwr():
    ctlocal = compose.TransformedTargetRegressor(regressor=LocallyWeightedRegressor(n_neighbors=15, gamma=.1), 
                                                # func=np.log1p, 
                                                func=lambda x: x, 
                                                #  inverse_func=np.expm1
                                                inverse_func=lambda x: x
                                                )

    union = pipeline.FeatureUnion([
        ("log", preprocessing.FunctionTransformer(func=np.log1p, inverse_func=np.expm1))
    ])


    local_pipe = pipeline.Pipeline([
        ("subselect", preprocessing.FunctionTransformer(lambda x: x['Input Raw'])),
        # ("comp", comp),
        ('imputer', impute.SimpleImputer(strategy='mean')),
        ('ink', union),
        ("rescale", preprocessing.StandardScaler()),
        ('mlp', ctlocal)
    ])
    local_pipe_gs = model_selection.GridSearchCV(
    estimator=local_pipe,
    param_grid={
        'mlp__regressor__n_neighbors': [5, 10, 15, 20],
#         'mlp__regressor__n_neighbors': [1],
        'mlp__regressor__gamma': [.7, 1, 2, 3]
    },
    scoring=scoring,
    refit='R2',
    cv=3,
)
    return local_pipe_gs

def getGBM():
    union = pipeline.FeatureUnion([
        ("log", preprocessing.FunctionTransformer(func=np.log1p, inverse_func=np.expm1))
    ])
    lgbm_pipe = pipeline.Pipeline([
        ('imputer', impute.SimpleImputer(strategy='mean')),
        ('ink', union),
        ("rescale", preprocessing.StandardScaler()),
        ('mlp', compose.TransformedTargetRegressor(regressor=lgb.LGBMRegressor(), func=lambda x : x, inverse_func=lambda x: x))
    ])

    lgbm_pipe_gs = model_selection.GridSearchCV(
        estimator=lgbm_pipe,
        param_grid={
            'mlp__regressor__reg_alpha': [0, 0.1, 0.5, 1, 2, 3, 4, 5, 10],
        },
        scoring=scoring,
        refit='R2',
        cv=3,
        n_jobs=-1,
        verbose=2,
    )
    return lgbm_pipe_gs

def getDummy():
    dummy_pipe = pipeline.Pipeline([
        ('imputer', impute.SimpleImputer(strategy='mean')),
        ("rescale", preprocessing.StandardScaler()),
        ('mlp', compose.TransformedTargetRegressor(regressor=linear_model.LinearRegression(), func=lambda x : x, inverse_func=lambda x: x))
    ])

    dummy_pipe_gs = model_selection.GridSearchCV(
        estimator=dummy_pipe,
        param_grid={
            "mlp__regressor__fit_intercept": [True, False],
        },
        scoring=scoring,
        refit='R2',
        cv=3,
        n_jobs=-1,
        verbose=2,
    )
    return dummy_pipe_gs