from sklearn.pipeline import Pipeline

from ML_Algos import SVDpp

pipe = Pipeline(
    [
        (
            'model', SVDpp(),
        ),
    ]
)
