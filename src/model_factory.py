from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def get_model(name: str, params: dict):
    if name == "random_forest":
        return RandomForestRegressor(**params)
    elif name == "gradient_boosting":
        return GradientBoostingRegressor(**params)
    elif name == "linear":
        return LinearRegression(**params)
    elif name == "xgboost":
        return XGBRegressor(**params)
    elif name == "lightgbm":
        return LGBMRegressor(**params)
    else:
        raise ValueError(f"Modle {name} is not supported.")
