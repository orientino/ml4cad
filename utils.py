from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


def get_preprocess_std_num(feat_names):
    """Preprocess only the numerical features."""

    def update_num_feats(x):
        if x in feat_names:
            return feat_names.index(x)

    feat_names_num = ["Age", "FE", "Creatinina"]
    num_feat_index = list(map(update_num_feats, feat_names_num))
    num_feat_index = [x for x in num_feat_index if x is not None]
    preprocess_std_num = ColumnTransformer(
                                transformers = [('stand', StandardScaler(), num_feat_index)], 
                                remainder="passthrough"
                            )
    return preprocess_std_num
