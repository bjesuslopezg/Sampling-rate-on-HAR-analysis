from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

def train_model(features, model_name):
    X = features.drop(columns=["activity", "userid"]).astype("float32").values
    y = features["activity"].values
    g = features["userid"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=g, random_state=42)

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_te_enc = le.transform(y_te)

    if model_name == "random_forest":
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    elif model_name == "xgboost":
        clf = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=len(le.classes_),
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss"
        )
    elif model_name == "gbc":
        clf = HistGradientBoostingClassifier(learning_rate=0.1, max_depth=5, random_state=42)
    else:
        raise ValueError(f"Model '{model_name}' not recognized")

    clf.fit(X_tr, y_tr_enc)
    y_pred_enc = clf.predict(X_te)
    y_pred = le.inverse_transform(y_pred_enc)
    print(classification_report(y_te, y_pred))
    print(f"Model '{model_name}' trained successfully with {len(le.classes_)} classes")