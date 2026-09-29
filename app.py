from flask import Flask, request, render_template_string
import pandas as pd
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

DATA_FILE = "churn(1).csv.csv"
DATABASE = "churn_predictions.db"

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_FILE)

df.columns = df.columns.str.strip()

# Convert numeric columns
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df["MonthlyCharges"] = pd.to_numeric(
    df["MonthlyCharges"], errors="coerce"
)

df["tenure"] = pd.to_numeric(
    df["tenure"], errors="coerce"
)

# Target
df["ChurnTarget"] = (
    df["Churn"]
    .astype(str)
    .str.strip()
    .map({
        "Yes": 1,
        "No": 0
    })
)

df = df.dropna(subset=["ChurnTarget"])

df["ChurnTarget"] = df["ChurnTarget"].astype(int)

# =========================================================
# FEATURES
# =========================================================

features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]

X = df[features]
y = df["ChurnTarget"]

numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

# =========================================================
# PREPROCESSING
# =========================================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================================================
# LOGISTIC REGRESSION
# =========================================================

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test, logistic_pred
)

logistic_precision = precision_score(
    y_test, logistic_pred, zero_division=0
)

logistic_recall = recall_score(
    y_test, logistic_pred, zero_division=0
)

logistic_f1 = f1_score(
    y_test, logistic_pred, zero_division=0
)

# =========================================================
# RANDOM FOREST
# =========================================================

random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

random_forest_model.fit(X_train, y_train)

rf_pred = random_forest_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test, rf_pred
)

rf_precision = precision_score(
    y_test, rf_pred, zero_division=0
)

rf_recall = recall_score(
    y_test, rf_pred, zero_division=0
)

rf_f1 = f1_score(
    y_test, rf_pred, zero_division=0
)

# =========================================================
# SELECT BEST MODEL
# =========================================================

if rf_f1 >= logistic_f1:

    best_model = random_forest_model
    best_model_name = "Random Forest"

else:

    best_model = logistic_model
    best_model_name = "Logistic Regression"

# =========================================================
# DATABASE
# =========================================================

def create_database():

    conn = sqlite3.connect(DATABASE)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            gender TEXT,

            senior_citizen TEXT,

            tenure INTEGER,

            contract TEXT,

            payment_method TEXT,

            monthly_charges REAL,

            prediction TEXT,

            probability REAL,

            created_at TIMESTAMP
            DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


create_database()

# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template_string("""

<!DOCTYPE html>

<html>

<head>

<title>Customer Churn Prediction</title>

<meta name="viewport"
content="width=device-width, initial-scale=1">

<link
href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
rel="stylesheet">

<style>

body {
    background:#f4f6f9;
}

.card {
    border:none;
    border-radius:15px;
    box-shadow:0 5px 20px rgba(0,0,0,.08);
}

.header {
    background:#212529;
    color:white;
    padding:25px;
}

</style>

</head>

<body>

<div class="header">

<div class="container">

<h2>Customer Churn Prediction System</h2>

<p class="mb-0">
AI-Based Customer Churn Prediction
</p>

</div>

</div>


<div class="container mt-4">

<div class="card p-4">

<form method="POST"
action="/predict">

<div class="row">

<div class="col-md-4 mb-3">

<label>Gender</label>

<select name="gender"
class="form-select">

<option>Female</option>
<option>Male</option>

</select>

</div>


<div class="col-md-4 mb-3">

<label>Senior Citizen</label>

<select name="senior"
class="form-select">

<option value="0">No</option>
<option value="1">Yes</option>

</select>

</div>


<div class="col-md-4 mb-3">

<label>Partner</label>

<select name="partner"
class="form-select">

<option>Yes</option>
<option>No</option>

</select>

</div>


<div class="col-md-4 mb-3">

<label>Dependents</label>

<select name="dependents"
class="form-select">

<option>Yes</option>
<option>No</option>

</select>

</div>


<div class="col-md-4 mb-3">

<label>Tenure (Months)</label>

<input type="number"
name="tenure"
class="form-control"
value="12"
required>

</div>


<div class="col-md-4 mb-3">

<label>Phone Service</label>

<select name="phone"
class="form-select">

<option>Yes</option>
<option>No</option>

</select>

</div>


<div class="col-md-4 mb-3">

<label>Multiple Lines</label>

<select name="multiple_lines"
class="form-select">

<option>No</option>
<option>Yes</option>
<option>No phone service</option>

</select>

</div>


<div class="col-md-4 mb-3">

<label>Internet Service</label>

<select name="internet"
class="form-select">

<option>DSL</option>
<option>Fiber optic</option>
<option>No</option>

</select>

</div>


<div class="col-md-4 mb-3">

<label>Online Security</label>

<select name="security"
class="form-select">

<option>No</option>
<option>Yes</option>
<option>No internet service</option>

</select>

</div>


<div class="col-md-4 mb-3">

<label>Online Backup</label>

<select name="backup"
class="form-select">

<option>No</option>
<option>Yes</option>
<option>No internet service</option>

</select>

</div>


<div class="col-md-4 mb-3">

<label>Device Protection</label>

<select name="device"
class="form-select">

<option>No</option>
<option>Yes</option>
<option>No internet service</option>

</select>

</div>


<div class="col-md-4 mb-3">

<label>Tech Support</label>

<select name="support"
class="form-select">

<option>No</option>
<option>Yes</option>
<option>No internet service</option>

</select>

</div>


<div class="col-md-4 mb-3">

<label>Streaming TV</label>

<select name="tv"
class="form-select">

<option>No</option>
<option>Yes</option>
<option>No internet service</option>

</select>

</div>


<div class="col-md-4 mb-3">

<label>Streaming Movies</label>

<select name="movies"
class="form-select">

<option>No</option>
<option>Yes</option>
<option>No internet service</option>

</select>

</div>


<div class="col-md-4 mb-3">

<label>Contract</label>

<select name="contract"
class="form-select">

<option>Month-to-month</option>
<option>One year</option>
<option>Two year</option>

</select>

</div>


<div class="col-md-4 mb-3">

<label>Paperless Billing</label>

<select name="paperless"
class="form-select">

<option>Yes</option>
<option>No</option>

</select>

</div>


<div class="col-md-4 mb-3">

<label>Payment Method</label>

<select name="payment"
class="form-select">

<option>Electronic check</option>
<option>Mailed check</option>
<option>Bank transfer (automatic)</option>
<option>Credit card (automatic)</option>

</select>

</div>


<div class="col-md-4 mb-3">

<label>Monthly Charges</label>

<input type="number"
step="0.01"
name="monthly"
class="form-control"
value="70"
required>

</div>


<div class="col-md-4 mb-3">

<label>Total Charges</label>

<input type="number"
step="0.01"
name="total"
class="form-control"
value="840"
required>

</div>

</div>


<div class="text-center">

<button
class="btn btn-dark btn-lg px-5">

Predict Churn

</button>

</div>

</form>

</div>


<div class="text-center mt-4">

<a href="/dashboard"
class="btn btn-outline-dark">

View Analytics Dashboard

</a>

</div>

</div>

</body>

</html>

""")


# =========================================================
# PREDICT
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        gender = request.form["gender"]

        senior = request.form["senior"]

        partner = request.form["partner"]

        dependents = request.form["dependents"]

        tenure = int(request.form["tenure"])

        phone = request.form["phone"]

        multiple_lines = request.form["multiple_lines"]

        internet = request.form["internet"]

        security = request.form["security"]

        backup = request.form["backup"]

        device = request.form["device"]

        support = request.form["support"]

        tv = request.form["tv"]

        movies = request.form["movies"]

        contract = request.form["contract"]

        paperless = request.form["paperless"]

        payment = request.form["payment"]

        monthly = float(request.form["monthly"])

        total = float(request.form["total"])


        input_data = pd.DataFrame({

            "gender": [gender],

            "SeniorCitizen": [senior],

            "Partner": [partner],

            "Dependents": [dependents],

            "tenure": [tenure],

            "PhoneService": [phone],

            "MultipleLines": [multiple_lines],

            "InternetService": [internet],

            "OnlineSecurity": [security],

            "OnlineBackup": [backup],

            "DeviceProtection": [device],

            "TechSupport": [support],

            "StreamingTV": [tv],

            "StreamingMovies": [movies],

            "Contract": [contract],

            "PaperlessBilling": [paperless],

            "PaymentMethod": [payment],

            "MonthlyCharges": [monthly],

            "TotalCharges": [total]

        })


        prediction_value = best_model.predict(
            input_data
        )[0]

        probability = (
            best_model.predict_proba(input_data)[0][1]
            * 100
        )


        if prediction_value == 1:

            result = "Churn"

        else:

            result = "No Churn"


        # Save prediction
        conn = sqlite3.connect(DATABASE)

        conn.execute("""
            INSERT INTO predictions
            (
                gender,
                senior_citizen,
                tenure,
                contract,
                payment_method,
                monthly_charges,
                prediction,
                probability
            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            gender,
            senior,
            tenure,
            contract,
            payment,
            monthly,
            result,
            round(probability, 2)
        ))

        conn.commit()

        conn.close()


        return render_template_string(f"""

<!DOCTYPE html>

<html>

<head>

<title>Prediction Result</title>

<link
href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
rel="stylesheet">

</head>


<body class="bg-light">


<div class="container mt-5">

<div class="card shadow p-5 text-center">

<h1>
Prediction Result
</h1>


<div class="alert
{'alert-danger' if result == 'Churn'
else 'alert-success'} mt-4">

<h2>
{result}
</h2>

<p>
Customer churn probability:
<strong>
{probability:.2f}%
</strong>
</p>

</div>


<h5>
Model Used: {best_model_name}
</h5>


<div class="progress mt-4"
style="height:35px">

<div class="progress-bar
{'bg-danger' if result == 'Churn'
else 'bg-success'}"
style="width:{probability}%">

{probability:.2f}%

</div>

</div>


<div class="mt-4">

<a href="/"
class="btn btn-dark">

New Prediction

</a>

<a href="/dashboard"
class="btn btn-outline-dark">

Dashboard

</a>

</div>

</div>

</div>

</body>

</html>

""")


    except Exception as e:

        return f"""
        <h2>Error</h2>
        <p>{str(e)}</p>
        <a href="/">Go Back</a>
        """, 500


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    conn = sqlite3.connect(DATABASE)

    predictions = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY id DESC",
        conn
    )

    conn.close()


    total = len(predictions)

    churn = len(
        predictions[
            predictions["prediction"] == "Churn"
        ]
    )

    no_churn = len(
        predictions[
            predictions["prediction"] == "No Churn"
        ]
    )


    if total > 0:

        churn_rate = churn / total * 100

    else:

        churn_rate = 0


    return render_template_string(f"""

<!DOCTYPE html>

<html>

<head>

<title>Churn Dashboard</title>

<link
href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
rel="stylesheet">

</head>


<body class="bg-light">


<div class="container mt-5">

<h1>
Churn Analytics Dashboard
</h1>

<p>
Best Model:
<strong>
{best_model_name}
</strong>
</p>


<div class="row mt-4">


<div class="col-md-3">

<div class="card p-4">

<h6>Total Predictions</h6>

<h2>{total}</h2>

</div>

</div>


<div class="col-md-3">

<div class="card p-4">

<h6>Churn</h6>

<h2 class="text-danger">
{churn}
</h2>

</div>

</div>


<div class="col-md-3">

<div class="card p-4">

<h6>No Churn</h6>

<h2 class="text-success">
{no_churn}
</h2>

</div>

</div>


<div class="col-md-3">

<div class="card p-4">

<h6>Churn Rate</h6>

<h2>
{churn_rate:.2f}%
</h2>

</div>

</div>

</div>


<div class="card p-4 mt-4">

<h4>
Model Comparison
</h4>

<table class="table table-bordered">

<tr>

<th>Model</th>
<th>Accuracy</th>
<th>Precision</th>
<th>Recall</th>
<th>F1 Score</th>

</tr>


<tr>

<td>Logistic Regression</td>

<td>
{logistic_accuracy * 100:.2f}%
</td>

<td>
{logistic_precision * 100:.2f}%
</td>

<td>
{logistic_recall * 100:.2f}%
</td>

<td>
{logistic_f1 * 100:.2f}%
</td>

</tr>


<tr>

<td>Random Forest</td>

<td>
{rf_accuracy * 100:.2f}%
</td>

<td>
{rf_precision * 100:.2f}%
</td>

<td>
{rf_recall * 100:.2f}%
</td>

<td>
{rf_f1 * 100:.2f}%
</td>

</tr>

</table>

</div>


<div class="card p-4 mt-4">

<h4>
Prediction History
</h4>

<div class="table-responsive">

<table class="table table-striped">

<tr>

<th>ID</th>
<th>Gender</th>
<th>Tenure</th>
<th>Contract</th>
<th>Payment</th>
<th>Monthly Charges</th>
<th>Prediction</th>
<th>Probability</th>

</tr>

{"".join(
f'''
<tr>
<td>{row["id"]}</td>
<td>{row["gender"]}</td>
<td>{row["tenure"]}</td>
<td>{row["contract"]}</td>
<td>{row["payment_method"]}</td>
<td>₹{row["monthly_charges"]:.2f}</td>
<td>{row["prediction"]}</td>
<td>{row["probability"]:.2f}%</td>
</tr>
'''
for _, row in predictions.iterrows()
)}

</table>

</div>

</div>


<a href="/"
class="btn btn-dark mt-4">

New Prediction

</a>


</div>

</body>

</html>

""")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    print("\n======================================")
    print(" CUSTOMER CHURN PREDICTION SYSTEM")
    print("======================================")

    print(
        f"Total records: {len(df)}"
    )

    print(
        f"Logistic Regression Accuracy: "
        f"{logistic_accuracy * 100:.2f}%"
    )

    print(
        f"Random Forest Accuracy: "
        f"{rf_accuracy * 100:.2f}%"
    )

    print(
        f"Selected Model: {best_model_name}"
    )

    print("======================================")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )