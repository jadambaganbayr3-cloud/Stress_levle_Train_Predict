import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("Stress_level_3.csv")

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("?", "", regex=False)
    .str.replace(".", "", regex=False)
)

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

X = df.drop("stress_level", axis=1)
y = df["stress_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

df["predicted_stress_level_code"] = model.predict(X)

stress_map = {
    0: "Low",
    1: "Medium",
    2: "High"
}

df["predicted_stress_level"] = df["predicted_stress_level_code"].map(stress_map)

df.to_csv("stress_predictions_with_label.csv", index=False)
print("Saved: stress_predictions_with_label.csv")

def predict_new_student(model, columns, student_data):
    """
    model        : trained model
    columns      : feature column names
    student_data : dict (new student info)
    """

    # dict → DataFrame
    student_df = pd.DataFrame([student_data])

    # column дарааллыг тааруулах
    student_df = student_df.reindex(columns=columns, fill_value=0)

    # prediction
    pred_code = model.predict(student_df)[0]

    # mapping
    stress_map = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    return pred_code, stress_map[pred_code]


# =========================
# 11. PREDICT NEW STUDENT
# =========================

new_student = {
    "age": 24,
    "gender": 1,   
    "Which_year_are_you_studying?": 4,
    "Are_you_enjoying_your_usual_activities_and_interests?": 3,
    "Do_you_sleep_well_and_wake_up_feeling_rested?": 2,
    "Do_you_have_enough_energy_to_get_through_the_day?": 2,
    "do_you_eat_regularly_and_maintain_a_balanced_appetite": 3
}

code, label = predict_new_student(
    model,
    X.columns,
    new_student
)

print("New student stress prediction:")
print("Code:", code)
print("Level:", label)
