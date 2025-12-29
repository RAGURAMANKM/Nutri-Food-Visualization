import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATASET_PATH =r'C:\Users\ragur\OneDrive\Desktop\DA_PROJECT\nutrifood.py\synthetic_food_dataset_imbalanced.csv'
TARGET_COLUMN = "Meal_Type"
RANDOM_STATE = 42



print("\nLoading dataset...")
df = pd.read_csv(DATASET_PATH)

print("\nDataset Preview:")
print(df.head())

print("\nDataset Information:")
print(df.info())



# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())



print("\nClass Distribution:")
print(df[TARGET_COLUMN].value_counts())



X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "bool"]).columns


numeric_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95))   # retain 95% variance
])

categorical_pipeline = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)



models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
}

accuracy_results = {}
confusion_matrices = {}



for name, model in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracy_results[name] = acc
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)

    cv_score = cross_val_score(pipeline, X_train, y_train, cv=5).mean()

    print(f"Accuracy: {acc:.4f}")
    print(f"Cross-Validation Accuracy: {cv_score:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))



fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()

for ax, (name, cm) in zip(axes, confusion_matrices.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.suptitle("Confusion Matrices of All Models", fontsize=18)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracy_results.keys()), y=list(accuracy_results.values()))
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()



best_model = max(accuracy_results, key=accuracy_results.get)

print("\nBest Performing Model:")
print(best_model)
print("Accuracy:", accuracy_results[best_model])
