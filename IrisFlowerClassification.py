''' 
Iris Flower Classification - CodeAlpha Internship
Name: Siyolise Mbadu
Task: Train a KNN model to classify Iris Species
'''

#--Imports--
import os
from sklearn.preprocessing import StandardScaler 
import pandas as pd 
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_score


#--Create output folders upfront so nothing fails later trying to save into them--
os.makedirs("results", exist_ok = True)
os.makedirs("images", exist_ok= True)


print("\n", "="* 50)
print("1. DATA LOADING")
print("=" * 50)

#--Load dataset from CSV.file--
df = pd.read_csv(r"C:\Users\User\Desktop\Internship\Iris.csv")
print("Data laoded successfully")


print("\n", "="* 50)
print("2. DATA EXPLORATION")
print("=" * 50)

#--Basic overview of the dataset: shape,stats, structure--
print("First 5 rows:")
print(df.head())
print("\n dataFrame shape:")
print(df.shape)
print("\n statistics:")
print(df.describe())
print("\n dataset info:")
print(df.info())


print("\n", "="* 50)
print("3. DATA CLEANING")
print("="*50)

#--Check for missing values--
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print(f"Missing values found: {missing_values}")
    print("Deleting the columns or rows that contain missing values")
else:
    print("No missing values Found")
    
print("\n data distribution:")
print(df['Species'].value_counts())

#--Check for and remove duplicate rows--
duplicates_no = df.duplicated().sum()
if duplicates_no > 0:
    print("Removing duplicates")
    df = df.drop_duplicates()


print("\n", "="* 50)
print("4. DATA VISUALIZATION")
print("="*50)

#--Scatter plot: petal length vs width, coloured by species--
plt.figure(figsize=(8,6))
plt.scatter(
    df[df["Species"] == "Iris-setosa"]["PetalLengthCm"],
    df[df["Species"] == "Iris-setosa"]["PetalWidthCm"],
    label = "Iris-setosa",
)

plt.scatter(
    df[df["Species"] == "Iris-versicolor"]["PetalLengthCm"],
    df[df["Species"] == "Iris-versicolor"]["PetalWidthCm"],
    label = "Iris-versicolor",
)

plt.scatter(
    df[df["Species"] == "Iris-virginica"]["PetalLengthCm"],
    df[df["Species"] == "Iris-virginica"]["PetalWidthCm"],
    label = "Iris-virginica",
)

plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.title("Iris Flower Classification")
plt.legend()
plt.savefig("images/petal_scatter.png", dpi=150, bbox_inches="tight") #save BEFORE show
plt.show()

#--Feature/target split--
# Drop Species (target) and Id (just a row identifier, not a real feature)
X = df.drop(["Species", "Id"], axis = 1, errors="ignore")
y = df["Species"]

#--Train/test split--
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42,
)

#--Feature scaling--
#KNN is distance-based, so features must be on the same scale.
#Fit the scaler on training data only, then apply it to test data.
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("\n", "="* 50)
print("5. BASELINE MODEL (K=3)")
print("="*50)

#--Train a baseline KNN model with an arbitrary starting K--
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train_scaled, y_train)
print("Model trained successfully.")

predictions = model.predict(x_test_scaled)
print(f"Predictions: {predictions}")
print("\nActual values: ", f"{y_test.values}")

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100}%") 


#--Confusion Matrix for the baseline--
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title ("Confusion Matrix")
plt.xlabel ("Predicted Species")
plt.ylabel ("Actual Species")

plt.xticks([0,1,2],["Setosa","Versicolor","Virginica"])
plt.yticks([0,1,2],["Setosa","Versicolor","Virginica"])

plt.savefig("images/Confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n", "="* 50)
print("6. FINDING THE BEST K (Cross-Validation)")
print("="*50)

#--Test K values 1-20 using 5-fold cross_validation--
#Cross-validation gives a more reliable accuracy estimate than a single split.
k_range = range(1,21)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring="accuracy")
    cv_scores.append(scores.mean())

best_k = k_range[np.argmax(cv_scores)]
print(f"Besk K: {best_k}, CV Accuracy: {max(cv_scores)*100:.2f}%")

#--Elbow plot: accuracy across all tested K values--
plt.figure(figsize=(8,5))
plt.plot(k_range, cv_scores, marker="o")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Cross-Validated Accuracy")
plt.title("KNN: Accuracy vs K")
plt.grid(True)
plt.savefig("images/Accuracy_vs_K.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n", "="* 50)
print("7. FINAL MODEL (Best K)")
print("="*50)

#--Train the final model using the best K found above--
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(x_train_scaled, y_train)

#--Save the trained model and scaler for reuse without retraining--
# Both must be saved together: new data needs the SAME scaling 
# transformating the training data went through.
joblib.dump(final_model, "results/knn_model.joblib")
joblib.dump(scaler, "results/scaler.joblib")
print("Model and scaler saved to results")

final_predictions = final_model.predict(x_test_scaled)
final_accuracy = accuracy_score(y_test, final_predictions)
print(f"Final model accuracy with K={best_k}: {final_accuracy*100:.2f}%")

print("\n", "="* 50)
print("8. CLASSIFICATION REPORT + SAVING RESULTS")
print("="*50)

#--Generate and print the classification report--
report = classification_report(y_test, final_predictions)
print(report)

#--Save report as plain text (readable, good for pasting into a README)--
with open("classification_report.txt", "w") as f:
    f.write(report)

#--Save report as CSV (structured, good for reuse/analysis/tables)--
report_dict = classification_report(y_test, final_predictions, output_dict = True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv("results/classification_report.csv")

#--Auto-generate a ready-to-paste Results section for the README
# Pulls every value directly from the results above, so its always accurate.
species_rows = ""
for species in ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]:
    row = report_dict[species]
    species_rows += (
        f"| {species} | {row['precision']:.2f} | {row['recall']:.2f} "
        f"| {row['f1-score']:.2f} | {int(row['support'])} |\n"
    )
    
results_block = f"""## Results
- **Best K:** {best_k}
- **Cross-validated accuracy:** {max(cv_scores)*100:.2f}%
- **Final test accuracy:** {final_accuracy*100:.2f}%
    
| Species | Precision | Recall | F1-score | Support |
|---------|-----------|--------|----------|---------|
{species_rows}
Full report: [results/classification_report.csv](results/classification_report.csv) . [results/classification_report.txt](results/classification_report.txt)
"""
    
with open("results/results_section.md", "w") as f:
    f.write(results_block)
    
print("\nResults section written to results/results_section.md - copy this your README.")
    