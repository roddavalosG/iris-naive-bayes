import sklearn
import matplotlib as plt
import seaborn as sns
import prettytable


#cargamos el dataset de iris
iris = sklearn.datasets.load_iris()
x = iris.data #atributos
y = iris.target #etiquetas

#dividimos el train y el test
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.3,random_state=42) #semilla para reproducibilidad

#creamos el clasificador Naive Bayes Gaussiano porque las variables son continuas 
nb_model = sklearn.naive_bayes.GaussianNB()
nb_model.fit(x_train,y_train)

#evaluar el modelo
y_pred = nb_model.predict(x_test)
precision = sklearn.metrics.accuracy_score(y_test,y_pred)
print(f"Precision: {precision}")

# Crear la tabla
table = prettytable.PrettyTable()
table.field_names = ["Clase", "Precision", "Recall", "F1-Score", "Soporte"]
reporte = sklearn.metrics.classification_report(y_test,y_pred,target_names=iris.target_names,output_dict=True)

for class_name, metrics in reporte.items():
    if isinstance(metrics, dict):
        table.add_row([
            class_name,
            f"{metrics['precision']:.2f}",
            f"{metrics['recall']:.2f}",
            f"{metrics['f1-score']:.2f}",
            f"{metrics['support']:.0f}"
        ])

print(table)
