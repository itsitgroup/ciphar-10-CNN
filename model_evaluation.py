import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

# Evaluate the model
evaluation = model.evaluate(X_test, y_cat_test)
print(f'Test Accuracy : {evaluation[1] * 100:.2f}%')

# Predict the labels
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Print classification report
print(classification_report(y_test, y_pred))

# Visualize an example prediction
my_image = X_test[100]
plt.imshow(my_image)
plt.title(f"True label: {labels[y_test[100][0]]}")
plt.show()

# Utility function to plot image
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel(f"{labels[predicted_label]} {100 * np.max(predictions_array):2.0f}% ({labels[true_label]})", color=color)

# Plot predictions
predictions = model.predict(X_test)
num_rows = 8
plt.figure(figsize=(2 * 2 * num_rows, 2 * num_rows))
for i in range(num_rows):
    plt.subplot(num_rows, 2, 2 * i + 1)
    plot_image(i, predictions[i], y_test, X_test)
plt.show()
