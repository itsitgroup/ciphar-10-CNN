from keras.applications.densenet import DenseNet121
from keras.layers import Dense
from keras.models import Sequential

# Define a function to create a DenseNet model
def create_densenet_model(input_shape, num_classes):
    base_model = DenseNet121(include_top=False, input_shape=input_shape, pooling='avg')
    model = Sequential()
    model.add(base_model)
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Example usage
densenet_model = create_densenet_model((32, 32, 3), 10)
densenet_model.summary()

# Visualization of the ANN (if necessary)
# !pip3 install ann_visualizer
# !pip3 install graphviz
# from ann_visualizer.visualize import ann_viz
# ann_viz(densenet_model, title="My ANN")
