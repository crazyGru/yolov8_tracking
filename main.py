from roboflow import Roboflow
rf = Roboflow(api_key="nECPBxrHa2AFNSOxLPla")
project = rf.workspace().project("engine-detector")
model = project.version(3).model

# infer on a local image
print(model.predict("temp.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("temp.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())