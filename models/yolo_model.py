import onnxruntime as ort

class YOLOModel:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def predict(self, input_tensor):
        return self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})
