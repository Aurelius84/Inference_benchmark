import os


def get_model_path(model_name, model_type='static'):
    """
    Return the saved model dir name
    """
    infer_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(infer_dir, "models", model_type, model_name)
    return model_path
