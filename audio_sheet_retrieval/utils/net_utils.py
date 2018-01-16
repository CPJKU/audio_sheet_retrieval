
import pickle


def load_net_params(file_path):
    """ Load model parameters.
        We wrote this function to circumvent the python2 / python3
        pickle compatibility issues.
    """

    try:
        with open(file_path, 'rb') as fp:
            params = pickle.load(fp)

    except:
        with open(file_path, 'rb') as fp:
            params = pickle.load(fp)

    return params
