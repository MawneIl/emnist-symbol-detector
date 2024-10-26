import os
import pickle

class Model:
    def __init__(self):
        model_path = os.path.join('myapp', 'model.pkl')
        # your code here

    def predict(self, x):
        '''
        Parameters
        ----------
        x : np.ndarray
            Входное изображение -- массив размера (28, 28)
        Returns
        -------
        pred : str
            Символ-предсказание 
        '''
        # your code here
        with open(os.path.join('myapp', 'labels_dict.pkl'),'rb') as f:
            labels_dict = pickle.load(f)

        with open(os.path.join('myapp', 'model.pkl'),'rb') as f:
            model = pickle.load(f)

        x = x.reshape(1,-1)
        pred = model.predict(x)

        return labels_dict[int(pred)]