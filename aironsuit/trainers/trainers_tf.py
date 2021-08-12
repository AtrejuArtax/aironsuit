import os
import glob


class AIronTrainer(object):

    def __init__(self, module, **kwargs):
        self.__module = module
        available_kargs = ['callbacks', 'mode', 'class_weight', 'path', 'batch_size']
        for wkarg in kwargs.keys():
            assert wkarg in available_kargs
        self.__callbacks = kwargs['callbacks'] if 'callbacks' in kwargs else None
        self.__mode = kwargs['mode'] if 'mode' in kwargs else None
        self.__class_weight = kwargs['class_weight'] if 'class_weight' in kwargs else None
        self.__path = kwargs['path'] if 'path' in kwargs else None
        self.__batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 32

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=30, verbose=0):

        class_weight = {output_name: self.__class_weight for output_name in self.__module.output_names} \
            if self.__class_weight else None

        best_model_name = None

        # Callbacks
        callbacks_ = []
        if isinstance(self.__callbacks, list):
            for callback_dict in self.__callbacks:
                callback_name = list(callback_dict.keys())[0]
                callback_dict_ = callback_dict[callback_name]
                if callback_name == 'ModelCheckpoint':
                    ext = '_' + self.__mode if self.__mode else ''
                    best_model_name = self.__path + 'best_epoch_model' + ext
                callbacks_ += [callback_dict_['callback'](**callback_dict_['kargs'])]
            best_model_files = glob.glob(best_model_name + '*')
            if len(best_model_files) > 0:
                for filename in glob.glob(best_model_name + '*'):
                    os.remove(filename)

        # Train model
        kargs = {'x': x_train,
                 'y': y_train,
                 'epochs': epochs,
                 'callbacks': callbacks_,
                 'class_weight': class_weight,
                 'shuffle': True,
                 'verbose': verbose,
                 'batch_size': self.__batch_size}
        if not any([val_ is None for val_ in [x_val, y_val]]):
            kargs.update({'validation_data': (x_val, y_val)})
        self.__module.fit(**kargs)

        # Best model
        if self.__callbacks:
            best_model_files = glob.glob(best_model_name + '*')
            if len(best_model_files) > 0:
                self.__module.load_weights(filepath=best_model_name)
                for filename in glob.glob(best_model_name + '*'):
                    os.remove(filename)

    def predict(self, x):
        return self.__module.predict(x)
