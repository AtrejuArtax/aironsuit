import numpy as np
from hyperopt import Trials, STATUS_OK, STATUS_FAIL
import hyperopt
from sklearn import metrics
import pandas as pd
import pickle
import math
import os
import glob

from aironsuit.utils import load_json, clear_session
from aironsuit.callbacks import get_basic_callbacks


class AIronSuit(object):

    def __init__(self, net_constructor, net_constructor_specs):

        self.__model = None
        self.__parallel_models = None
        self.__device = None
        self.__net_constructor = net_constructor
        self.__net_constructor_specs = net_constructor_specs

    def create(self, specs, metrics=None, net_name='NN'):
        self.__model = self.__net_constructor(specs=specs, metrics=metrics, net_name=net_name)

    def explore(self, x_train, y_train, x_val, y_val, space, model_specs, exploration_specs, path, max_evals,
                tensor_board=False, metric=None, trials=None, net_name='NN', verbose=0, seed=None,
                val_inference_in_path=None, callbacks=None):

        self.__parallel_models = exploration_specs['parallel_models']
        self.__device = exploration_specs['device']
        if trials is None:
            trials = Trials()

        def objective(space):

            # Create model
            specs = space.copy()
            specs.update(model_specs)
            specs.update(exploration_specs)
            # previous kargs: specs=specs, net_name=net_name,
            #                                    metrics=metric if metric is not None else specs['loss']
            model = self.__net_constructor(specs)

            # Print some information
            iteration = len(trials.losses())
            if verbose > 0:
                print('\n')
                print('iteration : {}'.format(0 if trials.losses() is None else iteration))
                [print('{}: {}'.format(key, value)) for key, value in specs.items()]
                print(model.summary(line_length=200))

            # Train model
            self.__train(x_train=x_train,
                         y_train=y_train,
                         x_val=x_val,
                         y_val=y_val,
                         model=model,
                         train_specs=exploration_specs,
                         mode='exploration',
                         path=path,
                         callbacks=callbacks,
                         verbose=verbose,
                         tensor_board=tensor_board,
                         batch_size=specs['batch_size'],
                         ext=iteration)

            # Exploration loss
            total_n_models = self.__parallel_models * len(self.__device)
            exp_loss = None
            if metric in [None, 'categorical_accuracy']:
                exp_loss = model.evaluate(x=x_val, y=y_val, verbose=verbose)
                if isinstance(exp_loss, list):
                    exp_loss = sum(exp_loss)
                exp_loss /= total_n_models
                if metric == 'categorical_accuracy':
                    exp_loss = 1 - exp_loss
            elif metric == 'i_auc':
                y_pred = model.predict(x_val)
                if not isinstance(y_pred, list):
                    y_pred = [y_pred]
                exp_loss = []
                for i in np.arange(0, total_n_models):
                    if len(np.bincount(y_val[i][:,-1])) > 1 and not math.isnan(np.sum(y_pred[i])):
                        fpr, tpr, thresholds = metrics.roc_curve(y_val[i][:, -1], y_pred[i][:, -1])
                        exp_loss += [(1 - metrics.auc(fpr, tpr))]
                exp_loss = np.mean(exp_loss) if len(exp_loss) > 0 else 1
            if verbose > 0:
                print('\n')
                print('Exploration Loss: ', exp_loss)
            status = STATUS_OK if not math.isnan(exp_loss) and exp_loss is not None else STATUS_FAIL

            # Save trials
            with open(path + 'trials.hyperopt', 'wb') as f:
                pickle.dump(trials, f)

            # Save model if it is the best so far
            best_exp_losss_name = path + 'best_' + net_name + '_exp_loss'
            best_exp_loss = None \
                if not os.path.isfile(best_exp_losss_name) else pd.read_pickle(best_exp_losss_name).values[0][0]
            print('best val loss so far: ', best_exp_loss)
            print('curren val loss: ', exp_loss)
            best_exp_loss_cond = best_exp_loss is None or exp_loss < best_exp_loss
            print('save: ', status, best_exp_loss_cond)
            if status == STATUS_OK and best_exp_loss_cond:
                df = pd.DataFrame(data=[exp_loss], columns=['best_exp_loss'])
                df.to_pickle(best_exp_losss_name)
                self.__save_json(filepath=path + 'best_exp_' + net_name + '_json', model=model)
                self.__save_weights(filepath=path + 'best_exp_' + net_name + '_weights', model=model)
                for dict_, name in zip([specs, space], ['_specs', '_hparams']):
                    with open(path + 'best_exp_' + net_name + name, 'wb') as f:
                        pickle.dump(dict_, f, protocol=pickle.HIGHEST_PROTOCOL)
                if val_inference_in_path is not None:

                    np.savetxt(val_inference_in_path + 'val_target.csv', np.concatenate(y_val, axis=1), delimiter=',')
                    y_inf = model.predict(x_val)
                    y_inf = y_inf if isinstance(y_inf, list) else [y_inf]
                    np.savetxt(val_inference_in_path + 'val_target_inference.csv',
                               np.concatenate(y_inf, axis=1), delimiter=',')

            clear_session()
            del model

            return {'loss': exp_loss, 'status': status}

        def optimize():

            if len(trials.trials) < max_evals:
                hyperopt.fmin(
                    objective,
                    rstate=None if seed is None else np.random.RandomState(seed),
                    space=space,
                    algo=hyperopt.tpe.suggest,
                    max_evals=max_evals,
                    trials=trials,
                    verbose=True,
                    return_argmin=False)
            with open(path + 'best_exp_' + net_name + '_hparams', 'rb') as f:
                best_hparams = pickle.load(f)
            with open(path + 'best_exp_' + net_name + '_specs', 'rb') as f:
                specs = pickle.load(f)
            best_model = self.__load_json(filepath=path + 'best_exp_' + net_name + '_json')
            best_model.load_weights(filepath=path + 'best_exp_' + net_name + '_weights')
            best_model.compile(optimizer=specs['optimizer'], loss=specs['loss'])

            print('best hyperparameters: ' + str(best_hparams))

            return best_model

        self.__model = optimize()

    def __train(self, x_train, y_train, x_val, y_val, model, train_specs, mode, path,
                verbose, tensor_board, batch_size, ext=None, callbacks=None):

        best_model_name = path + 'best_epoch_model_' + mode

        # Callbacks
        callbacks_ = []
        if callbacks:
            for callback_dict in callbacks:
                callbacks_ += [callback_dict['callback'](callbacks_dict['kargs'])]
            best_model_files = glob.glob(best_model_name + '*')
            if len(best_model_files) > 0:
                for filename in glob.glob(best_model_name + '*'):
                    os.remove(filename)

        # Train model
        class_weight = None if 'class_weight' not in train_specs.keys() \
            else {output_name: train_specs['class_weight'] for output_name in model.output_names}
        kargs = {'x': x_train,
                 'y': y_train,
                 'epochs': train_specs['epochs'],
                 'callbacks': callbacks_list,
                 'class_weight': class_weight,
                 'shuffle': True,
                 'verbose': verbose,
                 'batch_size': batch_size}
        if not any([val_ is None for val_ in [x_val, y_val]]):
            kargs.update({'validation_data': (x_val, y_val)})
        model.fit(**kargs)

        # Best model
        if callbacks:
            best_model_files = glob.glob(best_model_name + '*')
            if len(best_model_files) > 0:
                model.load_weights(filepath=best_model_name)
                for filename in glob.glob(best_model_name + '*'):
                    os.remove(filename)


    def train(self, x_train, y_train, train_specs, batch_size=30, x_val=None, y_val=None,
              path=None, verbose=0, tensor_board=False, callbacks=None):

        # Train model
        self.__train(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            model=self.__model,
            train_specs=train_specs,
            mode='training',
            path=path,
            callbacks=callbacks,
            verbose=verbose,
            tensor_board=tensor_board,
            batch_size=batch_size)

    def inference(self, x_pred):
        return self.__model.predict(x_pred)

    def evaluate(self, x, y):
        return self.__model.evaluate(x=x, y=y)

    def save_weights(self, filepath):
        self.__save_weights(filepath=filepath, model=self.__model)

    def __save_weights(self, filepath, model):
        model.save_weights(filepath=filepath)

    def load_weights(self, filepath):
        self.__load_weights(filepath=filepath, model=self.__model)

    def __load_weights(self, filepath, model):
        model.load_weights(filepath=filepath)

    def get_weights(self):
        return self.__model.get_weights()

    def set_weights(self, weights):
        self.__set_weights(weights=weights, model=self.__model)

    def __set_weights(self, weights, model):
        model.set_weights(weights=weights)

    def save_json(self, filepath):
        self.__save_json(filepath=filepath, model=self.__model)

    def __save_json(self, filepath, model):
        with open(filepath, "w") as json_file:
            json_file.write(model.to_json())

    def load_json(self, filepath):
        self.__model = __load_json(filepath)

    def clear_session(self):
        clear_session()

    def compile(self, loss, optimizer, metrics=None):
        self.__model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=metrics)
