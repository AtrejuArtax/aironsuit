import numpy as np
from hyperopt import Trials, STATUS_OK, STATUS_FAIL
import hyperopt
from sklearn import metrics
import pandas as pd
import pickle
import math
import os
from sklearn.metrics import accuracy_score

from aironsuit.backend import get_backend
from aironsuit.utils import load_model, save_model, clear_session
from aironsuit.callbacks import get_basic_callbacks
from aironsuit.trainers import airon_trainer

BACKEND = get_backend()


class AIronSuit(object):

    def __init__(self, model_constructor, trainer=None):

        self.__model = None
        self.__model_constructor = model_constructor
        self.__trainer = airon_trainer if not trainer else trainer
        self.__cuda = None
        self.__devices = None
        self.__parallel_models = None
        self.__total_n_models = None

    def create(self, specs, n_parallel_models=1, devices=None, cuda=None):

        self.__cuda = cuda
        self.__devices = devices if devices else []
        self.__total_n_models = n_parallel_models * len(self.__devices)
        self.__model = self.__model_constructor(**specs)
        if self.__cuda in specs and BACKEND != 'tensorflow':
            self.__model.cuda()

    def explore(self, x_train, y_train, x_val, y_val, space, model_specs, train_specs, path, max_evals, epochs,
                metric=None, trials=None, net_name='NN', verbose=0, seed=None, val_inference_in_path=None,
                callbacks=None, n_parallel_models=1, devices=None, cuda=None):

        self.__cuda = cuda
        self.__devices = devices if devices else []
        self.__total_n_models = n_parallel_models * len(self.__devices)
        if trials is None:
            trials = Trials()

        def objective(space):

            # Create model
            specs = space.copy()
            specs.update(model_specs)
            # previous kargs: specs=specs, net_name=net_name,
            #                                    metrics=metric if metric is not None else specs['loss']
            model = self.__model_constructor(**specs)
            if self.__cuda in specs and BACKEND != 'tensorflow':
                model.cuda()

            # Print some information
            iteration = len(trials.losses())
            if verbose > 0:
                print('\n')
                print('iteration : {}'.format(0 if trials.losses() is None else iteration))
                [print('{}: {}'.format(key, value)) for key, value in specs.items()]
                print(model.summary(line_length=200))

            # Train model
            trainer_kargs = train_specs.copy()
            trainer_kargs.update({'module': model})
            if callbacks:
                trainer_kargs.update({'callbacks': callbacks})
            trainer = self.__trainer(**trainer_kargs)
            trainer.fit(x_train, y_train, epochs=epochs)

            # Exploration loss
            exp_loss = None
            if metric in [None, 'categorical_accuracy', 'accuracy']:
                exp_loss = accuracy_score(y_val, trainer.predict(x_val))
                # exp_loss = model.evaluate(x=x_val, y=y_val, verbose=verbose)
                if isinstance(exp_loss, list):
                    exp_loss = sum(exp_loss)
                exp_loss /= self.__total_n_models
                if metric == 'categorical_accuracy':
                    exp_loss = 1 - exp_loss
            elif metric == 'i_auc':
                y_pred = model.predict(x_val)
                if not isinstance(y_pred, list):
                    y_pred = [y_pred]
                exp_loss = []
                for i in np.arange(0, self.__total_n_models):
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
                self.__save_model(model=model, name=path + 'best_exp_' + net_name + '_json')
                for dict_, name in zip([specs, space], ['_specs', '_hparams']):
                    with open(path + 'best_exp_' + net_name + name, 'wb') as f:
                        pickle.dump(dict_, f, protocol=pickle.HIGHEST_PROTOCOL)
                if val_inference_in_path is not None:
                    y_val_ = np.concatenate(y_val, axis=1) if isinstance(y_val, list) else y_val
                    np.savetxt(val_inference_in_path + 'val_target.csv', y_val_, delimiter=',')
                    y_inf = trainer.predict(x_val)
                    y_inf = np.concatenate(y_inf, axis=1) if isinstance(y_inf, list) else y_inf
                    np.savetxt(val_inference_in_path + 'val_target_inference.csv', y_inf, delimiter=',')

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
            best_model = self.__load_model(name=path + 'best_exp_' + net_name + '_json')
            if BACKEND == 'tensorflow':
                best_model.compile(optimizer=specs['optimizer'], loss=specs['loss'])
            else:
                best_model.cuda()

            print('best hyperparameters: ' + str(best_hparams))

            return best_model

        self.__model = optimize()

    def train(self, x_train, y_train, train_specs, batch_size=30, x_val=None, y_val=None,
              path=None, verbose=0, callbacks=None):

        # Train model
        self.__trainer(
            self.__model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            train_specs=train_specs,
            mode='training',
            path=path,
            callbacks=callbacks,
            verbose=verbose,
            batch_size=batch_size)

    def inference(self, x_pred):
        return self.__model.predict(x_pred)

    def evaluate(self, x, y):
        return self.__model.evaluate(x=x, y=y)

    def save_model(self, name):
        self.__save_model(model=self.__model, name=name)

    def load_model(self, name):
        self.__model = load_model(name)

    def clear_session(self):
        clear_session()

    def compile(self, loss, optimizer, metrics=None):
        self.__model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=metrics)

    def __save_model(self, model, name):
        save_model(model=model, name=name)

    def __load_model(self, name):
        return load_model(name=name)
