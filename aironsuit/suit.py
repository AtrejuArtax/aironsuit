import os
import warnings
import numpy as np
import pandas as pd
import pickle
import math
import tempfile
from inspect import getfullargspec
import hyperopt
from hyperopt import Trials, STATUS_OK, STATUS_FAIL
from sklearn.metrics import accuracy_score, roc_curve, auc
from aironsuit.trainers import AIronTrainer
from aironsuit.callbacks import init_callbacks, get_basic_callbacks
from aironsuit.backend import get_backend
from aironsuit.design.utils import setup_design_logs, update_design_logs
from airontools.constructors.utils import Model, get_latent_model
from airontools.visualization import save_representations
from airontools.interactors import load_model, save_model, clear_session, summary
from airontools.tools import path_management
BACKEND = get_backend()


class AIronSuit(object):
    """ AIronSuit is a model wrapper that takes care of the hyper-parameter optimization problem, training and inference
    among other functionalities.

        Attributes:
            model (Model): NN model.
            latent_model (Model): Latent NN model.
            results_path (str): Results path.
            logs_path (int): Logs path.
            __model_constructor (): NN model constructor.
            __trainer (object): NN model constructor instance.
            __trainer_class (AIronTrainer): NN model trainer.
            __cuda (bool): Whether to use cuda or not.
            __devices (list): Devices where to make the computations.
            __total_n_models (int): Total number of models in parallel.
    """

    def __init__(self,
                 model_constructor=None,
                 model=None,
                 results_path=os.path.join(tempfile.gettempdir(), 'airon') + os.sep,
                 logs_path=None,
                 trainer=None,
                 model_constructor_wrapper=None,
                 custom_objects=None,
                 force_subclass_weights_saver=False,
                 force_subclass_weights_loader=False,
                 ):
        """ Parameters:
                model_constructor (): Function that returns a model.
                model (Model): User customized model.
                results_path (str): Results path.
                logs_path (str): Logs path.
                trainer (): Model trainer.
                model_constructor_wrapper (): Model constructor wrapper.
                custom_objects (dict): Custom objects when loading Keras models.
                force_subclass_weights_saver (bool): To whether force the subclass weights saver or not, useful for
                keras subclasses models.
                force_subclass_weights_loader (bool): To whether force the subclass weights loader or not, useful for
                keras subclasses models.
        """

        self.model = model
        self.latent_model = None
        self.results_path = results_path
        self.logs_path = logs_path if logs_path is not None else os.path.join(results_path, 'logs')
        self.__model_constructor = model_constructor
        self.__trainer = None
        self.__trainer_class = AIronTrainer if not trainer else trainer
        self.__model_constructor_wrapper = model_constructor_wrapper
        self.__custom_objects = custom_objects
        self.__cuda = None
        self.__devices = None
        self.__total_n_models = None
        self.__force_subclass_weights_saver = force_subclass_weights_saver
        self.__force_subclass_weights_loader = force_subclass_weights_loader

    def design(self,
               x_train,
               x_val,
               hyper_space,
               train_specs,
               max_evals,
               epochs,
               y_train=None,
               y_val=None,
               model_specs=None,
               results_path=None,
               logs_path=None,
               metric=None,
               trials=None,
               name='NN',
               verbose=0,
               seed=None,
               raw_callbacks=None,
               cuda=None,
               use_basic_callbacks=True,
               patience=3,
               save_val_inference=False,
               ):
        """ Automatic model design.

            Parameters:
                x_train (list, np.array): Input data for training.
                x_val (list, np.array): Input data for validation.
                hyper_space (dict): Hyper parameter space for model design.
                train_specs (dict): Training specifications.
                results_path (str): Results path.
                logs_path (str): Logs path.
                max_evals (integer): Maximum number of evaluations.
                epochs (int): Number of epochs for model training.
                y_train (list, np.array): Output data for training.
                y_val (list, np.array): Output data for validation.
                model_specs (dict): Model specifications.
                metric (str): Metric to be used for model design. If None validation loss is used.
                trials (Trials): Object with design information.
                name (str): Name of the model.
                verbose (int): Verbosity.
                seed (int): Seed for reproducible results.
                raw_callbacks (list): Dictionary of raw callbacks.
                cuda (bool): Whether cuda is available or not.
                use_basic_callbacks (bool): Whether to use basic callbacks or not. Callbacks argument has preference.
                patience (int): Patience in epochs for validation los improvement, only active when use_basic_callbacks.
                save_val_inference (bool): Whether or not to save validation inference when the best model is found.
        """

        method_r_path = self.__manage_path(results_path, path_ext='design')
        method_l_path = self.__manage_path(logs_path, path_type='logs')

        setup_design_logs(method_l_path, hyper_space)

        self.__cuda = cuda
        if trials is None:
            trials = Trials()
        raw_callbacks = raw_callbacks if raw_callbacks else \
            get_basic_callbacks(
                path=method_r_path,
                patience=patience,
                name=name,
                verbose=verbose,
                epochs=epochs
            ) if use_basic_callbacks else None

        def design_trial(hyper_candidates):

            # Create model
            specs = hyper_candidates.copy()
            if model_specs:
                specs.update(model_specs)
            self.__create(**specs)

            # Print some information
            iteration = len(trials.losses())
            if verbose > 0:
                print('\n')
                print('iteration : {}'.format(0 if trials.losses() is None else iteration))
                [print('{}: {}'.format(key, value)) for key, value in specs.items()]
                print(self.model.summary(line_length=200))

            # Train model
            trainer = self.__train(
                train_specs=train_specs,
                model=self.model,
                epochs=epochs,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                callbacks=init_callbacks(raw_callbacks) if raw_callbacks else None,
                verbose=verbose
            )

            # Design loss
            # ToDo: compatible with custom metric
            if metric in ['categorical_accuracy', 'accuracy']:
                def prepare_for_acc(x):
                    if not isinstance(x, list):
                        x_ = [x]
                    else:
                        x_ = x.copy()
                    for i in range(len(x_)):
                        if len(x_[i].shape) == 1:
                            x_[i] = np.where(x_[i] > 0.5, 1, 0)
                        else:
                            x_[i] = np.argmax(x_[i], axis=-1)
                    return x_
                y_pred = prepare_for_acc(trainer.predict(x_val))
                y_val_ = prepare_for_acc(y_val)
                acc_score = []
                for i in range(len(y_pred)):
                    acc_score.append(accuracy_score(y_pred[i],  y_val_[i]))
                design_loss = 1 - np.mean(acc_score)
            elif metric == 'i_auc':  # ToDo: make this work
                y_pred = self.model.predict(x_val)
                if not isinstance(y_pred, list):
                    y_pred = [y_pred]
                design_loss = []
                for i in np.arange(0, self.__total_n_models):
                    if len(np.bincount(y_val[i][:, -1])) > 1 and not math.isnan(np.sum(y_pred[i])):
                        fpr, tpr, thresholds = roc_curve(y_val[i][:, -1], y_pred[i][:, -1])
                        design_loss += [(1 - auc(fpr, tpr))]
                design_loss = np.mean(design_loss) if len(design_loss) > 0 else 1
            else:
                if y_val is not None:
                    design_loss = self.model.evaluate(x_val, y_val)
                else:
                    design_loss = self.model.evaluate(x_val)
                if isinstance(design_loss, list) or isinstance(design_loss, tuple):
                    design_loss = design_loss[0]
                if isinstance(design_loss, dict):
                    design_loss = [loss for _, loss in design_loss.items()][0]

            if verbose > 0:
                print('\n')
                print('Design Loss: ', design_loss)
            status = STATUS_OK if not math.isnan(design_loss) and design_loss is not None else STATUS_FAIL

            # Update logs
            if status == STATUS_OK:
                update_design_logs(
                    path=os.path.join(method_l_path, str(len(trials.losses()))),
                    hparams={value['logs']: hyper_candidates[key] for key, value in hyper_space.items()},
                    value=design_loss
                )

            # Save trials
            with open(os.path.join(method_r_path, 'trials.hyperopt'), 'wb') as f:
                pickle.dump(trials, f)

            # Save model if it is the best so far
            best_design_loss_name = os.path.join(method_r_path, '_'.join(['best', name, 'design_loss']))
            trials_losses = [loss_ for loss_ in trials.losses() if loss_]
            best_design_loss = min(trials_losses) if len(trials_losses) > 0 else None
            print('best val loss so far: ' + str(best_design_loss))
            print('current val loss: ' + str(design_loss))
            best_design_loss_cond = best_design_loss is None or design_loss < best_design_loss
            save_cond = status == STATUS_OK and best_design_loss_cond
            print('save: ' + str(save_cond))
            if save_cond:
                df = pd.DataFrame(data=[design_loss], columns=['best_design_loss'])
                df.to_pickle(best_design_loss_name)
                self.__save_load_model(name=os.path.join(method_r_path, '_'.join(['best_design', name])), mode='save')
                with open(os.path.join(method_r_path, '_'.join(['best_design', name, 'hyper_candidates'])), 'wb') as f:
                    pickle.dump(hyper_candidates, f, protocol=pickle.HIGHEST_PROTOCOL)
                if save_val_inference and y_val is not None:
                    y_inf = trainer.predict(x_val)
                    y_inf = np.concatenate(y_inf, axis=1) if isinstance(y_inf, list) else y_inf
                    np.savetxt(os.path.join('inference', 'val_target_inference.csv'), y_inf, delimiter=',')

            clear_session()
            del self.model

            return {'loss': design_loss, 'status': status}

        def design():

            if len(trials.trials) < max_evals:
                hyperopt.fmin(
                    design_trial,
                    rstate=None if seed is None else np.random.default_rng(seed),
                    space={key: value['options'] for key, value in hyper_space.items()},
                    algo=hyperopt.tpe.suggest,
                    max_evals=max_evals,
                    trials=trials,
                    verbose=True,
                    return_argmin=False
                )
            with open(os.path.join(method_r_path, 'best_design_' + name + '_hyper_candidates'), 'rb') as f:
                best_hyper_candidates = pickle.load(f)

            # Best model
            specs = {}
            if model_specs:
                specs.update(model_specs.copy())
            specs.update(best_hyper_candidates)
            best_model = self.__save_load_model(
                name=os.path.join(method_r_path, '_'.join(['best_design', name])),
                mode='load',
                **{key: value for key, value in specs.items() if key != 'name'}
            )
            if BACKEND == 'tensorflow' and \
                    all([spec_ in specs.keys() for spec_ in ['optimizer', 'loss']]):
                compile_kwargs = {
                    'optimizer': specs['optimizer'],
                    'loss': specs['loss']
                }
                if 'metrics' in specs.keys():
                    compile_kwargs['metrics'] = specs['metrics']
                best_model.compile(**compile_kwargs)
            elif cuda:
                best_model.cuda()
            print('best hyper-parameters: ' + str(best_hyper_candidates))

            # Trainer
            trainer_kwargs = train_specs.copy()
            trainer_kwargs.update({'module': best_model})
            if raw_callbacks:
                trainer_kwargs.update({'callbacks': init_callbacks(raw_callbacks)})
            trainer = self.__trainer_class(**trainer_kwargs)
            if hasattr(trainer, 'initialize') and callable(trainer.initialize):
                trainer.initialize()

            return best_model, trainer

        self.model, self.__trainer = design()

    def train(self,
              epochs,
              x_train,
              y_train,
              x_val=None,
              y_val=None,
              batch_size=32,
              callbacks=None,
              verbose=None,
              use_basic_callbacks=True,
              results_path=None,
              logs_path=None,
              name='NN',
              patience=3
              ):
        """ Weight optimization.

            Parameters:
                epochs (int): Number of epochs for model training.
                x_train (list, np.array): Input data for training.
                y_train (list, np.array): Output data for training.
                x_val (list, np.array): Input data for validation.
                y_val (list, np.array): Output data for validation.
                batch_size (int): Batch size.
                callbacks (dict): Dictionary of callbacks.
                verbose (int): Verbosity.
                use_basic_callbacks (bool): Whether to use basic callbacks or not. Callbacks argument has preference.
                results_path (str): Results path.
                logs_path (str): Logs path.
                name (str): Name of the model.
                patience (int): Patience in epochs for validation los improvement, only active when use_basic_callbacks.
        """
        method_r_path = self.__manage_path(results_path, path_ext='train')
        method_l_path = self.__manage_path(logs_path, path_type='logs')
        train_specs = {
            'batch_size': batch_size,
            'path': method_r_path}
        callbacks_ = callbacks if callbacks else \
            get_basic_callbacks(
                path=method_r_path,
                patience=patience,
                name=name,
                verbose=verbose,
                epochs=epochs
            ) if use_basic_callbacks else None
        self.__trainer = self.__train(
                train_specs=train_specs,
                model=self.model,
                epochs=epochs,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                callbacks=callbacks_,
                verbose=verbose
        )

    def inference(self, x, use_trainer=False):
        """ Inference.

            Parameters:
                x (list, np.array): Input data for training.
                use_trainer (bool): Whether to use the current trainer or not.
        """
        return self.__get_model_interactor(use_trainer).predict(x)
    
    def latent_inference(self, x, layer_names=None):
        """ Latent inference.

            Parameters:
                x (list, np.array): Input data for training.
                layer_names (str): Layer names.
        """
        assert all([var is not None for var in [layer_names, self.latent_model]])
        if layer_names:
            self.latent_model = get_latent_model(self.model, layer_names)
        return self.latent_model.predict(x)

    def create_latent_model(self, hidden_layer_names):
        """ Create latent model given a model and hidden layer names.

            Parameters:
                hidden_layer_names (str): Layer names.
        """
        assert self.model is not None
        self.latent_model = get_latent_model(self.model, hidden_layer_names)

    def evaluate(self, x, y=None, use_trainer=False):
        """ Evaluate.

            Parameters:
                x (list, np.array): Input data for evaluation.
                y (list, np.array): Target data for evaluation.
                use_trainer (bool): Whether to use the current trainer or not.
        """
        args = [x]
        if y is not None:
            args += [y]
        return self.__get_model_interactor(use_trainer).evaluate(*args)

    def save_model(self, name):
        """ Save the model.

            Parameters:
                name (str): Model name.
        """
        self.__save_load_model(name=name, mode='save')

    def load_model(self, name, **kwargs):
        """ Load the model.

            Parameters:
                name (str): Model name.
                kwargs (dict): Custom or other arguments.
        """
        self.model = self.__save_load_model(name=name, mode='load', **kwargs)

    def clear_session(self):
        """ Clear session.
        """
        clear_session()

    def summary(self):
        """ Show model summary.
        """
        if self.model:
            summary(self.model)

    def visualize_representations(self,
                                  x,
                                  metadata=None,
                                  logs_path=None,
                                  hidden_layer_name=None,
                                  latent_model_output=False,
                                  ):
        """ Visualize representations.

        To visualize the representations on TensorBoard follow the steps:
        1) Use the command line: ' + 'tensorboard --logdir=<logs_path>
        alt-1) I previous step does not work, use the command line:
            python <where TensorBoard package is installed>/main.py --logdir=<logs_path>
        2) Use an internet browser: http://localhost:6006/#projector'

            Parameters:
                x (list, array): Data to be mapped to latent representations.
                metadata (list(array), array): Metadata (a list of arrays or an array).
                logs_path (str): Logs path.
                hidden_layer_name (str): Name of the hidden layer to get insights from.
                latent_model_output (bool): Whether to directly use the output of the latent model.
        """
        if latent_model_output and self.latent_model is None:
            warnings.warn('latent model should be created first')
        method_l_path = self.__manage_path(logs_path, path_type='logs')
        if hidden_layer_name is not None:
            model = get_latent_model(self.model, hidden_layer_name)
        else:
            if latent_model_output:
                model = self.latent_model
            else:
                model = self.model
        representations_name = model.output_names[0]
        save_representations(
            representations=model.predict(x),
            path=method_l_path,
            representations_name=representations_name,
            metadata=metadata
        )

    def __save_load_model(self, name, mode, **kwargs):
        if mode == 'save':
            if self.__force_subclass_weights_saver:
                self.model.save_weights(name)
            else:
                save_model(model=self.model, name=name)
        elif mode == 'load':
            if self.__force_subclass_weights_loader:
                model = self.__model_constructor(**kwargs)
                model.load_weights(name)
                return model
            else:
                return load_model(name, custom_objects=self.__custom_objects)

    def __train(self,
                train_specs,
                model,
                epochs,
                x_train,
                y_train,
                x_val=None,
                y_val=None,
                callbacks=None,
                verbose=None
                ):
        trainer_kwargs = train_specs.copy()
        trainer_kwargs.update({'module': model})
        if callbacks:
            trainer_kwargs.update({'callbacks': callbacks})
        trainer = self.__trainer_class(**trainer_kwargs)
        trainer_fullargspec = list(getfullargspec(trainer.fit))[0]
        train_kwargs = {}
        if not any([val_ is None for val_ in [x_val, y_val]]) and \
                all([val_ in trainer_fullargspec for val_ in ['x_val', 'y_val']]):
            train_kwargs.update({'x_val': x_val, 'y_val': y_val})
        train_kwargs.update({'epochs': epochs})
        for karg, val in zip(['verbose'], [verbose]):
            train_kwargs.update({karg: val})
        trainer.fit(x_train, y_train, **train_kwargs)
        return trainer

    def __get_model_interactor(self, use_trainer):
        if use_trainer:
            if self.__trainer:
                instance = self.__trainer
            else:
                instance = self.__trainer_class(model=self.model)
                if hasattr(instance, 'initialize') and callable(instance.initialize):
                    instance.initialize()
        else:
            instance = self.model
        return instance

    def __create(self, **kwargs):
        self.model = self.__model_constructor(**kwargs)
        if self.__model_constructor_wrapper:
            self.__model_constructor_wrapper(self.model)
        if self.__cuda in kwargs and BACKEND != 'tensorflow':
            self.model.cuda()

    def __manage_path(self, path, path_ext=None, path_type='results'):
        default_path = self.results_path if path_type == 'results' else self.logs_path
        path_ = path if path is not None else default_path
        if path_ext:
            path_ = os.path.join(path_, path_ext)
        path_management(path_)
        return path_
