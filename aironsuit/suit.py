import math
import os
import pickle
import random
import tempfile
import warnings

import hyperopt
import numpy as np
import pandas as pd
import tensorflow as tf
from airontools.constructors.models.model import Model
from airontools.constructors.utils import get_latent_model
from airontools.interactors import clear_session, load_model, save_model, summary
from airontools.tensorboard_utils import save_representations
from airontools.tools import path_management
from hyperopt import STATUS_FAIL, STATUS_OK, Trials

from aironsuit._utils import to_sum
from aironsuit.callbacks import get_basic_callbacks, init_callbacks
from aironsuit.design.utils import setup_design_logs, update_design_logs


class AIronSuit(object):
    """AIronSuit is a model wrapper that takes care of the hyper-parameter optimization problem, training and inference
    among other functionalities.

        Attributes:
            model (Model): NN model.
            latent_model (Model): Latent NN model.
            results_path (str): Results path.
            logs_path (int): Logs path.
            __model_constructor (): NN model constructor.
            __devices (list): Devices where to make the computations.
            __total_n_models (int): Total number of models in parallel.
    """

    def __init__(
        self,
        model_constructor=None,
        model=None,
        results_path=os.path.join(tempfile.gettempdir(), "airon") + os.sep,
        logs_path=None,
        custom_objects=None,
        name="NN",
    ):
        """Parameters:
        model_constructor (): Function that returns a model.
        model (Model): User customized model.
        results_path (str): Results path.
        logs_path (str): Logs path.
        custom_objects (dict): Custom objects when loading Keras models.
        name (str): Name of the model.
        """

        self.model = model
        self.latent_model = None
        self.results_path = results_path
        self.logs_path = (
            os.path.join(logs_path, "log_dir")
            if logs_path is not None
            else os.path.join(results_path, "log_dir")
        )
        self.__model_constructor = model_constructor
        self.__custom_objects = custom_objects
        self.__devices = None
        self.__total_n_models = None
        self.name = name
        for path_ in [self.results_path, self.logs_path]:
            path_management(path_)

    def design(
        self,
        x_train,
        x_val,
        hyper_space,
        max_evals,
        epochs,
        batch_size=32,
        y_train=None,
        y_val=None,
        sample_weight=None,
        sample_weight_val=None,
        model_specs=None,
        metric=None,
        trials=None,
        verbose=0,
        seed=None,
        raw_callbacks=None,
        use_basic_callbacks=True,
        patience=3,
        save_val_inference=False,
        optimise_hypers_on_the_fly=False,
        additional_train_kwargs=None,
        additional_evaluation_kwargs=None,
    ):
        """Automatic model design.

        Parameters:
            x_train (list, np.array): Input data for training.
            x_val (list, np.array): Input data for validation.
            hyper_space (dict): Hyper parameter space for model design.
            max_evals (integer): Maximum number of evaluations.
            epochs (int): Number of epochs for model training.
            batch_size (int): Number of samples per batch.
            y_train (list, np.array): Output data for training.
            y_val (list, np.array): Output data for validation.
            sample_weight (np.array): Weight per sample to be computed with the train metric and losses.
            sample_weight_val (np.array): Weight per sample to be computed with the validation metric and losses.
            model_specs (dict): Model specifications.
            metric (str, int, list, function): Metric to be used for model design. If None validation loss is used.
            trials (Trials): Object with design information.
            verbose (int): Verbosity.
            seed (int): Seed for reproducible results.
            raw_callbacks (list): Dictionary of raw callbacks.
            use_basic_callbacks (bool): Whether to use basic callbacks or not. Callbacks argument has preference.
            patience (int): Patience in epochs for validation los improvement, only active when use_basic_callbacks.
            save_val_inference (bool): Whether or not to save validation inference when the best model is found.
            optimise_hypers_on_the_fly (bool): Whether to perform optimisation of hypers on the fly.
            additional_train_kwargs (dict): Additional key arguments for training.
            additional_evaluation_kwargs (dict): Additional key arguments for evaluation.
        """

        additional_train_kwargs = (
            additional_train_kwargs if additional_train_kwargs is not None else {}
        )
        additional_evaluation_kwargs = (
            additional_evaluation_kwargs
            if additional_evaluation_kwargs is not None
            else {}
        )

        setup_design_logs(
            path=self.logs_path,
            hyper_space=hyper_space,
            metric=metric if isinstance(metric, str) else "val_loss",
        )

        if trials is None:
            trials = Trials()
        raw_callbacks = (
            raw_callbacks
            if raw_callbacks
            else get_basic_callbacks(
                path=self.logs_path,
                patience=patience,
                name=self.name,
                verbose=verbose,
                epochs=epochs,
            )
            if use_basic_callbacks
            else None
        )

        def design_trial(hyper_candidates):
            # Save trials
            with open(os.path.join(self.results_path, "trials.hyperopt"), "wb") as f:
                pickle.dump(trials, f)

            # Create model
            specs = hyper_candidates.copy()
            if model_specs:
                specs.update(model_specs)
            self.model = self.__model_constructor(**specs)

            # Print some information
            iteration = len(trials.losses())
            print("\n")
            print("iteration : {}".format(0 if trials.losses() is None else iteration))
            [print("{}: {}".format(key, value)) for key, value in specs.items()]
            if verbose > 0:
                print(self.model.summary())

            # Train model
            self.__train(
                epochs=epochs,
                batch_size=batch_size,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                sample_weight=sample_weight,
                sample_weight_val=sample_weight_val,
                raw_callbacks=raw_callbacks,
                verbose=verbose,
                optimise_hypers_on_the_fly=optimise_hypers_on_the_fly,
                additional_evaluation_kwargs=additional_evaluation_kwargs,
                **additional_train_kwargs
            )

            # Model evaluation
            evaluation = self.__evaluate(
                x=x_val,
                y=y_val,
                batch_size=batch_size,
                sample_weight=sample_weight_val,
                metric=metric,
                verbose=verbose,
                **additional_evaluation_kwargs
            )
            if verbose > 0:
                print("\n")
                print("Model Evaluation: ", evaluation)

            # Define status
            status = STATUS_OK if not math.isnan(evaluation) else STATUS_FAIL
            print("status: ", status)

            # Save model if it is the best so far
            evaluation_file_name = os.path.join(
                self.results_path, "_".join([self.name, "loss"])
            )
            trials_losses = [loss_ for loss_ in trials.losses() if loss_ is not None]
            best_evaluation = min(trials_losses) if len(trials_losses) > 0 else None
            print("best evaluation so far: " + str(best_evaluation))
            print("current evaluation: " + str(evaluation))
            evaluation_cond = best_evaluation is None or evaluation < best_evaluation
            save_cond = status == STATUS_OK and evaluation_cond
            print("save: " + str(save_cond))
            if save_cond:
                df = pd.DataFrame(data=[evaluation], columns=["evaluation"])
                df.to_pickle(evaluation_file_name)
                self.model.save_weights(os.path.join(self.results_path, self.name))
                with open(
                    os.path.join(
                        self.results_path,
                        "_".join([self.name, "hyper_candidates"]),
                    ),
                    "wb",
                ) as f:
                    pickle.dump(hyper_candidates, f, protocol=pickle.HIGHEST_PROTOCOL)
                if save_val_inference and y_val is not None:
                    y_inf = self.model.predict(x_val)
                    y_inf = (
                        np.concatenate(y_inf, axis=1)
                        if isinstance(y_inf, list)
                        else y_inf
                    )
                    np.savetxt(
                        os.path.join("inference", "val_target_inference.csv"),
                        y_inf,
                        delimiter=",",
                    )
                # Update logs
                update_design_logs(
                    path=os.path.join(self.logs_path, str(len(trials.losses()))),
                    hparams={
                        value["logs"]: specs[key] for key, value in hyper_space.items()
                    },
                    value=evaluation,
                    step=len(trials.losses()),
                    metric=metric if isinstance(metric, str) else "val_loss",
                )

            clear_session()
            del self.model

            return {"loss": evaluation, "status": status}

        def design():
            if len(trials.trials) < max_evals:
                self.fmin = hyperopt.fmin(
                    design_trial,
                    rstate=None if seed is None else np.random.default_rng(seed),
                    space={key: value["options"] for key, value in hyper_space.items()},
                    algo=hyperopt.tpe.suggest,
                    max_evals=max_evals,
                    trials=trials,
                    verbose=True,
                    return_argmin=False,
                )
                # Save trials
                with open(
                    os.path.join(self.results_path, "trials.hyperopt"), "wb"
                ) as f:
                    pickle.dump(trials, f)
            hyper_candidates = self.load_hyper_candidates()

            # Best model
            specs = {}
            if model_specs:
                specs.update(model_specs.copy())
            specs.update(hyper_candidates)
            self.model = self.__model_constructor(**specs)
            self.model.load_weights(os.path.join(self.results_path, self.name))
            print("hyper-parameters: " + str(hyper_candidates))

        design()

    def train(
        self,
        epochs,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        batch_size=32,
        callbacks=None,
        verbose=0,
        use_basic_callbacks=True,
        patience=3,
        optimise_hypers_on_the_fly=False,
    ):
        """Weight optimization.

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
            patience (int): Patience in epochs for validation los improvement, only active when use_basic_callbacks.
            optimise_hypers_on_the_fly (bool): Whether to perform optimisation of hypers on the fly.
        """
        raw_callbacks = (
            callbacks
            if callbacks
            else get_basic_callbacks(
                path=self.logs_path,
                patience=patience,
                name=self.name,
                verbose=verbose,
                epochs=epochs,
            )
            if use_basic_callbacks
            else None
        )
        self.__train(
            epochs=epochs,
            batch_size=batch_size,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            raw_callbacks=raw_callbacks,
            verbose=verbose,
            optimise_hypers_on_the_fly=optimise_hypers_on_the_fly,
        )

    def inference(self, x):
        """Inference.

        Parameters:
            x (list, np.array): Input data for training.
        """
        return self.model.predict(x)

    def latent_inference(self, x, layer_names=None):
        """Latent inference.

        Parameters:
            x (list, np.array): Input data for training.
            layer_names (str): Layer names.
        """
        assert all([var is not None for var in [layer_names, self.latent_model]])
        if layer_names:
            self.latent_model = get_latent_model(self.model, layer_names)
        return self.latent_model.predict(x)

    def create_latent_model(self, hidden_layer_names):
        """Create latent model given a model and hidden layer names.

        Parameters:
            hidden_layer_names (str): Layer names.
        """
        assert self.model is not None
        self.latent_model = get_latent_model(self.model, hidden_layer_names)

    def evaluate(
        self,
        x,
        y=None,
        batch_size=32,
        sample_weight=None,
        metric=None,
        verbose=0,
        return_sum=False,
        **kwargs
    ):
        """Evaluate.

        Parameters:
            x (list, np.array): Input data for evaluation.
            y (list, np.array): Output data for evaluation.
            batch_size (int): Number of samples per batch.
            sample_weight (np.array): Weight per sample to be computed for the evaluation.
            metric (str, int, list, function): Metric to be used for model design. If None validation loss is used.
            verbose (int): Verbosity.
            return_sum (bool): Whether to return just the sum of the metrics.
        """
        return self.__evaluate(
            x,
            y,
            batch_size=batch_size,
            sample_weight=sample_weight,
            metric=metric,
            verbose=verbose,
            return_number=return_sum,
            **kwargs
        )

    def save_model(self, name):
        """Save the model.
        Parameters:
            name (str): Model name.
        """
        save_model(model=self.model, name=name)

    def load_model(self, name, **kwargs):
        """Load the model.
        Parameters:
            name (str): Model name.
            kwargs (dict): Custom or other arguments.
        """
        self.model = load_model(name, custom_objects=self.__custom_objects)

    def clear_session(self):
        """Clear session."""
        clear_session()

    def summary(self):
        """Show model summary."""
        if self.model:
            summary(self.model)

    def visualize_representations(
        self,
        x,
        metadata=None,
        logs_path=None,
        hidden_layer_name=None,
        latent_model_output=False,
    ):
        """Visualize representations.

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
            warnings.warn("latent model should be created first")
        if hidden_layer_name is not None:
            model = get_latent_model(self.model, hidden_layer_name)
        else:
            if latent_model_output:
                model = self.latent_model
            else:
                model = self.model
        save_representations(
            representations=model.predict(x),
            path=self.logs_path,
            representations_name=model.output_names[0],
            metadata=metadata,
        )

    def load_hyper_candidates(self):
        """Load hyper candidates."""
        with open(
            os.path.join(self.results_path, "_".join([self.name, "hyper_candidates"])),
            "rb",
        ) as f:
            hyper_candidates = pickle.load(f)
        return hyper_candidates

    def __train(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        batch_size=32,
        sample_weight=None,
        sample_weight_val=None,
        raw_callbacks=None,
        patience=10,
        optimise_hypers_on_the_fly=False,
        verbose=0,
        additional_evaluation_kwargs=None,
        metric=None,
        **kwargs
    ):
        additional_evaluation_kwargs = (
            additional_evaluation_kwargs if additional_evaluation_kwargs is None else {}
        )
        train_kwargs = kwargs.copy()
        if isinstance(self.model, Model):
            train_kwargs["verbose"] = verbose
        if raw_callbacks is not None:
            if all([isinstance(callback, dict) for callback in raw_callbacks]):
                callbacks = init_callbacks(raw_callbacks)
            else:
                callbacks = raw_callbacks
            train_kwargs.update({"callbacks": callbacks})
        fit_args = [x_train]
        if y_train is not None:
            fit_args += [y_train]
        val_data = []
        for val_data_ in [x_val, y_val, sample_weight_val]:
            if val_data_ is not None:
                val_data += [val_data_]
        if len(val_data) != 0:
            train_kwargs.update({"validation_data": tuple(val_data)})
        if all([isinstance(data, tf.data.Dataset) for data in fit_args]):
            train_kwargs["validation_data"] = tf.data.Dataset.zip(
                train_kwargs["validation_data"]
            ).batch(batch_size)
            if sample_weight is not None:
                warnings.warn(
                    "sample weight for training combined with tf datasets is not supported at the moment"
                )
            fit_args = [tf.data.Dataset.zip(tuple(fit_args)).batch(batch_size)]
        else:
            if sample_weight is not None:
                train_kwargs["sample_weight"] = sample_weight
            train_kwargs["batch_size"] = batch_size
        self.model.fit(*fit_args, **train_kwargs)
        if optimise_hypers_on_the_fly:
            if "epochs" in train_kwargs.keys():
                train_kwargs["epochs"] = 1
            hyper_designs = {
                method: getattr(self.model, method).actions_space
                for method in dir(self.model)
                if "hyper_design" in method
            }
            if len(hyper_designs) == 0:
                warnings.warn("could not find hyper designs to perform on the fly")
            else:
                print("Starting optimisation of hypers on the fly...")
                prev_evaluation = self.__evaluate(
                    x=x_val,
                    y=y_val,
                    batch_size=batch_size,
                    sample_weight=sample_weight_val,
                    metric=metric,
                    verbose=verbose,
                    **additional_evaluation_kwargs
                )
                improvement = False
                for i in range(patience):
                    if not improvement:
                        for hyper_design_name, action_space in hyper_designs.items():
                            getattr(self.model, hyper_design_name).set_action(
                                random.choice(list(action_space.keys()))
                            )
                    self.model.fit(*fit_args, **train_kwargs)
                    evaluation = self.__evaluate(
                        x=x_val,
                        y=y_val,
                        batch_size=batch_size,
                        sample_weight=sample_weight_val,
                        metric=metric,
                        verbose=verbose,
                        **additional_evaluation_kwargs
                    )
                    improvement = evaluation < prev_evaluation
                    prev_evaluation = evaluation

    def __evaluate(
        self,
        x,
        y,
        batch_size=32,
        sample_weight=None,
        metric=None,
        verbose=0,
        return_number=True,
        **kwargs
    ):
        evaluate_args = [x]
        if y is not None:
            evaluate_args += [y]
        evaluate_kwargs = kwargs.copy()
        if isinstance(self.model, Model):
            evaluate_kwargs["verbose"] = verbose
        if sample_weight is not None:
            evaluate_kwargs["sample_weight"] = sample_weight
        keras_model = isinstance(self.model, Model)
        data_as_tfrecords = all(
            [isinstance(data, tf.data.Dataset) for data in evaluate_args]
        )
        if (
            any([isinstance(metric, var_type) for var_type in [int, str, list]])
            or metric is None
        ):
            if data_as_tfrecords and keras_model:
                if sample_weight is not None:
                    evaluate_args += [evaluate_kwargs["sample_weight"]]
                    del evaluate_kwargs["sample_weight"]
                    evaluate_args = tf.data.Dataset.from_tensor_slices(
                        tuple(
                            [
                                list(eval_data_.as_numpy_iterator())
                                for eval_data_ in evaluate_args
                            ]
                        )
                    )
                else:
                    evaluate_args = tf.data.Dataset.zip(tuple(evaluate_args))
                evaluate_args = evaluate_args.batch(batch_size)
            if isinstance(metric, str):
                evaluate_kwargs.update({"return_dict": True})
            if data_as_tfrecords:
                evaluation = self.model.evaluate(evaluate_args, **evaluate_kwargs)
            else:
                evaluation = self.model.evaluate(*evaluate_args, **evaluate_kwargs)
            if isinstance(metric, list):
                evaluation = [evaluation[key] for key in metric]
            elif metric is not None:
                evaluation = evaluation[metric]
        else:
            evaluate_kwargs["model"] = self.model
            evaluation = metric(
                evaluate_args,
                **{
                    key: value
                    for key, value in evaluate_kwargs.items()
                    if key in metric.__annotations__.keys()
                }
            )
        if return_number:
            evaluation = to_sum(evaluation)
        return evaluation
