import os
import subprocess
import time
from itertools import product
import sys


def test_application(repos_path, application_name, execution_mode):

    # Install backend

    # Manage paths
    scripts_path = os.path.join(repos_path, application_name)
    logs_path = os.path.join(repos_path, 'test_logs', application_name).replace('.py', '')
    os.makedirs(logs_path, exist_ok=True)

    # Test scripts
    if os.path.isdir(scripts_path):
        script_names = [os.path.join(scripts_path, script_name) for script_name in os.listdir(scripts_path)
                        if '.py' in script_name]
    else:
        script_names = [scripts_path]
    test_scripts(
        script_names=script_names,
        logs_path=logs_path,
        execution_mode=execution_mode
    )


def test_scripts(script_names, logs_path, execution_mode):
    
    # Test scripts
    for script_name in script_names:

        start_time = time.time()
        arguments_list = get_script_arguments(script_name)
        arguments_list = arguments_list if len(arguments_list) > 0 else [None]
        while len(arguments_list) > 0:
            arguments = arguments_list[0]
            arguments_ = arguments.split() if arguments is not None else []
            command_list = [sys.executable, '-u', script_name] + arguments_
            print('testing: ' + ' '.join(command_list))
            env = os.environ.copy()
            env["EXECUTION_MODE"] = execution_mode
            proc = subprocess.Popen(
                command_list,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            log_file_name = os.path.join(logs_path, script_name.split(os.sep)[-1]).replace('.py', '') + "_"
            log_file_name += '_'.join([execution_mode + "_test"] + arguments_) + '.txt'
            if os.path.isfile(log_file_name):
                os.remove(log_file_name)
            for line in proc.stdout:
                with open(log_file_name, 'a') as logfile:
                    logfile.write(line.decode('utf-8'))
            while proc.poll() is None:
                time.sleep(10)
            status = [script_name, 'PASSED' if proc.poll() == 0 else 'FAILED']
            print('{} ({} seconds): {}'.format(log_file_name, round(time.time() - start_time), status[1]))
            arguments_list.remove(arguments)


def get_script_arguments(script_name):
    project = ['--project mnist', '--project fashion_mnist', '--project wallmart']
    precision = ['--precision float32', '--precision float16']
    new_design = ['--new_design True', '']
    max_evals = ['--max_evals 2', '--max_evals 1']
    epochs = ['--epochs 1']
    arguments = []
    arguments_options = None
    if os.path.join('benchmarks', 'runner') in script_name:
        arguments_options = [
            project,
            precision
        ]
    elif os.path.join('aironsuit', 'examples', 'standard_classification_pipeline') in script_name or\
            os.path.join('data_cleaner', 'main') in script_name or\
            os.path.join('benchmarks', 'runner') in script_name:
        arguments_options = [
            precision,
            new_design,
            max_evals,
            epochs
        ]
    if arguments_options is not None:
        arguments += [' '.join(list(args)) for args in list(product(*arguments_options))]
    return arguments
