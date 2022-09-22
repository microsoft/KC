# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import subprocess
import argparse
import os

def run_command(bash_command):
    process = subprocess.Popen(bash_command.split())
    output, error = process.communicate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_script',
                        type=str,
                        default='run_config.json',
                        help='This config should mimc the config.py config json with parameters you want to override.'
                             'You can also override the parameters from config_script by passing them in directly after config_script. E.g., --train_config.batch_size 5')
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=["train", "eval", "dump_preds", "dump_embs"])
    parser.add_argument('--base_dir',
                        type=str,
                        default='')
    parser.add_argument('--experiment_name',
                        type=str,
                        default='test')
    parser.add_argument('--tensorboard_dir',
                        type=str,
                        default='')
    args = parser.parse_args()
    print("RUN python setup.py develop")
    run_command("python setup.py develop")
    if not os.path.exists(args.tensorboard_dir):
        os.mkdir(args.tensorboard_dir)
    cmd = f"python bootleg/run.py --config_script {args.config_script} --mode {args.mode} --base_dir {args.base_dir} --experiment_name {args.experiment_name} --tensorboard_dir {args.tensorboard_dir}"
    print("RUN {}".format(cmd))
    run_command(cmd)