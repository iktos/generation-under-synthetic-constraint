from __future__ import print_function

import argparse
import json
import logging
import os

import numpy as np
from guacamol.guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.guacamol.utils.helpers import setup_default_logger

from .ppo_directed_generator import PPODirectedGenerator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--episode_size', type=int, default=8192)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--entropy_weight', type=int, default=1)
    parser.add_argument('--kl_div_weight', type=int, default=10)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--clip_param', type=int, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--suite', default='pi3kmtor')
    parser.add_argument('--database_name')
    parser.add_argument('--pretrained_on',type=str,help="chembl or pi3kmtor (pi3kmtor means chembl+pi3kmtor)")


    args = parser.parse_args()

    np.random.seed(args.seed)

    setup_default_logger()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    if args.pretrained_on == "pi3kmtor":
        model_path = os.path.join(os.getcwd(), "guacamol_baselines","smiles_lstm_hc","pretrained_model_pi3kmtor","model_final_0.126.pt")
    
    elif args.pretrained_on=="chembl":
        model_path = os.path.join(os.getcwd(), "guacamol_baselines","smiles_lstm_hc", 'pretrained_model', 'model_final_0.473.pt')


   ## if args.model_path is None:
     #   dir_path = os.path.dirname(os.path.realpath(__file__))
        #args.model_path = os.path.join(dir_path, 'pretrained_model', 'model_final_0.473.pt')
      #  args.model_path = os.path.join(os.getcwd(), "guacamol_baselines","smiles_lstm_hc", 'pretrained_model', 'model_final_0.473.pt')

            #args.model_path = os.path.join(os.getcwd(), "guacamol_baselines/smiles_lstm_hc/pretrained_model_pi3kmtor/model_final_0.126.pt")

    
    # save command line args
    with open(os.path.join(args.output_dir, 'goal_directed_params.json'), 'w') as jf:
        json.dump(vars(args), jf, sort_keys=True, indent=4)
    optimiser = PPODirectedGenerator(pretrained_model_path=model_path,
                                     num_epochs=args.num_epochs,
                                     episode_size=args.episode_size,
                                     batch_size=args.batch_size,
                                     entropy_weight=args.entropy_weight,
                                     kl_div_weight=args.kl_div_weight,
                                     clip_param=args.clip_param,
                                     database_name=args.database_name
                                     )

    json_file_path = os.path.join(args.output_dir, 'PPO_results.json')
    assess_goal_directed_generation(optimiser, json_output_file=json_file_path,
                                    benchmark_version=args.suite)


if __name__ == "__main__":
    main()
 # python -m guacamol_baselines.smiles_lstm_ppo.goal_directed_generation --suite pi3kmtor --pretrained_on pi3kmtor --database_name myPPOgenerationpi3kmtor 
