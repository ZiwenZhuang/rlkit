import argparse
import pickle
from os import path as osp
from rlkit.core import logger
from rlkit.core.repeat_logger import RepeatLogger, RepeatPlotter
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch import pytorch_util as ptu
from rlkit.envs.vae_wrapper import VAEWrappedEnv
import mujoco_py

def simulate_policy(args):
    # start a useless environment incase opengl version error
    mujoco_py.MjViewer(mujoco_py.MjSim(mujoco_py.load_model_from_path(osp.join(osp.dirname(__file__), "Dummy.xml"))))
    logger.log("finish adding dummy context viewer, start loading file")
    with open(args.file, "rb") as f:
        data = pickle.load(open(args.file, "rb"))
        policy = data['evaluation/policy']
        env = data['evaluation/env']
        print("Policy and environment loaded")
    if args.redump:
        # re-dump the data
        with open(args.file, "wb") as f:
            pickle.dump(data, f)
    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.to(ptu.device)
    if isinstance(env, VAEWrappedEnv) and hasattr(env, 'mode'):
        env.mode(args.mode)
    if args.enable_render or hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env.enable_render()
    if args.log_dir != None:
        # time to setup logger to dump recordings to file
        # It should be safer to use absolute directory
        logger.set_snapshot_dir(osp.abspath(args.log_dir))
        logger.add_tabular_output('rollouts.csv', relative_to_snapshot_dir=True)
        success_logger = RepeatLogger(osp.join(osp.abspath(args.log_dir), 'image_success.csv'))
        success_logger = RepeatLogger(osp.join(osp.abspath(args.log_dir), 'vae_dist.csv'))
    paths = []
    for ite in range(64): # incase the testing takes too much physical memory
        paths.append(multitask_rollout(
            env,
            policy,
            max_path_length=args.H,
            render=not args.hide,
            observation_key='observation',
            desired_goal_key='desired_goal',
        ))
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
        if hasattr(env, "get_diagnostics"):
            for k, v in env.get_diagnostics(paths).items():
                logger.record_tabular(k, v)
        if args.log_dir != None:
            # this data has to be chosen by specific path field.
            success_logger.record([paths[-1]['env_infos'][i]['image_success'] for i in range(len(paths[-1]['env_infos']))])
            success_logger.record([paths[-1]['env_infos'][i]['vae_dist'] for i in range(len(paths[-1]['env_infos']))])
        logger.dump_tabular()
        logger.log("Rollout done: # %d" % ite)

    print("Testing learning result done...")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--mode', default='video_env', type=str,
                        help='env mode')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--log_dir', type=str, default= None,
                        help='Specify the log directory, no logging if not')
    parser.add_argument('--redump', action='store_true', default=False,
                        help='restore the data if you need to some modification (not recommended)')
    args = parser.parse_args()

    simulate_policy(args)
