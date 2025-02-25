import argparse
import pickle
from os import path as osp
from rlkit.core import logger
from rlkit.core.repeat_logger import RepeatLogger, RepeatPlotter
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch import pytorch_util as ptu
from rlkit.envs.vae_wrapper import VAEWrappedEnv
import numpy as np
import mujoco_py
import pickle as pkl

def simulate_policy(args):
    # # start a useless environment incase opengl version error
    # mujoco_py.MjViewer(mujoco_py.MjSim(mujoco_py.load_model_from_path(osp.join(osp.dirname(__file__), "Dummy.xml"))))

    logger.log("finish adding dummy context viewer, start loading file")
    with open(args.file, "rb") as f:
        data = pickle.load(open(args.file, "rb"))
        policy = data['evaluation/policy']
        env = data['evaluation/env']
        logger.log("Policy and environment loaded")
    if args.redump:
        # re-dump the data
        with open(args.file, "wb") as f:
            pickle.dump(data, f)
        logger.log("Finish redump")
    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.to(ptu.device)
    if isinstance(env, VAEWrappedEnv) and hasattr(env, 'mode'):
        env.mode(args.mode)
    if (args.enable_render or hasattr(env, 'enable_render')) and not args.hide:
        # some environments need to be reconfigured for visualization
        logger.log("Enable Rendering")
        env.enable_render()
    if args.log_dir != None:
        # time to setup logger to dump recordings to file
        # It should be safer to use absolute directory
        logger.set_snapshot_dir(osp.abspath(args.log_dir))
        logger.add_tabular_output('rollouts.csv', relative_to_snapshot_dir=True)
        success_logger = RepeatLogger(osp.join(osp.abspath(args.log_dir), 'image_success.csv'))
        vae_logger = RepeatLogger(osp.join(osp.abspath(args.log_dir), 'vae_dist.csv'))
        ag_logger = RepeatLogger(osp.join(osp.abspath(args.log_dir), 'effector2goal_distance.csv'))
        logger.log("Setup loggers")
    if hasattr(env, '_goal_sampling_mode') and env._goal_sampling_mode == 'custom_goal_sampler' and env.custom_goal_sampler == None:
        # This a deep hack, to make the sample directly from env wrapped by image_env
        # ---------------- change to use presampled goal for RIG_door algorithm -------------
        # env.custom_goal_sampler = env._customed_goal_sampling_func
        # logger.log("Change env.custom_goal_sampler to its _customed_goal_sampling_func")
        env._goal_sampling_mode = "presampled"
    paths = []
    logger.log("Start Rollout")
    for ite in range(64): # incase the testing takes too much physical memory
        paths.append(multitask_rollout(
            env,
            policy,
            max_path_length=args.H,
            render=not args.hide,
            observation_key=data['evaluation/observation_key'],
            desired_goal_key=data['evaluation/desired_goal_key'],
        ))
        logger.log("iter %d: Finish rollout" % ite)
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
            logger.log("iter %d: Log diagnostics" % ite)
        if hasattr(env, "get_diagnostics"):
            for k, v in env.get_diagnostics(paths).items():
                logger.record_tabular(k, v)
            logger.log("iter %d: Get diagnostics" % ite)
        if args.log_dir != None:
            # this data has to be chosen by specific path field.
            success_logger.record([paths[-1]['env_infos'][i]['image_success'] for i in range(len(paths[-1]['env_infos']))])
            vae_logger.record([paths[-1]['env_infos'][i]['vae_dist'] for i in range(len(paths[-1]['env_infos']))])
            if "effector2goal_distance" in paths[-1]['env_infos'][0].keys():
                goal_dist = [np.linalg.norm(paths[-1]['env_infos'][i]['effector2goal_distance']) for i in range(len(paths[-1]['env_infos']))]
                ag_logger.record(goal_dist)
                with open(args.log_dir + "2goal_dist.pkl", "wb+") as f:
                    pkl.dump(goal_dist, f)
            logger.log("iter %d: Log into files" % ite)

        logger.dump_tabular()
        logger.log("Rollout done: # %d" % ite)

    logger.log("Testing learning result done...")

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
