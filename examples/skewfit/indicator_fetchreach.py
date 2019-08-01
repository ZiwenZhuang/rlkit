import rlkit.util.hyperparameter as hyp
# following import is no use in this starting script
# from rlkit.envs.goal_generation.pickup_goal_dataset import (
#     generate_vae_dataset,
#     get_image_presampled_goals_from_vae_env,
# )
from rlkit.envs.goal_generation.fetch_goal_dataset import \
    get_image_presampled_goals_from_vae_env

import rlkit.torch.vae.vae_schedules as vae_schedules
from multiworld.envs.mujoco.cameras import (
    sawyer_pick_and_place_camera,
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.skewfitIndicator_experiments import skewfit_full_experiment
from rlkit.torch.vae.conv_vae import imsize48_default_architecture
from copy import deepcopy

def main(args):
    num_images = 1
    image_env_kwargs = dict(
        depth= True,
        reward_type= 'image_indicator',
    )
    vae_wrapped_env_kwargs = dict(
            sample_from_true_prior=False,
            disable_vae= True,
            goal_sampling_mode= "env",
            reward_params=dict(
                # type='latent_sparse',
                type='wrapped_env',
            ),
        )
    imsize48_architecture = deepcopy(imsize48_default_architecture)
    imsize48_architecture['deconv_args']['deconv_output_channels'] = 4
    variant = dict(
        algorithm='Skew-Fit',
        imsize=48,
        double_algo=False,
        env_id="FetchReach-v0",
        skewfit_variant=dict(
            sample_goals_from_buffer=True,
            save_video=True,
            save_video_period=50,
            presample_goals=True,
            custom_goal_sampler='replay_buffer',
            online_vae_trainer_kwargs=dict(
                beta=30,
                lr=1e-3,
            ),
            generate_goal_dataset_fctn=get_image_presampled_goals_from_vae_env, #TODO: choose a proper function, refering to 'get_image_presampled_goals_from_vae_env'
            goal_generation_kwargs=dict(
                num_presampled_goals=500,
            ),
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            vf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            max_path_length=50,
            algo_kwargs=dict(
                batch_size=1024,
                num_epochs=200,
                num_eval_steps_per_epoch=500,
                num_expl_steps_per_train_loop=500,
                num_trains_per_train_loop=5000,
                min_num_steps_before_training=5000,
                vae_training_schedule=vae_schedules.never_train,
                oracle_data=False,
                vae_save_period=50,
                parallel_vae_train=False,
            ),
            twin_sac_trainer_kwargs=dict(
                reward_scale=1,
                discount=0.99,
                soft_target_tau=1e-3,
                target_update_period=1,
                use_automatic_entropy_tuning=True,
                use_reward_indicator= True,
                use_reward_filter= False,
                filtering_probs= (0.9, 0.8),
            ),
            replay_buffer_kwargs=dict(
                start_skew_epoch=10,
                max_size=int(5000),
                fraction_goals_rollout_goals=0.2,
                fraction_goals_env_goals=0.5,
                exploration_rewards_type='None',
                vae_priority_type='vae_prob',
                priority_function_kwargs=dict(
                    sampling_method='importance_sampling',
                    decoder_distribution='gaussian_identity_variance',
                    num_latents_to_sample=10,
                ),
                power=-1,
                relabeling_goal_sampling_mode='env',
                disable_vae= vae_wrapped_env_kwargs['disable_vae'],
            ),
            exploration_goal_sampling_mode='env',
            evaluation_goal_sampling_mode='env',
            normalize=False,
            render=False,
            exploration_noise=0.0,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type=vae_wrapped_env_kwargs['reward_params']['type'],
            ),
            observation_key='observation',
            desired_goal_key='desired_goal',
            vae_wrapped_env_kwargs= vae_wrapped_env_kwargs,
            image_env_kwargs= image_env_kwargs,
        ),
        train_vae_variant=dict(
            representation_size=16,
            beta=5,
            num_epochs=0,
            dump_skew_debug_plots=True,
            decoder_activation='gaussian',
            vae_kwargs=dict(
                decoder_distribution='gaussian_identity_variance',
                input_channels=4,
                architecture=imsize48_architecture,
            ),
            # generate_vae_data_fctn=None, # TODO: choose aproper funciton, refering to 'generate_vae_dataset'
            generate_vae_dataset_kwargs=dict(
                N=10,
                oracle_dataset=True,
                use_cached=False,
                num_channels=4*num_images,
                image_env_kwargs= image_env_kwargs,
            ),


            algo_kwargs=dict(
                start_skew_epoch=12000,
                is_auto_encoder=False,
                batch_size=64,
                lr=1e-3,
                skew_config=dict(
                    method='vae_prob',
                    power=0,
                ),
                skew_dataset=True,
                priority_function_kwargs=dict(
                    decoder_distribution='gaussian_identity_variance',
                    sampling_method='true_prior_sampling',
                    num_latents_to_sample=10,
                ),
                use_parallel_dataloading=False,
            ),
            save_period=10,
        ),
        init_camera=sawyer_pick_and_place_camera,
    )

    search_space = {}
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    # n_seeds = 3
    # mode = 'gcp'
    # exp_prefix = 'skew-fit-pickup-reference-post-refactor'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                skewfit_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                snapshot_gap=200,
                snapshot_mode='gap_and_last',
                num_exps_per_instance=3,
                base_log_dir=args.base_log_dir,
                gcp_kwargs=dict(
                    zone='us-west1-b',
                ),
            )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--log_dir', help= 'the root directory to store experiment data',
        dest= 'base_log_dir', type= str, default= None)

    main(parser.parse_args())
