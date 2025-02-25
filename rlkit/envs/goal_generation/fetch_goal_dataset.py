'''
in order to meet the gap between the rlkit
goal generation and original FetchEnv environment.
'''
import numpy as np
import cv2
import os.path as osp
import random

from rlkit.util.io import local_path_from_s3_or_local_path

def get_image_presampled_goals_from_vae_env(env, env_id= None, num_presampled_goals= 1000):
    ''' Based on the understaning of the function, it takes in following parameters:
            env: ususally the VAE Wrapper environment
            env_id: the name (I don't think there are much of use)
            num_presampled_goals: ...
            ... (other arguments it need which can be specified on the starting script)
        outputs: dictionary with following items
            'desired_goal': an array of flatterned image
            'image_desired_goal': an array of flatterned image
            'proprio_desired_goal': an array of gripper position goal ('g_state' in FetchEnv)
            'state_desired_goal': an array of gripper
    '''
    image_env = env.wrapped_env
    return get_image_presampled_goals(image_env, num_presampled_goals)

def get_image_presampled_goals(image_env, num_presampled_goals):
    ''' To meet the interface, which should be written in image_env instead of sawyer... environment.
        A complementary function is implemented here.
    '''
    image_env.reset()
    fetch_env = image_env.wrapped_env

    # start to collect goal samples one by one
    desired_goal = []
    image_desired_goal = []
    proprio_desired_goal = []
    state_desired_goal = []
    for _ in range(num_presampled_goals):\
        # reset the environment, which should generate new goal
        obs = fetch_env.reset()
        # from the FetchEnv, the observation need to be flattened
        obs = {
            k: v.flatten() for k, v in obs.items()
        }
        # normalize image incase it is raw image
        for k, v in obs.items():
            if v.dtype == np.uint8:
                obs[k] = v / 255.0
        info = fetch_env.get_current_info()
        
        desired_goal.append(obs['desired_goal'])
        image_desired_goal.append(obs['desired_goal'])

        proprio_desired_goal.append(info['ag_state'])
        state_desired_goal.append(info['g_state'])

    return {
        'desired_goal': np.array(desired_goal),
        'image_desired_goal': np.array(image_desired_goal),
        'proprio_desired_goal': np.array(proprio_desired_goal),
        'state_desired_goal': np.array(state_desired_goal),
    }

def setup_pickup_image_env(image_env, num_presampled_goals):
    """
    Image env and pickup env will have presampled goals. VAE wrapper should
    encode whatever presampled goal is sampled.
    """
    presampled_goals = get_image_presampled_goals(image_env,
                                                  num_presampled_goals)
    image_env._presampled_goals = presampled_goals
    image_env.num_goals_presampled = \
    presampled_goals[random.choice(list(presampled_goals))].shape[0]    


# It seems like the following function is not being used.
def generate_vae_dataset_from_params(
        env_class=None,
        env_kwargs=None,
        env_id=None,
        N=10000,
        test_p=0.9,
        use_cached=True,
        imsize=84,
        num_channels=1,
        show=False,
        init_camera=None,
        dataset_path=None,
        oracle_dataset=False,
        n_random_steps=100,
        vae_dataset_specific_env_kwargs=None,
        save_file_prefix=None,
):
    from multiworld.core.image_env import ImageEnv, unormalize_image
    import time

    assert oracle_dataset == True

    if env_kwargs is None:
        env_kwargs = {}
    if save_file_prefix is None:
        save_file_prefix = env_id
    if save_file_prefix is None:
        save_file_prefix = env_class.__name__
    filename = "/tmp/{}_N{}_{}_imsize{}_oracle{}.npy".format(
        save_file_prefix,
        str(N),
        init_camera.__name__ if init_camera else '',
        imsize,
        oracle_dataset,
    )
    info = {}
    if dataset_path is not None:
        filename = local_path_from_s3_or_local_path(dataset_path)
        dataset = np.load(filename)
        np.random.shuffle(dataset)
        N = dataset.shape[0]
    elif use_cached and osp.isfile(filename):
        dataset = np.load(filename)
        np.random.shuffle(dataset)
        print("loaded data from saved file", filename)
    else:
        now = time.time()

        if env_id is not None:
            import gym
            import multiworld
            multiworld.register_all_envs()
            env = gym.make(env_id)
        else:
            if vae_dataset_specific_env_kwargs is None:
                vae_dataset_specific_env_kwargs = {}
            for key, val in env_kwargs.items():
                if key not in vae_dataset_specific_env_kwargs:
                    vae_dataset_specific_env_kwargs[key] = val
            env = env_class(**vae_dataset_specific_env_kwargs)
        if not isinstance(env, ImageEnv):
            env = ImageEnv(
                env,
                imsize,
                init_camera=init_camera,
                transpose=True,
                normalize=True,
            )
        setup_pickup_image_env(env, num_presampled_goals=N)
        env.reset()
        info['env'] = env

        dataset = np.zeros((N, imsize * imsize * num_channels), dtype=np.uint8)
        for i in range(N):
            img = env._presampled_goals['image_desired_goal'][i]
            dataset[i, :] = unormalize_image(img)
            if show:
                img = img.reshape(3, imsize, imsize).transpose()
                img = img[::-1, :, ::-1]
                cv2.imshow('img', img)
                cv2.waitKey(1)
                time.sleep(.2)
                # radius = input('waiting...')
        print("done making training data", filename, time.time() - now)
        np.random.shuffle(dataset)
        np.save(filename, dataset)

    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset, info



