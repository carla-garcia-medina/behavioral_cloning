import torch

'''
Create Dataset Class for Generating + Storing Data
'''
class BCDataset(torch.utils.data.Dataset):
  def __init__(self, env, n_episodes, expert_policy):
    self.env = env
    self.n_episodes = n_episodes
    self.expert_policy = expert_policy
    self._generate()
    self._compute_stats()
  
  def _save(self, path):
    '''
    Save dataset to path as an hd5f file
    '''
    assert path[-4:] == 'hdf5'
    with h5py.File(path, 'w') as f:
      f['expert_obs'] = self.expert_obs.numpy()[:]
      f['expert_actions'] = self.expert_actions.numpy()[:]
  
  def _load(self, path):
    '''
    Load dataset from path
    '''
    assert os.path.exists(path)
    assert path[-4:] == 'hdf5'
    with h5py.File(path, 'r') as f:
      assert 'expert_obs' in f
      assert 'expert_actions' in f
      self.expert_obs = f['expert_obs'][:]
      self.expert_actions = f['expert_actions'][:]
    self._compute_stats()
  
  def _generate(self):
    '''
    Generate dataset by rolling out the expert policy in the input env for
    'n_episodes' total episodes
    '''
    self.expert_obs = []
    self.expert_actions = []
    for _ in range(self.n_episodes):
      obs = self.env.reset()
      while True:
        action, _ = self.expert_policy.predict(torch.as_tensor(obs))
        self.expert_obs += [obs[:]]
        self.expert_actions += [[action]]
        obs, rewards, done, info = env.step(action)
        if done: 
            break;
    self.expert_obs = torch.as_tensor(np.stack(self.expert_obs)).float()
    self.expert_actions = torch.as_tensor(np.stack(self.expert_actions)).float()
  
  def _compute_stats(self):
    '''
    Compute some basic stats that can use later on to normalize the training
    data / criterion function
    '''
    self.mean_expert_obs = torch.mean(self.expert_obs, axis=0)
    self.std_expert_obs = torch.std(self.expert_obs, axis=0)
    self.prop_pos_exmpls = torch.sum(self.expert_actions)/len(self)

  def __len__(self):
    '''
    Get total number of (obs, action) samples
    '''
    return self.expert_obs.shape[0]
  
  def __getitem__(self, i):
    '''
    Retrieve the ith (obs, action) sample
    '''
    return self.expert_obs[i], self.expert_actions[i]


def collate_fn(batch, obs_mean, obs_std):
  '''
  Simple collate function that reformates a list of (obs, action) tuples into
  stacked torch Tensors, with observations (which are input into the model
  during training) normalized via input (mean, std) torch vectors
  '''
  all_expert_obs = []
  all_expert_actions = []
  for item in batch:
    obs, actions = item
    all_expert_obs += [obs]
    all_expert_actions += [actions]
  return (torch.stack(all_expert_obs) - obs_mean)/obs_std, torch.stack(all_expert_actions)
