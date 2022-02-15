import os
import yaml

yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GAIC_params.yaml')
assert os.path.exists(yaml_path), yaml_path

def load_yaml_params():
    with open(yaml_path, 'r') as yaml_file:
        params = yaml.full_load(yaml_file.read())
        return params

def refresh_yaml_params(args):
    yaml_params = load_yaml_params()
    for arg in vars(args):
        # print(arg, type(arg), getattr(args, arg))
        assert arg in yaml_params, arg
        yaml_params[arg] = getattr(args, arg)

    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(yaml_params, yaml_file)

class Config:
    data_root = '../../dataset'
    GAIC_folder = os.path.join(data_root, 'GAICD')

    def __init__(self):
        self.refresh_params()

    def refresh_params(self):
        self.load_params_from_yaml()
        self.generate_path()

    def load_params_from_yaml(self):
        # add parameters from yaml file
        names = self.__dict__
        params = load_yaml_params()
        for k, v in params.items():
            # print(v, type(v))
            names[k] = v

    def generate_path(self):
        prefix = 'GAIC-{}-re{}dim'.format(self.backbone, self.reddim)
        exp_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments')
        exp_name = prefix
        exp_path = os.path.join(exp_root, prefix)
        while os.path.exists(exp_path):
            index = os.path.basename(exp_path).split(prefix)[-1].split('repeat')[-1]
            try:
                index = int(index) + 1
            except:
                index = 1
            exp_name = prefix + ('_repeat{}'.format(index))
            exp_path = os.path.join(exp_root, exp_name)
        # print('Experiment name {} \n'.format(os.path.basename(exp_path)))
        self.exp_name = exp_name
        self.exp_path = exp_path
        self.checkpoint_dir = os.path.join(exp_path, 'checkpoints')
        self.log_dir = os.path.join(exp_path, 'logs')
        self.code_dir = os.path.join(exp_path, 'code')

    def create_path(self):
        print('Create experiment directory: ', self.exp_path)
        os.makedirs(self.exp_path)
        os.makedirs(self.checkpoint_dir)
        os.makedirs(self.log_dir)
        os.makedirs(self.code_dir)

cfg = Config()