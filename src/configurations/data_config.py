# Data , files and paths handling
import numpy as np
import os, glob, pprint, itertools
import inspect
import pickle
from datetime import datetime
import subprocess

def sanitize(v):
    "Round float values to 4 digits"
    if isinstance(v, float):
        return f"{v:.4g}"  # round to 4 significant digits
    return str(v).replace(' ', '-')

def dict_to_name(dic, sep='_', key_value_sep=''):
    """Convert a dictionary of parameters into a consistent string."""
    items = sorted(dic.items())  # sort for reproducibility
    return sep.join(f"{k}{key_value_sep}{sanitize(v)}" for k, v in items)

def make_paths_general(base_dir,subfolder_names, file_name ,dic = None ,ext=None):
    """
    Create a consistent folder structure for experiment results.
    """
    # Turn parameter dictionary into a friendly name
    if dic is not None:
        param_str = dict_to_name(dic)
        filename = f'{file_name}_{param_str}'
    else:
        param_str = ''
        filename = file_name
    
    if ext is not None: 
        filename += '.' + ext
    # Build full paths
    if isinstance(subfolder_names,list):
        dir_path = os.path.join(base_dir, *subfolder_names)
    else:
        dir_path = os.path.join(base_dir, subfolder_names)

    file_path = os.path.join(dir_path,filename)
    # normalize before checking or opening
    # file_path = os.path.normpath(file_path)
    # dir_path = os.path.normpath(dir_path)
    # normalize to absolute canonical paths (one-line patch)
    file_path, dir_path = map(lambda p: os.path.abspath(os.path.normpath(p)), (file_path, dir_path))
    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)
    return file_path , filename, dir_path

def make_data_paths(file_name, experiment_name= '', params=None,ext='pkl',base_dir='./data'):

    if params is None or 'fixed' not in params.keys() :
        subfolder_names = experiment_name
        params_file = params
    elif 'fixed' in params.keys() and 'variable' in params.keys():
        subfolder_names = [experiment_name,dict_to_name(params['fixed'])]
        params_file = params['variable']

    file_path, filename , dir_path = make_paths_general(base_dir,subfolder_names, file_name ,params_file ,ext=ext)

    return file_path , filename, dir_path


def make_params_dict(*names):
    "Make a dictionary with the variables in names and the values they have."
    caller_locals = inspect.currentframe().f_back.f_locals
    if len(names) == 1:
        dict = {name: caller_locals[name] for name in names[0]}
    elif len(names) == 2:
        dict = {'fixed': {name: caller_locals[name] for name in names[0]},
                'variable': {name: caller_locals[name] for name in names[1]}}
    else:
        raise ValueError(f'Revise names provided, maximum 2 inputs. Got {names = }')
    return dict

def save_fig(fig, file_name, params=None, show = True, ext='png',base_dir="../plots",date=False,bbox_inches='tight',dpi=200):
    subfolder_names = datetime.now().strftime("%Y-%m") if date else ''
    file_path, filename , dir_path = make_paths_general(base_dir,subfolder_names, file_name ,params ,ext=ext)
    fig.savefig(file_path, dpi=dpi,bbox_inches=bbox_inches)
    if show:
        print(f'Figure saved on {dir_path} as {filename}')



def save_data(data, file_name, experiment_name= '', params=None,show=True,ext='pkl',base_dir='./data'):

    file_path, filename , dir_path = make_data_paths(file_name, experiment_name, params,ext,base_dir)
    
    if ext == 'txt':
        if not isinstance(data,np.ndarray):
            raise TypeError(f'Data must be np array, got {type(data)}.')
        elif data.ndim > 2:
            raise TypeError('Array must be at most 2D to save as txt.')
        else:
            np.savetxt(file_path,data)
            message = 'File saved with np.savetxt'
    
    elif ext == 'npy':
        if not isinstance(data,np.ndarray):
            raise TypeError(f'Data must be np array, got {type(data)}.')    
        else:
            np.save(file_path,data)
            message = 'File saved with np.save'

    else:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        message = 'File saved with pickle.dump'

    if show:
        print(f'{message} on {dir_path} as {filename}')



def load_data(file_name, experiment_name= '', params=None,show=True,ext='pkl',base_dir='../data'):
    
    if params is None or 'fixed' not in params.keys() :
        subfolder_names = experiment_name
        params_file = params
    elif 'fixed' in params.keys() and 'variable' in params.keys():
        subfolder_names = [experiment_name,dict_to_name(params['fixed'])]
        params_file = params['variable']

    file_path, filename , dir_path = make_paths_general(base_dir,subfolder_names, file_name ,params_file ,ext=ext)

    if ext == 'txt':
        data = np.loadtxt(filename)
        message = 'loaded with np.loadtxt'
    
    elif ext == 'npy':
        data = np.load(filename)
        message = 'loaded with np.load'

    else:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        message = 'loaded with pickle.load'

    if show:
        print(f'File {filename} {message} from {dir_path}')
    
    return data

def load_histogram_data(file_name, experiment_name= '', params=None,show=True,ext='txt',base_dir='./data'):
    """
    Load histogram data from file.
    Parameters:
    - file_name: str, name of the file
    - experiment_name: str, name of the experiment
    - params: dict, parameters for the data
    - show: bool, whether to print loading message
    - ext: str, file extension
    - base_dir: str, base directory for data
    Returns:
    - data: list of (2, N) numpy arrays
    Raises:
    - ValueError: if file format is incorrect
    """

    if params is None or 'fixed' not in params.keys() :
        subfolder_names = experiment_name
        params_file = params
    elif 'fixed' in params.keys() and 'variable' in params.keys():
        subfolder_names = [experiment_name,dict_to_name(params['fixed'])]
        params_file = params['variable']

    file_path, filename , dir_path = make_paths_general(base_dir,subfolder_names, file_name ,params_file ,ext=ext)

    
    # Load all the lines from the file 
    with open(file_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) % 2 != 0:
        raise ValueError(f'Expected an even number of non-empty lines in {file_path}, got {len(lines)}')

    # Save the data every 2 lines in an array (using numpy)
    # At the end the data will be a list of arrays
    data = []
    for i in range(0, len(lines), 2):
        x = np.fromstring(lines[i], sep=' ')
        y = np.fromstring(lines[i + 1], sep=' ')
        if x.size != y.size:
            raise ValueError(f'Line pair {i//2} has mismatched lengths: {x.size} vs {y.size}')
        data.append(np.vstack((x, y)))
    # Return the list of (2, N) arrays
    return data

    data = load_data(file_name, experiment_name, params, show, ext, base_dir)
    return data


def download_cluster_data(server_name,path_cluster,path_local,filename_cluster,filename_local=None,show=True):
    """
    Download data file from cluster to local machine using scp.
    Parameters:
    - server_name: str, name of the server ('ulysses' or 'peralba')
    - path_cluster: str, path on the cluster where the file is located
    - path_local: str, local path where to save the file
    - filename_cluster: str, name of the file on the cluster
    - filename_local: str, name of the file locally (if None, use same as cluster)
    - show: bool, whether to print success message
    Returns:
    - None
    Raises:
    - ValueError: if server_name is invalid   
    """

    # Define available servers
    servers = {'ulysses':'cerazova@frontend2.hpc.sissa.it:~/',
                'peralba':'cerazova@peralba.phys.sissa.it:/u/c/cerazova/'}
    if server_name not in ['ulysses','peralba']:
        raise ValueError('Invalid server name')
    else:
        server = servers[server_name]
    
    # If filename_local not provided use the same as in cluster
    if filename_local is None : 
        filename_local = filename_cluster

    # Construct paths
    cluster = os.path.join(server,path_cluster ,filename_cluster)
    local = os.path.join(path_local,filename_local)

    # Check if file exist locally
    if os.path.exists(local):
        print(f'File already exist: {local}')
        return

    # Run scp command to copy from cluster
    result = subprocess.run(['scp', cluster, local], capture_output=True, text=True)

    # Report
    if result.stderr:
        print(f'File not found: {cluster}')
    else:
        if show: 
            print(f'SUCCESS LOADING FILE: {filename_cluster}')
