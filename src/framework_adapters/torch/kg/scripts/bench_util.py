import paramiko
import itertools
import subprocess
import concurrent


def GenProduct(config):
    keys, values = zip(*config.items())
    permutations_config = [dict(zip(keys, v))
                           for v in itertools.product(*values)]
    return permutations_config


def GenBinding(config_list):
    r = []
    for each in config_list:
        r += GenProduct(each)
    return r



def LocalExecute(command, path, print_show=True):
    import re
    print_command = re.sub(r' +', ' ', command)
    # print_command = command.replace('\t', ' ')
    if print_show:
        print(f"===Local=== {print_command}")
    if path == '':
        subprocess.run(command, shell=True, check=True)
    else:
        subprocess.run(f'cd {path}; {command}', shell=True, check=True)


def RemoteExecute(server, command, path, print_show=True):
    import re
    print_command = re.sub(r' +', ' ', command)
    # print_command = command.replace('\t', ' ')
    if print_show:
        print(f"==={server}=== {print_command}")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server)

    import os
    temp = os.getenv('LC_DEBUG_XMH')
    environment_dict = dict()
    if temp == "LC_DEBUG_XMH":
        print("set environment to xmh debug")
        environment_dict["LC_DEBUG_XMH"] = "LC_DEBUG_XMH"

    if path == '':
        stdin, stdout, stderr = client.exec_command(
            command, environment=environment_dict)
    else:
        stdin, stdout, stderr = client.exec_command(
            f'cd {path}; {command}', environment=environment_dict)

    stdout_iter = iter(stdout.readline, '')
    stderr_iter = iter(stderr.readline, '')

    from itertools import zip_longest

    for out, err in zip_longest(stdout_iter, stderr_iter):
        if out:
            if print_show:
                print(out.strip())
        if err:
            if print_show:
                print(err.strip())

    # for line in stdout:
    #     print(line, end='')
    client.close()
    return stdout.channel.recv_exit_status()


def Pnuke(servers, pattern):
    print(f"==={servers}=== Pnuke {pattern}")
    if type(servers) is not list:
        servers = [servers]
    import subprocess
    import concurrent.futures

    def command_fn(host):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host)
        ret = 0
        while ret == 0:
            command = f"ps aux |grep {pattern}| grep -v grep | awk '{{print $2}}' | xargs kill -9"
            stdin, stdout, stderr = client.exec_command(
                command
            )
            ret = stdout.channel.recv_exit_status()
        client.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        for each in servers:
            executor.submit(command_fn, each)


def disjoint_dicts_to_one_dict(dicts):
    a = dicts[0].copy()
    for i in range(1, len(dicts)):
        a.update(dicts[i])
    return a


def StringnizeConfig(config):
    for each_k in config.keys():
        config[each_k] = [str(each) for each in config[each_k]]


def PreprocessConfig(config):
    # StringnizeConfig(config)
    if 'binding' in config:
        config_binding = config['binding']
        del config['binding']
        permutations_binding_config = GenBinding(config_binding)
        permutations_config = GenProduct(config)
        permutations_config = itertools.product(
            permutations_config, permutations_binding_config, )

        # [(dictA, dictB), (dictA, dictB), (dictA, dictB),]
        permutations_config = [disjoint_dicts_to_one_dict(each)
                               for each in permutations_config]
        return permutations_config
    else:
        return GenProduct(config)

        
def ParallelSSH(hosts, command):
    print(hosts, command)
    SSH = "ssh -o StrictHostKeyChecking=no"

    def command_fn(host):
        subprocess.run(
            f'''{SSH} {host} "{command}"''', shell=True, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        for each in hosts:
            executor.submit(command_fn, each)
