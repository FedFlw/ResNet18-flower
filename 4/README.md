
## Setup

While there are different ways of setting up your Python environment, here I'll assume a [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installation is reachable from a standard bash/zsh terminal. These are the steps to setup the environment:

```bash
# create environment and activate
conda create -n flowermonthly python=3.10 -y
source activate flowermonthly

# install pytorch et al (you might want to adjust the command below depending on your platform/OS: https://pytorch.org/get-started/locally/)
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# install flower and other deps
pip install -r requirements.txt
```


## Flower + Hydra for beginners

This section provides an introductory look into [`Hydra`](https://hydra.cc/) in the context of Federated Learning with Flower. You can run the code just as originally designed (no change in parameters) like this:
```bash
python main.py # this will use the default configs (i.e. everything defined in conf/base.yaml)
```

With Hydra you can easily change whatever you need from the config file without having to add a new `argparse` argument each time. For example:
```bash
python main.py server.client_per_round=5 # will use 5 clients per round instead of the default 10
python main.py dataset.lda_alpha=1000 # will use LDA alpha=1000 (making it IID) instead of the default value (1.0)

python main.py client.resources.num_cpus=4 # allocates 4 CPUs to each client (instead of the default 2 as defined in conf/client/cpu_client.yaml -- cpu_client is the default client to use as defined in conf/base.yaml->default.client)
```

In some settings, you might want to make more substantial changes to the default config. For that, even though you could probably still doing from the command line, it can get messy... Instead, you can directly replace entire structures in your config with others. For example, let's say you want to change your entire client definition from the default one (check it in `conf/client/cpu_client.yaml`). You'll need to create a new yaml file, respecting the expected structure and place it in the same directory as `cpu_client.yaml`. This is exactly what I did with `gpu_client.yaml`. You can use the latter client as follows:
```bash
python main.py # will use the default `cpu_client.yaml`

# note that you'll need a GPU for this
python main.py client=gpu_client # will use the client as defined in `conf/client/gpu_client.yaml`

yaml
# this defines a top-level config (just like base.yaml) does but with changes to the `defaults` and the FL setup parameterised in `server:`

defaults: # i.e. configs used if you simply run the code as `python main.py`
  - client: gpu_client # this points to the file: client/cpu_client.yaml
[...] # rest of the necessary elements: dataset, server, misc
```

The above config can be found in `conf/base_v2.yaml`.

## Different Experiments in this repo

This repo contains a collection of experiments, all parameterised via the Hydra config structure inside `conf/`. The current list of experiments are:

```bash
# runs the default experiment using standard FedAvg
python main.py

# you can change the default strategy to point to another one.
# This example points to that in config `conf/strategy/strategy_model_saving.yaml`
# it essentially shows how you can keep track of elements in your experiment
# and retrieve (e.g. for saving them to disk) once simulation is completed
python main.py strategy=strategy_model_saving

# Overrides the config hardcoded in the @hydra decorator in the main.py to point to `conf/base_v2`
# this experiments uses the CustomFedAvg and shows how you can change the behaviour of how
# clients are sampled, udpates are aggregated, and the frequency at which the global model is evaluated
python main.py --config-name=base_v2
# If you'd like to run it with the cpu_client instead
python main.py --config-name=base_v2 client=cpu_client

# Run the `conf/base_kd.yaml` config to test a simple federated distillation setting
# where the teacher is first pre-trained in the server and send to the clients along with
# the smaller student network (i.e. the one that's being trained in a federated manner)
python main.py --config-name=base_kd

# and if you still want to override some of the settings you can totally do so as shown earlier in the readme
# will change the temperature used in FlowerClientWithKD's fit() method
python main.py --config-name=base_kd strategy.kd_config.student_train.temperature=5 
```
