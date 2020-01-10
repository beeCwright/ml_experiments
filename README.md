# ml-experiments
This repository contains python code for managing experiment recommendations. It is built on top of GPyOpt, to perform Bayesian Optimizastion over a set of hyperparameters. It uses MongoDB as a central store of all experiments and results, allowing any number to nodes to asynchronously join an experiment and to checkout and evaluate trials.

---


### Installation

Install the GPyOpt repository, this requires you have gcc installed.

```
$ pip3 install git+https://github.com/SheffieldML/GPyOpt.git
```

### MongoDB Configuration

Pull down the latest mongo docker

```
sudo docker pull mongo
```

 - Generate certificates, making sure to replace the Common Name (CN) attribute with the host name where mongo will be running.

``` sh
# Generate Root Certificate and Root Key
$ openssl req -out rootCA.pem -keyout rootKey.pem -new -x509 -days 3650 -nodes -subj "/C=US/ST=MA/O=CCDS/CN=root"

# Generate Server key and Signed Certificates
$ echo "00" > file.srl
$ openssl genrsa -out server.key 2048
$ openssl req -key server.key -new -out server.req -subj  "/C=US/ST=MA/O=CCDS/CN=d7920-12.ccds.io"
$ openssl x509 -req -in server.req -CA rootCA.pem -CAkey rootKey.pem -CAserial file.srl -out server.crt -days 3650
$ cat server.key server.crt > server.pem
$ openssl verify -CAfile ca.pem server.pem

# Generate Client Key and Signed Certificates
$ openssl genrsa -out client.key 2048
$ openssl req -key client.key -new -out client.req -subj "/C=US/ST=MA/O=CCDS/CN=d7920-12.ccds.io"
$ openssl x509 -req -in client.req -CA rootCA.pem -CAkey rootKey.pem -CAserial file.srl -out client.crt -days 3650
$ cat client.key client.crt > client.pem
$ openssl verify -CAfile rootCA.pem client.pem
```

 - Start a mongo container.
 - Make sure the generated keys are mounted into the container
 - Forward a host port to the containers internal port 27017


```
ex)
$ sudo docker run -dit -p 27021:27017 -v /path/to/keys/dir:/mnt/certs/ mongo /bin/bash
```

 - Start the mongo server
 
```
> mongod --bind_ip_all --sslMode requireSSL --sslPEMKeyFile /mnt/certs/server.pem --sslCAFile /mnt/certs/rootCA.pem 
```

 -  Make sure to update the .yaml config file with the appropriate database configurations after starting the mongo server

### Usage

 - Create a controller instance which will read a yaml configuration file, and get the latest experiment updates, or create an experiment entry if none exists.

``` python
import ml_experiments as me

config = './ml_experiments/demo_config_small.yaml'
controller = me.ExperimentController(config)
```

 - The controller bundles the hyperparameters into the format GPyOpt requires.

``` python
controller.bounds
[{'name': 'learning_rate', 'type': 'discrete', 'domain': (0.005, 0.001)},
 {'name': 'num_convs', 'type': 'discrete', 'domain': (1, 2)},
 {'name': 'network', 'type': 'discrete', 'domain': (0, 1, 2, 3, 4)},
 {'name': 'dropout', 'type': 'discrete', 'domain': (0, 0.1, 0.2)},
 {'name': 'kernel', 'type': 'discrete', 'domain': (3, 5)},
 {'name': 'normalize', 'type': 'discrete', 'domain': (0, 1)},
 {'name': 'vertical_flip', 'type': 'discrete', 'domain': (0, 1)}]
```

 - Upon experiment creation, a set of warm-up trials are generated and used as the initial recommendations.  
 - After `num_warm_up` iterations have passed, the GPyOpt library is used to recommend the next trial.

``` python
local_iter = 0
current_iter = controller.next_iter
while (current_iter < controller.max_iter) and (local_iter < controller.max_local_iter):

    # Get the next trial, and increment the global next_iter
    this_trial = controller.get_next_suggestion()

    # Evaluate (custom code goes here)
    loss = evaluate(this_trial)

    # Send the update back to the controller
    controller.update_design(this_trial, [loss])

    # Claim the next iteration for this local instance
    current_iter = controller.next_iter
    local_iter += 1
```