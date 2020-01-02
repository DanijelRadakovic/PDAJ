from datetime import datetime
import os
import socket
import subprocess
import time

from celery import chain, chord
from celery.exceptions import Reject
import numpy as np
import csv

from ..app import app
from .worker import solve


# Monitoring tasks

@app.task
def monitor_queues(ignore_result=True):
    server_name = app.conf.MONITORING_SERVER_NAME
    server_port = app.conf.MONITORING_SERVER_PORT
    metric_prefix = app.conf.MONITORING_METRIC_PREFIX

    queues_to_monitor = ('server', 'worker')

    output = subprocess.check_output('rabbitmqctl -q list_queues name messages consumers', shell=True)
    lines = (line.split() for line in output.splitlines())
    data = ((queue, int(tasks), int(consumers)) for queue, tasks, consumers in lines if queue in queues_to_monitor)

    timestamp = int(time.time())
    metrics = []
    for queue, tasks, consumers in data:
        metric_base_name = "%s.queue.%s." % (metric_prefix, queue)

        metrics.append("%s %d %d\n" % (metric_base_name + 'tasks', tasks, timestamp))
        metrics.append("%s %d %d\n" % (metric_base_name + 'consumers', consumers, timestamp))

    sock = socket.create_connection((server_name, server_port), timeout=10)
    sock.sendall(''.join(metrics))
    sock.close()


# Recording the experiment status

def get_experiment_status_filename(status):
    return os.path.join(app.conf.STATUS_DIR, status)


def get_experiment_status_time():
    """Get the current local date and time, in ISO 8601 format (microseconds and TZ removed)"""
    return datetime.now().replace(microsecond=0).isoformat()


@app.task
def record_experiment_status(status):
    with open(get_experiment_status_filename(status), 'w') as fp:
        fp.write(get_experiment_status_time() + '\n')


# Seeding the computations

def gen_simulation_model_params(theta_resolution):
    search_space = np.linspace(0, 2 * np.pi, theta_resolution)
    for theta1_init in search_space:
        for theta2_init in search_space:
            yield theta1_init, theta2_init


@app.task
def simulate_pendulum():
    if os.path.exists(get_experiment_status_filename('started')):
        raise Reject('Computations have already been seeded!')

    record_experiment_status.si('started').delay()

    theta_resolution = app.conf.RESOLUTION
    dt = app.conf.DT
    tmax = app.conf.TMAX
    L1 = app.conf.L1
    L2 = app.conf.L2
    m1 = app.conf.M1
    m2 = app.conf.M2
    results_path = app.conf.RESULTS_PATH

    chord(
        (solve.s(L1, L2, m1, m2, tmax, dt, np.array([theta1_init, 0, theta2_init, 0]), theta1_init, theta2_init)
         for theta1_init, theta2_init in gen_simulation_model_params(theta_resolution)),
        store_results.s(results_path)
    ).delay()


@app.task
def store_results(results, file):
    with open(file, 'w') as file:
        writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

        # write header
        writer.writerow(['theta1_init', 'theta2_init', 'theta1', 'theta2'])

        # write results
        writer.writerows(results)
    record_experiment_status('completed')
