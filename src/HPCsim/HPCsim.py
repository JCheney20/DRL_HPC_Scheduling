import numpy as np
import copy
import pandas as pd
import time
import os

from HPCsim.Cluster import *
from HPCsim.Evaluator import *
from HPCsim.Queue import *
from HPCsim.Scheduler import *
from HPCsim.Trace_Reader import *
import gymnasium as gym


class ENV_allocator(gym.Env):
    def __init__(self,
                env,
                seed=None):
        self.env = env
        super().reset(seed=seed)
        super().__init__()
        action_dim = self.env.cluster.resource['node']
        self.waiting_time_list = []
        self.agent_node_list = []
        self.action_space = gym.spaces.Discrete(action_dim)
        # cluster state: node * 8 + job_place * 6
        # queue state: window * 7 + 1
        self.observation_space = gym.spaces.Dict(
            {
                'cluster': gym.spaces.Box(0, max(self.env.trace.longest_requested_time, self.env.cluster.max_memory),
                                          shape=(5 * self.env.cluster.resource['node'] + 5 * self.env.cluster.resource[
                                              'max_jobs'],),
                                          dtype=float),
                'queue': gym.spaces.Box(0, max(self.env.trace.longest_requested_time, self.env.cluster.max_memory),
                                        shape=(self.env.window_size * 7 + 2,), dtype=float),
            }
        )

    def step(self, action):
        action = [action]
        reward = 0
        meta_reward = 0
        meta_obs = None
        success = False
        done = False
        forward_time = False
        if len(self.env.event_queue) + len(self.env.queue.job_queue) == 0:
            epi_done = True
        else:
            epi_done = False
        if self.env.allocation_counts == self.env.info['selected_job'].requested_node:
            self.agent_node_list = []

        self.agent_node_list.append(action[0])
        self.env.allocation_counts -= 1

        if self.env.allocation_counts == 0:
            node_list = []
            index = np.arange(self.env.cluster.resource['node'])
            for i in self.agent_node_list:
                node_list.append(f'n{index[i]}')
            success = self.env.cluster.allocation(self.env.info['selected_job'], node_list, self.env.time)
        if success:
            job_wait = self.env.time - self.env.info['selected_job'].system_submit
            self.env.waiting_time_list.append(job_wait)
            self.waiting_time_list.append(job_wait)
            if job_wait > self.env.max_waiting:
                self.env.max_waiting = job_wait
            self.env.allocated_job_count += 1
            self.env.queue.pop_sched_job(self.env.info['selected_job'])
            print(f"Job ID {self.env.info['selected_job'].id} has been allocated.")
            self.env.add_job_completion(self.env.info['selected_job'])
            self.env.sort_event_queue()
            queue, job_count, job_mask = self.env.queue.get_state(window=self.env.window_size, cluster=self.env.cluster,
                                                          tail=self.env.tail_size, time=self.env.time)
            while (job_count == 0) and not epi_done:
                self.env.forward_system_time()
                forward_time = True
                queue, job_count, job_mask = self.env.queue.get_state(window=self.env.window_size,
                                                               cluster=self.env.cluster,
                                                               tail=self.env.tail_size, time=self.env.time)
                if len(self.env.event_queue) + len(self.env.queue.job_queue) == 0:
                    epi_done = True
                else:
                    epi_done = False
            self.agent_node_list = []
            reward = self.get_reward()
            meta_obs, mask = self.env.get_state()
            self.env.current_valid_job = mask
            meta_reward = self.env.get_reward()
        obs = self.get_state(success)
        info = {'meta_new_obs':meta_obs, 'meta_reward':meta_reward, 'epi_done':epi_done, 'forward':forward_time, 'success':success}
        if epi_done:
            done = True
        return obs, reward, done, False, info

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.agent_node_list = []
        obs = self.get_state(True)
        self.waiting_time_list = []
        info = {}
        return obs, info

    def get_reward(self):
        reward = 0
        if len(self.waiting_time_list) == 100:
            for i in self.env.queue.job_queue:
                self.waiting_time_list.append(self.env.time-i.system_submit)
            average_waiting = np.mean(self.waiting_time_list)
            # self.waiting_time = 0.2*self.waiting_time + 0.8*average_waiting
            reward -= average_waiting
            self.waiting_time_list = []

        return reward

    def get_state(self, done):
        if done:
            queue, _, _ = self.env.queue.get_state(window=self.env.window_size, cluster=self.env.cluster,
                                                   tail=self.env.tail_size,
                                                   time=self.env.time)
            queue.append(-1)
            return {'cluster': self.env.cluster.get_state(self.env.time), 'queue': queue}
        else:
            allocate_node_name = []
            for i in self.agent_node_list:
                allocate_node_name.append(f'n{i}')
            selected_job = self.env.info['selected_job']
            time = self.env.time
            cluster_state = []
            for name, node in self.env.cluster.node_dict.items():
                node_state = []
                if node.gpu_enable:
                    job_place_holder = node.gpu
                else:
                    job_place_holder = node.core
                if name in allocate_node_name:
                    node_state += [node.free_core-selected_job.task_core,
                                   node.free_memory-selected_job.task_memory,
                                   node.free_gpu-selected_job.task_gpu,
                                   self.env.cluster.node_index[node.id],
                                   self.env.cluster.switch_index[node.connect_switch]]
                    job_state = []
                    count = 0
                    for id, job in node.tasks_dict.items():
                        job_state += [job.task_core,
                                      job.task_memory,
                                      job.task_gpu,
                                      job.requested_time,
                                      time - job.start_time]
                        count += 1
                    job_state += [selected_job.task_core,
                                  selected_job.task_memory,
                                  selected_job.task_gpu,
                                  selected_job.requested_time,
                                  0]
                    count += 1
                    for _ in range(job_place_holder - count):
                        job_state += [0, 0, 0, 0, 0]
                    node_state += job_state
                    cluster_state += node_state
                else:
                    node_state += [node.free_core,
                                   node.free_memory,
                                   node.free_gpu,
                                   self.env.cluster.node_index[node.id],
                                   self.env.cluster.switch_index[node.connect_switch]]
                    job_state = []
                    count = 0
                    for id, job in node.tasks_dict.items():
                        job_state += [job.task_core,
                                      job.task_memory,
                                      job.task_gpu,
                                      job.requested_time,
                                      time - job.start_time]
                        count += 1
                    for _ in range(job_place_holder - count):
                        job_state += [0, 0, 0, 0, 0]
                    node_state += job_state
                    cluster_state += node_state
            queue, _, _ = self.env.queue.get_state(window=self.env.window_size, cluster=self.env.cluster,
                                                   tail=self.env.tail_size,
                                                   time=self.env.time)
            queue[self.env.info['action']*6] -= len(self.agent_node_list)
            queue.append(self.env.info['action'])
            return {'cluster': cluster_state, 'queue': queue}

    def action_masks(self):
        selected_job = self.env.info['selected_job']
        if selected_job is not None:
            mask_nodes = []
            _, nodes = self.env.cluster.check_allocate_list(selected_job)
            node_name = list(nodes.keys())
            if self.agent_node_list:
                for i in self.agent_node_list:
                    if f'n{i}' in node_name:
                        node_name.remove(f'n{i}')
            for i in range(self.env.cluster.resource['node']):
                if f'n{i}' in node_name:
                    mask_nodes.extend([True])
                else:
                    mask_nodes.extend([False])
        else:
            mask_nodes = [False] * self.env.cluster.resource['node']
        return mask_nodes


class HPCsim(gym.Env):
    def __init__(self,
                 scheduler='fcfs',
                 allocator='topology_aware',
                 backfill_enable=True,
                 topology_file='topology/deeplearn_topology.txt',
                 node_file='topology/nodes.csv',
                 trace_file='deeplearn_job.csv',
                 trace_start=None,
                 allocate_weight=None,
                 partition=None,
                 window_size=100,
                 tail_size=0,
                 seed=None,
                 random_job=False):
        self.current_valid_job = None
        self.scheduler_factor = scheduler
        self.allocator_factor = allocator
        self.backfill_enable = backfill_enable
        self.topology_file = topology_file
        self.node_file = node_file
        self.trace_file = trace_file
        self.allocate_weight = allocate_weight
        self.cluster = Cluster(topology_file, node_file)
        self.cluster.place_holder_list()
        self.queue = Queue()
        self.scheduler = Scheduler(default=scheduler)
        self.allocator = Allocator(default=allocator)
        self.random_job = random_job
        self.trace = Trace(trace_file, partition=partition, start=trace_start, random_job=random_job)
        self.evaluator = Evaluator(scheduler, allocator)
        self.time = 0
        self.init_time = self.trace.first_arrival
        self.event_queue = {self.time: [('arrival',)]}
        # First job arrival event
        self.received_job_count = 0
        self.allocated_job_count = 0
        self.completed_job_count = 0
        self.forward_count = 0
        # cluster.reset()
        super().reset(seed=seed)
        super().__init__()
        self.window_size = window_size
        self.tail_size = tail_size
        action_dim = self.window_size + 1
        self.action_space = gym.spaces.Discrete(action_dim)
        # cluster state: node 8 + job_place*6
        # queue state: window*7 + 1
        self.observation_space = gym.spaces.Dict(
            {
                'cluster': gym.spaces.Box(0, max(self.trace.longest_requested_time, self.cluster.max_memory),
                                          shape=(5 * self.cluster.resource['node'] + 5 * self.cluster.resource['max_jobs'],),
                                          dtype=float),
                'queue': gym.spaces.Box(0, max(self.trace.longest_requested_time, self.cluster.max_memory),
                                        shape=(self.window_size * 7 + 1,), dtype=float),
            }
        )
        self.waiting_time = 0
        self.max_waiting = 0
        self.waiting_time_list = []
        self.action_counts = 0

    def get_state(self):
        queue, _, mask = self.queue.get_state(window=self.window_size, cluster=self.cluster, tail=self.tail_size,
                                        time=self.time)
        return {'cluster': self.cluster.get_state(self.time), 'queue': queue}, mask

    def step(self, action):
        self.action_counts += 1
        action = [action]
        reward = 0
        forward_time = False
        allocation = False
        # find selected job
        if action[0] == self.window_size:
            forward_time = True
            selected_job = None
            self.forward_count += 1
            reward = self.get_reward()
        elif len(self.queue.job_queue) > self.window_size:
            self.forward_count = 0
            if action[0] < self.window_size - self.tail_size:
                selected_job = self.queue.job_queue[action[0]]
            else:
                index = action[0] - self.window_size
                selected_job = self.queue.job_queue[index]
        elif action[0] < len(self.queue.job_queue):
            self.forward_count = 0
            selected_job = self.queue.job_queue[action[0]]
        else:
            forward_time = True
            selected_job = None
        # process selected job
        if selected_job is not None:
            can, node_dict = self.cluster.check_allocate_list(selected_job)
            if can:
                allocation = True
            else:
                forward_time = True

        if forward_time:
            if len(self.event_queue) + len(self.queue.job_queue) == 0:
                done = True
            else:
                done = False
                if len(self.event_queue) > 0:
                    self.forward_system_time()
            queue, job_count, mask = self.queue.get_state(window=self.window_size, cluster=self.cluster, tail=self.tail_size, time=self.time)
            # if there is no job can be allocated by the agent, we continue forwarding
            while (job_count == 0) and not done:
                if len(self.event_queue) > 0:
                    self.forward_system_time()
                queue, job_count, mask = self.queue.get_state(window=self.window_size, cluster=self.cluster,
                                                        tail=self.tail_size, time=self.time)
                if len(self.event_queue) + len(self.queue.job_queue) == 0:
                    done = True
                else:
                    done = False
            obs = {'cluster': self.cluster.get_state(self.time), 'queue': queue}
        else:
            obs, mask = self.get_state()
            if len(self.event_queue) + len(self.queue.job_queue) == 0:
                done = True
            else:
                done = False
        if allocation:
            self.allocation_counts = copy.deepcopy(selected_job.requested_node)
            queue, _, _ = self.queue.get_state(window=self.window_size, cluster=self.cluster, tail=self.tail_size,
                                                  time=self.time)
            queue.append(action[0])
            allocation_obs =  {'cluster': self.cluster.get_state(self.time), 'queue': queue}
        else:
            allocation_obs = None
        self.info = {'allocation': allocation, 'selected_job':selected_job, 'action':action[0], 'forward':forward_time}
        info = {'allocation': allocation, 'selected_job': selected_job, 'action': action[0], 'allocation_obs':allocation_obs, 'forward':forward_time}
        self.current_valid_job = mask

        return obs, reward, done, False, info

    def get_reward(self):
        reward = 0
        if self.action_counts == 512:
            for i in self.queue.job_queue:
                self.waiting_time_list.append(self.time-i.system_submit)
            average_waiting = np.mean(self.waiting_time_list)
            # self.waiting_time = 0.2*self.waiting_time + 0.8*average_waiting
            # reward -= self.waiting_time
            reward -= average_waiting
            self.waiting_time_list = []
            self.action_counts = 0

        return reward

    def action_masks(self):
        mask = self.current_valid_job
        if self.forward_count >= 10:
            mask[-1] = False
        return mask

    def run(self):
        start = time.time()
        self.result_dict = {}
        self.result_dict['time'] = []
        self.result_dict['node_utilization'] = []
        self.result_dict['cpu_utilization'] = []
        self.result_dict['gpu_utilization'] = []
        self.result_dict['mem_utilization'] = []
        self.last_time = 0
        self.check_queue_length = []
        os.makedirs('result', exist_ok=True)
        while (len(self.event_queue) + len(self.queue.job_queue)) > 0:
            self.check_queue_length.append(len(self.queue.job_queue))
            self.job_schedule_allocation()
            current_resource = self.cluster.available_resource()
            node_utilization = (self.cluster.resource['node'] - current_resource['node']) / self.cluster.resource['node']
            cpu_utilization = (self.cluster.resource['core'] - current_resource['core']) / self.cluster.resource['core']
            gpu_utilization = ((self.cluster.resource['gpu'] - current_resource['gpu']) / self.cluster.resource['gpu'] if self.cluster.resource['gpu'] > 0 else 0.0)
            mem_utilization = (self.cluster.resource['memory'] - current_resource['memory']) / self.cluster.resource['memory']
            self.result_dict['time'].append(self.time)
            self.result_dict['node_utilization'].append(node_utilization)
            self.result_dict['cpu_utilization'].append(cpu_utilization)
            self.result_dict['gpu_utilization'].append(gpu_utilization)
            self.result_dict['mem_utilization'].append(mem_utilization)
            self.last_time = self.time
            self.forward_system_time()
        end = time.time()
        print(f"Running time: {end - start:.4f} seconds")
        pd.DataFrame(self.result_dict).to_csv(f'result/{self.scheduler_factor}+{self.allocator_factor}.csv', index=False)

    def forward_system_time(self):
        next_system_time = list(self.event_queue.items())[0][0]
        self.time = next_system_time
        self.process_event()

    def process_event(self):
        events = self.event_queue[self.time]
        for event in events:
            if event[0] == 'arrival':
                self.job_arrival()
            elif event[0] == 'complete':
                self.job_completion(event[1])
        self.event_queue.pop(self.time, 'Event Queue Pop error')

    def job_completion(self, job_id):
        completed_job = self.cluster.release_job(job_id, self.time, back=False)
        print(f'Job completed ID {job_id} at time {self.time}.')
        self.completed_job_count += 1
        self.evaluator.check_in_job(completed_job)

    def job_arrival(self):
        arrival_list = self.trace.get_next_jobs()
        try:
            arrival_check = int((arrival_list[0].submit - self.init_time).total_seconds())
        except AttributeError:
            arrival_check = int(arrival_list[0].submit - self.init_time)
        if arrival_check == self.time:
            self.queue.receive_job(arrival_list, self.time)
            for i in arrival_list:
                print(f'Job arrival ID {i.id} at time {self.time}')
            self.received_job_count += len(arrival_list)
            if self.trace.job_heap:
                next_arrival = self.trace.get_next_arrival_time()
                try:
                    next_arrival_time = int((next_arrival - self.init_time).total_seconds())
                except AttributeError:
                    next_arrival_time = int((next_arrival - self.init_time))
                if next_arrival_time in self.event_queue:
                    self.event_queue[next_arrival_time].append(('arrival',))
                else:
                    self.event_queue[next_arrival_time] = [('arrival',)]
                self.sort_event_queue()

    def job_schedule_allocation(self):
        while True:
            if self.queue.job_queue:
                can_allocate, prior_job, info = self.scheduler.scheduler(self.queue.job_queue, self.cluster, self.time)
                if can_allocate:
                    node_list = self.allocator.allocator(prior_job, info, self.cluster.topology,
                                                         weight=self.allocate_weight)
                    success = self.cluster.allocation(prior_job, node_list, self.time)
                    if success:
                        self.allocated_job_count += 1
                        self.queue.pop_sched_job(prior_job)
                        print(f'Job ID {prior_job.id} has been allocated.')
                        self.add_job_completion(prior_job)
                        self.sort_event_queue()
                    else:
                        break
                elif self.backfill_enable:
                    backfill_dict = self.scheduler.backfill(prior_job, info, self.time, self.cluster, self.allocator)
                    if backfill_dict:
                        self.allocated_job_count += len(backfill_dict.keys())
                        print(f'Backfilled job ID: {list(backfill_dict.keys())}.')
                        for _, backfilled in backfill_dict.items():
                            self.queue.pop_sched_job(backfilled)
                            self.add_job_completion(backfilled)
                        self.sort_event_queue()
                    else:
                        break
                else:
                    break
            else:
                print('Job queue is empty.')
                break

    def add_job_completion(self, job):
        complete_time = int(job.total_run + self.time)
        if complete_time in self.event_queue:
            self.event_queue[complete_time].append(('complete', job.id))
        else:
            self.event_queue[complete_time] = [('complete', job.id)]

    def sort_event_queue(self):
        self.event_queue = {k: self.event_queue[k] for k in sorted(self.event_queue)}

    def job_waiting_time(self):
        return self.evaluator.waiting_time()

    def turnaround_time(self):
        return self.evaluator.average_turnaround()

    def utilization(self):
        return self.evaluator.average_utilization(self.cluster.resource, self.time)

    def reset(self,
              scheduler=None,
              allocator=None,
              backfill_enable=None,
              allocate_weight=None,
              seed=None):
        super().reset(seed=seed)
        self.time = 0
        if scheduler:
            self.scheduler_factor = scheduler
        if allocator:
            self.allocator_factor = allocator
        if backfill_enable:
            self.backfill_factor = backfill_enable
        self.evaluator.reset(scheduler, allocator, resource=self.cluster.resource, time=self.time)
        self.scheduler = Scheduler(default=self.scheduler_factor)
        self.allocator = Allocator(default=self.allocator_factor)
        self.cluster.reset()
        self.trace.reset()
        self.queue.reset()
        if allocate_weight:
            self.allocate_weight = allocate_weight
        self.init_time = self.trace.first_arrival
        self.event_queue = {self.time: [('arrival',)]}
        # First job arrival event
        self.received_job_count = 0
        self.allocated_job_count = 0
        self.completed_job_count = 0
        self.waiting_time = 0
        self.max_waiting = 0
        self.waiting_time_list = []
        self.action_counts = 0
        self.forward_count = 0
        self.forward_system_time()
        obs, mask = self.get_state()
        self.current_valid_job = mask
        info = {}
        return obs, info
