import numpy as np
import random
import pandas as pd
import networkx as nx


class Cluster:
    def __init__(self, topology_file, node_file):
        switches = self.topology_parse_file(topology_file)
        self.node_dict = {}
        self.switches = self.create_nodes(switches, node_file)
        self.switches = switches
        self.topology = self.create_topology(self.switches)
        self.job_node_dict = {}
        self.resource = self.available_resource()
        self.switch_topology()
        self.node_index = {i: int(i[1:]) for i in list(self.node_dict.keys())}

    def switch_topology(self):
        switch_nodes = [n for n, d in self.topology.nodes(data=True) if d.get('type') == 'switch']
        switch_graph = self.topology.subgraph(switch_nodes)
        self.switch_index = {node: int(node[1:]) for node in switch_graph.nodes()}
        # self.switch_edge_list = [(self.switch_index[edge[0]], self.switch_index[edge[1]]) for edge in
        # switch_graph.edges()]

    def available_resource(self):
        free_node = 0
        free_cores = 0
        free_gpu = 0
        free_memory = 0
        job_place_holder = 0
        for node_id, node in self.node_dict.items():
            if len(node.tasks_dict) == 0:
                free_node += 1
            job_place_holder += node.job_place_holder
            free_cores += node.free_core
            free_gpu += node.free_gpu
            free_memory += node.free_memory
        # print(f'Free nodes:{free_node}, free cores:{free_cores}, free gpu:{free_gpu}, max_jobs:{job_place_holder}')
        return {'node':free_node, 'core':free_cores, 'gpu':free_gpu, 'memory':free_memory, 'max_jobs':job_place_holder}

    def allocation(self, job, node_list, time):
        running_list = []
        if job.balanced:
            for i in node_list:
                check_job_run = self.node_dict[i].run_task(job)
                if check_job_run:
                    running_list.append(i)
                else:
                    for j in running_list:
                        self.node_dict[j].release_task(job.id)
                    # print(f'job {job.id} allocation fail')
                    return False
            job.start_time = time
            self.job_node_dict[job.id] = (job, running_list)
            return True

    def check_allocate_list(self, job):
        if job.balanced:
            job.task_core = int(job.requested_core/job.requested_node)
            job.task_memory = job.requested_memory/job.requested_node
            job.task_gpu = int(job.requested_gpu/job.requested_node)
            possible_node = {}
            for i,j in self.node_dict.items():
                if j.free_core >= job.task_core:
                    if j.free_memory >= job.task_memory:
                        if j.free_gpu >= job.task_gpu:
                            possible_node[i] = j
            if len(possible_node.values()) >= job.requested_node:
                return True, possible_node
            else:
                return False, possible_node

    def check_earliest_runtime(self, job, time):
        node_dict = {}
        for key, node in self.node_dict.items():
            earliest_time = node.earliest_runtime(job, time)
            if earliest_time != -1:
                node_dict[key] = earliest_time
        if len(node_dict) >= job.requested_node:
            return sorted(list(node_dict.values()))[job.requested_node-1]
        else:
            return -1

    def create_nodes(self, switches, node_file):
        node_df = pd.read_csv(node_file)
        switch_id = 0
        node_id = 0
        self.max_memory = 0
        # Renaming switches
        for switch in switches:
            # Assigning new id for switches
            switch['id'] = f's{switch_id}'

            # If 'Nodes' field exists, rename nodes
            if 'Nodes' in switch:
                nodes_df = pd.DataFrame({'node_type':switch['Nodes']})
                merged_df = pd.merge(nodes_df, node_df, on='node_type', how='inner')
                merged_dict = merged_df.to_dict('records')
                for i in range(len(switch['Nodes'])):
                    # Assigning new id for nodes
                    merged_dict[i]['node_id'] = f'n{node_id}'
                    node = Node(switch['id'], merged_dict[i])
                    if node.memory > self.max_memory:
                        self.max_memory = node.memory
                    self.node_dict[f'{node.id}'] = node
                    switch['Nodes'][i] = {'node': node, 'node_id': f'n{node_id}'}
                    node_id += 1
            switch_id += 1
        switch_name_to_id = {switch['SwitchName']: switch['id'] for switch in switches}
        # Iterate over the switches and replace SwitchName in 'Switches' field with respective id
        for switch in switches:
            if 'Switches' in switch:
                switch['Switches'] = [switch_name_to_id[name] for name in switch['Switches'].split(',')]
        return switches

    def create_topology(self, switches):
        G = nx.Graph()

        # Add switches as nodes
        for switch in switches:
            G.add_node(switch['id'], attr_dict=switch, type='switch')

        # If there's a 'Switches' field, add edges between switches
        for switch in switches:
            if 'Switches' in switch:
                for linked_switch in switch['Switches']:
                    G.add_edge(switch['id'], linked_switch, usage=0)

        # Add nodes connected to each switch, if 'Nodes' key is present
        for switch in switches:
            if 'Nodes' in switch:
                for node in switch['Nodes']:
                    #add a node attribute here
                    G.add_node(node['node_id'], compute_node=node['node'], type='node')
                    G.add_edge(switch['id'], node['node_id'], usage=0)

        return G

    def topology_parse_node_string(self, node_str):
        node_base, node_nums = node_str.split('[')
        node_nums = node_nums.rstrip(']')
        nodes = []
        for part in node_nums.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                nodes.extend(f"{node_base}{i:03d}" for i in range(start, end + 1))
            else:
                nodes.append(f"{node_base}{int(part):03d}")
        return nodes

    def topology_parse_file(self, file_name):
        switches = []
        with open(file_name, 'r') as f:
            for line in f:
                # Skip blank lines
                if line.strip() == '':
                    continue
                switch = {}
                parts = line.split()
                for part in parts:
                    key, value = part.split('=')
                    if key == 'Nodes':
                        nodes = []
                        for node_str in value.split(','):
                            if '[' in node_str:
                                nodes.extend(self.topology_parse_node_string(node_str))
                            else:
                                nodes.append(node_str)
                        switch[key] = nodes
                    else:
                        switch[key] = value
                switches.append(switch)
        return switches

    def release_job(self, job_id, time, back=False):
        (job, node_list) = self.job_node_dict[job_id]
        for i in node_list:
            self.node_dict[i].release_task(job_id)
        check_release = self.job_node_dict.pop(job_id, 'Not found')
        if not back:
            job.end_time = time
        if check_release == 'Not found':
            # print(f'release error job id {job_id} in node {self.id}')
            return None
        else:
            return job

    def place_holder_list(self):
        list_place_holder = []
        for name, node in self.node_dict.items():
            if node.gpu_enable:
                job_place_holder = node.gpu
            else:
                job_place_holder = node.core
            list_place_holder.append(job_place_holder)
        self.list_place_holder = list_place_holder

    def get_state(self, time):
        cluster_state = []
        for name, node in self.node_dict.items():
            node_state = []
            if node.gpu_enable:
                job_place_holder = node.gpu
            else:
                job_place_holder = node.core
            node_state += [node.free_core, node.free_memory, node.free_gpu, self.node_index[node.id], self.switch_index[node.connect_switch]]
            job_state = []
            count = 0
            for id, job in node.tasks_dict.items():
                job_state += [job.task_core, job.task_memory, job.task_gpu, job.requested_time, time-job.start_time]
                count += 1
            for _ in range(job_place_holder-count):
                job_state += [0, 0, 0, 0, 0]
            node_state += job_state
            cluster_state += node_state
        return cluster_state

    def reset_communication(self):
        for u, v, attrs in self.topology.edges(data=True):
            attrs['usage'] = 0

    def compute_node_utilization(self, node):
        core_util = 1 - node.free_core / node.core if node.core > 0 else 0
        mem_util = 1 - node.free_memory / node.memory if node.memory > 0 else 0
        gpu_util = 1 - node.free_gpu / node.gpu if node.gpu > 0 else 0
        return (core_util + mem_util + gpu_util) / 3 if node.gpu_enable else (core_util + mem_util) / 2
        #return (core_util + gpu_util) / 2 if node.gpu_enable else core_util

    def reset(self):
        self.job_node_dict = {}
        for u, v, attrs in self.topology.edges(data=True):
            attrs['usage'] = 0
        for node in self.topology.nodes():
            if self.topology.nodes[node]['type'] == 'node':
                self.topology.nodes[node]['compute_node'].reset()

class Node:
    def __init__(self, switch, resource):
        self.id = resource['node_id']
        self.node_type = resource['node_type']
        if 'cpu_type' in resource:
            self.cpu_type = resource['cpu_type']
        else:
            self.cpu_type = None
        if 'cpu_frq' in resource:
            self.cpu_frq = resource['cpu_frq']
        else:
            self.cpu_frq = None
        self.core = int(resource['core'])
        self.free_core = self.core
        self.memory = resource['memory']
        self.free_memory = self.memory
        if resource['gpu'] != '(null)':
            self.gpu_enable = True
            self.gpu_type = resource['gpu']
            self.gpu = int(resource['gpu_number'])
            self.job_place_holder = self.gpu
            self.free_gpu = self.gpu
        else:
            self.gpu_enable = False
            self.gpu_type = None
            self.gpu = 0
            self.job_place_holder = self.core
            self.free_gpu = 0
        self.tasks_dict = {}
        self.connect_switch = switch

    def earliest_runtime(self, job, time):
        earliest_task = sorted(self.tasks_dict, key=lambda x: self.remain_time(self.tasks_dict[x], time))
        need_core = job.task_core - self.free_core
        need_gpu = job.task_gpu - self.free_gpu
        need_memory = job.task_memory - self.free_memory
        if need_core<=0 and need_gpu<=0 and need_memory<=0:
            return 0
        else:
            for i in earliest_task:
                run_job = self.tasks_dict[i]
                need_core -= run_job.task_core
                need_gpu -= run_job.task_gpu
                need_memory -= run_job.task_memory
                if need_core<=0 and need_gpu<=0 and need_memory<=0:
                    return self.remain_time(run_job, time)
            return -1

    def remain_time(self, job, time):
        remain = job.requested_time - (time - job.start_time)
        return remain

    def run_task(self, job):
        if job.balanced:
            if self.free_core >= job.task_core:
                self.free_core -= job.task_core
                if self.free_memory >= job.task_memory:
                    self.free_memory -= job.task_memory
                    if self.free_gpu >= job.task_gpu:
                        self.free_gpu -= job.task_gpu
                        self.tasks_dict[job.id] = job
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False

    def release_task(self, job_id):
        check_release = self.tasks_dict.pop(job_id, 'Not found')
        if check_release == 'Not found':
            pass
            # print(f'release error job id {job_id} in node {self.id}')
        elif check_release.balanced:
            self.free_core += check_release.task_core
            self.free_memory += check_release.task_memory
            self.free_gpu += check_release.task_gpu

    def reset(self):
        self.free_core = self.core
        self.free_memory = self.memory
        self.free_gpu = self.gpu
        self.tasks_dict = {}
