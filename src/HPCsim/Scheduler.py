import heapq
import math
import networkx as nx
import time as ts

class Scheduler:
    def __init__(self, default='fcfs'):
        self.scheduler_map = {
            'fcfs': self.fcfs,
            'lcfs': self.lcfs,
            'sjf': self.sjf,
            'wfp3': self.wfp3,
            'unicep': self.unicep,
            'f_1': self.f_1,
            'f_2': self.f_2,
        }

        self.scheduler = self.scheduler_map.get(default, self.fcfs)

    # NOTE FOR FUTURE REFACTOR:
    # All 7 selector methods (fcfs, lcfs, sjf, wfp3, unicep, f_1, f_2) share identical
    # structure: build a heapq with a priority key, pop the top job, call
    # check_allocate_list, return True/False. Only the priority key expression differs.
    # These could be unified into a single _select(priority_fn) dispatcher:
    #
    #   def _select(self, job_queue, cluster, priority_fn):
    #       job_heap = []
    #       for counter, job in enumerate(job_queue):
    #           heapq.heappush(job_heap, (priority_fn(job), counter, job))
    #       _, _, prio_job = heapq.heappop(job_heap)
    #       can, node_dict = cluster.check_allocate_list(prio_job)
    #       return (True, prio_job, node_dict) if can else (False, prio_job, job_heap)
    #
    # Each selector would then be a one-liner calling self._select(..., lambda job: <key>).
    # Deferred to avoid risk of changing working simulation code.

    def fcfs(self, job_queue, cluster, time):
        job_heap = []
        counter = 0
        # Add jobs to the heap
        for job in job_queue:
            heapq.heappush(job_heap, (job.system_submit, counter, job))
            counter += 1

        _, _, prio_job = heapq.heappop(job_heap)
        can, node_dict = cluster.check_allocate_list(prio_job)
        if can:
            return True, prio_job, node_dict
        else:
            return False, prio_job, job_heap

    def lcfs(self, job_queue, cluster, time):
        job_heap = []
        counter = 0
        # Add jobs to the heap
        for job in job_queue:
            neg_timestamp = -job.system_submit
            heapq.heappush(job_heap, (neg_timestamp, counter, job))
            counter += 1

        _, _, prio_job = heapq.heappop(job_heap)
        can, node_dict = cluster.check_allocate_list(prio_job)
        if can:
            return True, prio_job, node_dict
        else:
            return False, prio_job, job_heap

    def sjf(self, job_queue, cluster, time):
        job_heap = []
        counter = 0
        # Add jobs to the heap
        for job in job_queue:
            heapq.heappush(job_heap, (job.requested_time, counter, job))
            counter += 1

        _, _, prio_job = heapq.heappop(job_heap)
        can, node_dict = cluster.check_allocate_list(prio_job)
        if can:
            return True, prio_job, node_dict
        else:
            return False, prio_job, job_heap

    def wfp3(self, job_queue, cluster, time):
        #-(waiting/req_time)^3*req_core
        job_heap = []
        counter = 0
        # Add jobs to the heap
        for job in job_queue:
            job_priority = -((time-job.system_submit)/job.requested_time)**3*job.requested_core
            heapq.heappush(job_heap, (job_priority, counter, job))
            counter += 1

        _, _, prio_job = heapq.heappop(job_heap)
        can, node_dict = cluster.check_allocate_list(prio_job)
        if can:
            return True, prio_job, node_dict
        else:
            return False, prio_job, job_heap

    def unicep(self, job_queue, cluster, time):
        #-waiting/(log_2(req_core)*req_time)
        job_heap = []
        counter = 0
        # Add jobs to the heap
        for job in job_queue:
            job_priority = -(time-job.system_submit)/(math.log2(job.requested_core+1e-6)*job.requested_time)
            heapq.heappush(job_heap, (job_priority, counter, job))
            counter += 1

        _, _, prio_job = heapq.heappop(job_heap)
        can, node_dict = cluster.check_allocate_list(prio_job)
        if can:
            return True, prio_job, node_dict
        else:
            return False, prio_job, job_heap

    def f_1(self, job_queue, cluster, time):
        # log_10(req_time)*req_core+870*log_10(submit)
        job_heap = []
        counter = 0
        # Add jobs to the heap
        for job in job_queue:
            job_priority = math.log10(job.requested_time+1e-6)*job.requested_core+870*math.log10(job.system_submit+1e-6)
            heapq.heappush(job_heap, (job_priority, counter, job))
            counter += 1

        _, _, prio_job = heapq.heappop(job_heap)
        can, node_dict = cluster.check_allocate_list(prio_job)
        if can:
            return True, prio_job, node_dict
        else:
            return False, prio_job, job_heap

    def f_2(self, job_queue, cluster, time):
        job_heap = []
        counter = 0
        # Add jobs to the heap
        for job in job_queue:
            job_priority = math.sqrt(job.requested_time+1e-6)*job.requested_core+25600*math.log10(job.system_submit+1e-6)
            heapq.heappush(job_heap, (job_priority, counter, job))
            counter += 1

        _, _, prio_job = heapq.heappop(job_heap)
        can, node_dict = cluster.check_allocate_list(prio_job)
        if can:
            return True, prio_job, node_dict
        else:
            return False, prio_job, job_heap

    def slurm(self, job_queue, cluster, time):
        #priority=age+job_size+Fairshare+Qos
        pass

    def backfill(self, prio_job, job_heap, time, cluster, allocator):
        check_time = []
        check_time.append(ts.time())
        prio_job_start = cluster.check_earliest_runtime(prio_job, time)
        check_time.append(ts.time())
        backfill_dict = {}
        loop_length = len(job_heap)
        while job_heap:
            _, _, job = heapq.heappop(job_heap)
            can, node_dict = cluster.check_allocate_list(job)
            if can:
                node_list = allocator.allocator(job, node_dict, cluster.topology)
                cluster.allocation(job, node_list, time)
                new_start = cluster.check_earliest_runtime(prio_job, time)
                if prio_job_start == new_start:
                    job.scheduler = 'backfill'
                    backfill_dict[job.id] = job
                else:
                    cluster.release_job(job.id, time, back=True)
        check_time.append(ts.time())
        #print(check_time, 'queue length:', loop_length)
        return backfill_dict

class Allocator:
    def __init__(self, weights={'cpu': 1, 'gpu': 100}, default='best_fit'):
        self.weights = weights
        self.allocator_map = {
            'best_fit': self.best_fit,
            'first_available': self.first_available,
            'topology_aware': self.topology_aware
        }

        self.allocator = self.allocator_map.get(default, self.best_fit)

    def first_available(self, job, node_dict, topology, weight=None):
        node_list = []
        possible_list = list(node_dict.keys())
        for i in range(job.requested_node):
            node_list.append(possible_list[i])
        return node_list

    def best_fit(self, job, node_dict, topology, weight=None):
        #two kinds of codes: cpu nodes and gpu nodes
        node_heap = []
        for name, node in node_dict.items():
            waste = node.free_gpu * self.weights['gpu'] + node.free_core * self.weights['cpu']
            heapq.heappush(node_heap, (waste, name))
        return [heapq.heappop(node_heap)[1] for _ in range(min(job.requested_node, len(node_heap)))]

    def topology_aware(self, job, node_dict, topology, weight=None):
        switches = [node for node, attr in topology.nodes(data=True) if attr['type'] == 'switch']

        # Build the subgraph with both the qualified nodes and all switches
        subgraph = topology.subgraph(list(node_dict.keys()) + switches)

        # Run Dijkstra's algorithm for each qualified node as a starting point
        allocations = []
        for start_node in node_dict.keys():
            # Use Dijkstra's algorithm to find the shortest paths from the starting node to all other nodes
            if weight:
                shortest_paths = nx.single_source_dijkstra_path(subgraph, start_node, weight=weight)
            else:
                # Use unweighted shortest path if no weight argument is given
                shortest_paths = nx.single_source_shortest_path(subgraph, start_node)

            # Initialize the list of allocated nodes with the starting node
            allocated_nodes = [start_node]

            # Add nodes to the list of allocated nodes in order of increasing path length
            for node in sorted(shortest_paths.keys(), key=lambda x: len(shortest_paths[x])):
                if node in node_dict and node not in allocated_nodes:
                    allocated_nodes.append(node)
                if len(allocated_nodes) == job.requested_node:
                    break

            allocations.append(allocated_nodes)

        # Among all feasible allocations, choose the one that leaves the least amount of free resources
        best_allocation = min(allocations, key=lambda x: sum(node_dict[node].free_gpu * self.weights['gpu'] + node_dict[node].free_core * self.weights['cpu'] for node in x))

        if len(best_allocation) < job.requested_node:
            raise ValueError("Not enough contiguous nodes to fulfill request")

        return best_allocation
