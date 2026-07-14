import pandas as pd
import heapq
import random
from datetime import datetime
from dateutil.parser import parse
import re


class Job:
    def __init__(self, data, random_job=False):
        if not random_job:
            self.id = data['JobID']
            self.uid = data['UID']
            self.gid = data['GID']
            self.account = data['Account']
            self.allocated_cores = data['AllocCPUS']
            self.allocated_nodes = data['AllocNodes']
            self.allocated_gpu = data['Allgpu']
            self.allocated_memory = self.convert_to_mb(data['Allmem'])
            self.requested_core = data['ReqCPUS']
            self.requested_node = data['ReqNodes']
            self.requested_gpu = data['Reqgpu']
            self.requested_memory = self.convert_to_mb(data['ReqMem'])
            self.requested_time = int(data['TimelimitRaw']*60)
            self.admin_comment = data['AdminComment']
            self.allocate_structure = self.convert_admin_comment_to_dict(self.admin_comment)
            cpu_values = [node['cpu'] for node in self.allocate_structure.values()]
            if len(set(cpu_values)) == 1:
                self.balanced = True
            else:
                self.balanced = False
            self.constraints = data['Constraints']
            self.cpu_time_raw = data['CPUTimeRAW']
            self.total_run = min(data['ElapsedRaw'], self.requested_time)
            #times
            self.eligible = self.convert_datetime(data['Eligible'])
            self.submit = self.convert_datetime(data['Submit'])
            self.start = self.convert_datetime(data['Start'])
            self.end = self.convert_datetime(data['End'])
            self.state = data['State']
            self.node_list = data['NodeList']
            self.partition = data['Partition']
            self.reserved = data['Reserved']
            self.qos = data['QOS']
            self.qos_raw = data['QOSRAW']
            self.reason = data['Reason']
            self.difference = data['difference']
            # The reason that cause requested resource is not equal to the allocated resource is that the requested resource
            # is incomplete. In other words, the user only specifies a part of the resource, not the complete resource they
            # need, then the system sets unspecified resources with default values.
            if self.requested_core != self.allocated_cores:
                self.requested_core = self.allocated_cores
            if self.requested_node != self.allocated_nodes:
                self.requested_node = self.allocated_nodes
            if self.requested_gpu != self.allocated_gpu:
                self.requested_gpu = self.allocated_gpu
            if self.requested_memory != self.allocated_memory:
                self.requested_memory = self.allocated_memory
        else:
            self.id = data['JobID']
            self.uid = 0
            self.gid = 0
            self.balanced = True
            self.allocated_cores = data['AllocCPUS']
            self.allocated_nodes = data['AllocNodes']
            self.allocated_gpu = data['Allgpu']
            self.allocated_memory = self.convert_to_mb(data['Allmem'])
            self.requested_core = data['ReqCPUS']
            self.requested_node = data['ReqNodes']
            self.requested_gpu = data['Reqgpu']
            self.requested_memory = self.convert_to_mb(data['ReqMem'])
            self.requested_time = int(data['TimelimitRaw'])
            self.submit = data['Submit']
            self.total_run = min(data['ElapsedRaw'], self.requested_time)

        # Cluster.check_allocate_list derives per-task resources by dividing by
        # requested_node on essentially every observation build; a row with
        # requested_node < 1 would raise ZeroDivisionError deep in a run.
        # Fail loud at load time (t=0) instead, naming the offending job — do
        # not silently mutate the trace.
        if self.requested_node < 1:
            raise ValueError(
                f"Job {self.id}: requested_node must be >= 1, got "
                f"{self.requested_node!r}"
            )

    def convert_datetime(self, time_str):
        try:
            time_dt = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
        except ValueError:
            time_dt = parse(time_str)

        return time_dt

    def convert_to_mb(self, memory_str):
        memory_str = memory_str.upper()
        units = {
            'K': 1 / 1024,
            'M': 1,
            'G': 1024,
            'T': 1024 * 1024
        }

        unit = memory_str[-1]
        if unit.isdigit():
            return int(memory_str)  # If no unit is specified, assume it's in MB by default

        value = float(memory_str[:-1])
        return int(value * units[unit])

    def convert_admin_comment_to_dict(self, admin_comment):
        nodes = admin_comment.split(";")[:-1]  # split by ";", the last element is empty due to trailing ";" so discard
        node_dict = {}

        for node in nodes:
            # split by "|" and discard first and last empty elements
            node_info = node.split("|")[1:-1]

            node_name = node_info[0]

            # get cpu counts
            cpu_info = node_info[1].split(",")
            cpu_count = 0
            for info in cpu_info:
                if '-' in info:
                    start, end = map(int, info.split("-"))
                    cpu_count += end - start + 1
                else:
                    cpu_count += 1

            # initialize gpu_count with a default value
            gpu_count = 0

            # get gpu count if available
            if len(node_info) > 2:  # gpu info present
                gpu_info = node_info[2]
                gpu_match = re.search(r'gpu:[^:]+:(\d+(-\d+)?|\(IDX:\d+(-\d+)?\))', gpu_info)
                if gpu_match:
                    gpu_count = gpu_match.group(1)  # extract number or range after the second ":"
                    if '-' in gpu_count:  # if range, adjust gpu_count accordingly
                        start, end = map(int, gpu_count.split("-"))
                        gpu_count = end - start + 1
                    else:
                        gpu_count = int(gpu_count)  # convert to integer
            node_dict[node_name] = {'cpu': cpu_count, 'gpu': gpu_count}
        return node_dict


class Trace:
    def __init__(self, trace_file, partition=None, start=None, random_job=False, num_random_jobs=1000):
        self.random_job = random_job
        self.num_random_jobs = num_random_jobs
        if not random_job:
            if start:
                self.start = datetime.strptime(start, '%Y-%m-%d')
            else:
                self.start = None
            self.job_list = self.read_jobs_from_csv(trace_file)
            partition_list = []
            if partition:
                for i in self.job_list:
                    if i.partition == partition:
                        partition_list.append(i)
                self.job_list = partition_list
            self.job_heap = []
            if start:
                start_list = []
                counter = 0
                for job in self.job_list:
                    if job.submit >= self.start:
                        start_list.append(job)
                        heapq.heappush(self.job_heap, (job.submit, counter, job))
                        counter += 1
                self.job_list = start_list
            else:
                counter = 0
                for job in self.job_list:
                    heapq.heappush(self.job_heap, (job.submit, counter, job))
                    counter += 1
        else:
            self.start = None
            self.job_list, self.job_heap, self.longest_requested_time = self.generate_jobs()
        self.first_arrival = self.job_heap[0][0]

    def generate_jobs(self):
        job_list = []
        job_heap = []
        id = 0
        submit = 0
        counter = 0
        longest_requested_time = 0
        for _ in range(self.num_random_jobs):
            data = {}
            data['JobID'] = id
            data['AllocNodes'] = int(random.random() * 10) + 1
            if data['AllocNodes'] <= 5:
                data['AllocCPUS'] = data['AllocNodes'] * (int(random.random() * 32) + 1)
            else:
                data['AllocCPUS'] = data['AllocNodes'] * (int(random.random() * 16) + 1)
            data['Allgpu'] = data['AllocNodes'] * (int(random.random() * 4) + 1)
            data['Allmem'] = str(data['AllocNodes'] * (int(random.random() * 100) + 1)) + 'G'
            data['ReqCPUS'] = data['AllocCPUS']
            data['ReqNodes'] = data['AllocNodes']
            data['Reqgpu'] = data['Allgpu']
            data['ReqMem'] = data['Allmem']
            data['TimelimitRaw'] = int(random.random() * 240) + 1
            data['Submit'] = submit
            data['ElapsedRaw'] = int(data['TimelimitRaw'] * random.random())
            job = Job(data, random_job=True)
            if job.requested_time > longest_requested_time:
                longest_requested_time = job.requested_time
            job_list.append(job)
            heapq.heappush(job_heap, (job.submit, counter, job))
            id += 1
            counter += 1
            submit += int(random.random() * 60)
        return job_list, job_heap, longest_requested_time

    def read_jobs_from_csv(self, file_name):
        df = pd.read_csv(file_name, delimiter='\t', low_memory=False)

        jobs = []
        self.longest_requested_time = 0
        for index, row in df.iterrows():
            job = Job(row)
            jobs.append(job)
            if job.requested_time > self.longest_requested_time:
                self.longest_requested_time = job.requested_time

        return jobs

    def get_next_jobs(self):
        # Pop the earliest job
        earliest_job_time, _, earliest_job = heapq.heappop(self.job_heap)

        # Prepare a list to hold all jobs with the same submit time
        same_time_jobs = [earliest_job]

        # If there are more jobs with the same submit time, pop them as well
        while self.job_heap and self.job_heap[0][0] == earliest_job_time:
            _, _, same_time_job = heapq.heappop(self.job_heap)
            same_time_jobs.append(same_time_job)

        return same_time_jobs

    def get_next_arrival_time(self):
        return heapq.nsmallest(1, self.job_heap)[0][0]

    def reset(self):
        if not self.random_job:
            self.job_heap = []
            if self.start:
                counter = 0
                for job in self.job_list:
                    if job.submit >= self.start:
                        job.system_submit = -1
                        job.start_time = -1
                        job.end_time = -1
                        heapq.heappush(self.job_heap, (job.submit, counter, job))
                        counter += 1
            else:
                counter = 0
                for job in self.job_list:
                    job.system_submit = -1
                    job.start_time = -1
                    job.end_time = -1
                    heapq.heappush(self.job_heap, (job.submit, counter, job))
                    counter += 1
        else:
            self.start = None
            self.job_list, self.job_heap, self.longest_requested_time = self.generate_jobs()
        self.first_arrival = self.job_heap[0][0]
