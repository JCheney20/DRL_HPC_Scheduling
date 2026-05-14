import numpy as np


class Evaluator:
    def __init__(self, scheduler, allocator):
        self.completed_job = []
        self.scheduler_pair = (scheduler, allocator)
        self.job_dict = {}

    def check_in_job(self, job):
        self.completed_job.append(job)

    def waiting_time(self):
        waiting = []
        for i in self.completed_job:
            waiting.append(i.start_time-i.system_submit)
        if waiting:
            max_waiting = max(waiting)
            average_waiting = np.mean(waiting)
            return max_waiting, average_waiting

    def bounded_slowdown(self):
        slowdown = []
        for i in self.completed_job:
            #runtime = max(i.end_time - i.start_time, 10)
            runtime = max(i.total_run, 10)
            slowdown.append(max(((i.end_time - i.system_submit) / runtime), 1))
        if slowdown:
            max_slowdown = max(slowdown)
            average_slowdown = np.mean(slowdown)
            return max_slowdown, average_slowdown

    def average_turnaround(self):
        turnaround = []
        for i in self.completed_job:
            turnaround.append(i.end_time-i.system_submit)
        if turnaround:
            average_turnaround = np.mean(turnaround)
            return average_turnaround

    def average_utilization(self, resource, time):
        core_utilization = []
        gpu_utilization = []
        for i in self.completed_job:
            coretime = (i.end_time - i.start_time) * i.requested_core
            gputime = (i.end_time - i.start_time) * i.requested_gpu
            core_utilization.append(coretime)
            gpu_utilization.append(gputime)
        utilised_core = sum(core_utilization)
        utilization_core = utilised_core/(resource['core'] * time + 1e-6)
        utilised_gpu = sum(gpu_utilization)
        utilization_gpu = utilised_gpu / (resource['gpu'] * time + 1e-6)
        return utilization_core, utilization_gpu

    def reset(self, scheduler, allocator, resource, time):
        self.job_dict[self.scheduler_pair] = [self.waiting_time(), self.average_turnaround(), self.average_utilization(resource, time)]
        self.completed_job = []
        self.scheduler_pair = (scheduler, allocator)