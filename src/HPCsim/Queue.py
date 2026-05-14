class Queue:
    def __init__(self):
        self.job_queue = []

    def receive_job(self, job_list, time):
        self.job_queue.extend(job_list)
        for i in job_list:
            i.system_submit = time

    def pop_sched_job(self, job):
        self.job_queue.remove(job)

    def get_state(self, window, cluster, tail=0, time=0):
        queue_state = []
        action_mask = []
        possible_jobs = 0
        if len(self.job_queue) >= window:
            head_queue = self.job_queue[:(window-tail)]
            if tail > 0:
                tail_queue = self.job_queue[-tail:]
            else:
                tail_queue = []
            state_jobs = head_queue + tail_queue
            for job in state_jobs:
                can, _ = cluster.check_allocate_list(job)
                if can:
                    possible_jobs += 1
                    action_mask.append(True)
                    queue_state += [job.requested_node, job.task_core, job.task_memory, job.task_gpu, job.requested_time, time-job.system_submit, 1]
                else:
                    action_mask.append(False)
                    queue_state += [job.requested_node, job.task_core, job.task_memory, job.task_gpu, job.requested_time, time-job.system_submit, 0]
            queue_state += [len(self.job_queue)]
        else:
            dummy_jobs = window - len(self.job_queue)
            for job in self.job_queue:
                can, _ = cluster.check_allocate_list(job)
                if can:
                    possible_jobs += 1
                    action_mask.append(True)
                    queue_state += [job.requested_node, job.task_core, job.task_memory, job.task_gpu, job.requested_time, time-job.system_submit, 1]
                else:
                    action_mask.append(False)
                    queue_state += [job.requested_node, job.task_core, job.task_memory, job.task_gpu, job.requested_time, time-job.system_submit, 0]
            for _ in range(dummy_jobs):
                action_mask.append(False)
                queue_state += [0, 0, 0, 0, 0, 0, 0]
            queue_state += [len(self.job_queue)]
        action_mask.append(True)
        return queue_state, possible_jobs, action_mask

    def reset(self):
        self.job_queue = []
