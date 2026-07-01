from src.HPCsim.HPCsim import *
import json
import os
from src.utils import ArgumentParserWithDefaults


def parse_args():
    parser = ArgumentParserWithDefaults(description="Scheduler testing.")
    parser.add_argument(
        "--selector",
        default="fcfs",
        dest="selector",
        metavar="SELECTOR",
        help="The selector to use.",
        type=str,
    )
    parser.add_argument(
        "--allocator",
        default="best_fit",
        dest="allocator",
        metavar="ALLOCATOR",
        help="The allocator to use.",
        type=str,
    )
    parser.add_argument(
        "--backfill",
        default=True,
        dest="backfill",
        metavar="BACKFILL",
        help="Backfill enable.",
        type=bool,
    )
    parser.add_argument(
        "--topology",
        default="physical_topology.txt",
        dest="topology_file",
        metavar="TOPOLOGY_FILE",
        help="The topology file of the environment.",
        type=str,
    )
    parser.add_argument(
        "--trace",
        default="physical_job.csv",
        dest="trace_file",
        metavar="TRACE_FILE",
        help="The trace file of the environment.",
        type=str,
    )
    parser.add_argument(
        "--node",
        default="nodes.csv",
        dest="node_file",
        metavar="NODE_FILE",
        help="The node file of the environment.",
        type=str,
    )
    args = parser.parse_args()
    print(args)
    return (
        args.selector,
        args.allocator,
        args.backfill,
        args.topology_file,
        args.trace_file,
        args.node_file,
    )


if __name__ == "__main__":
    selector, allocator, backfill, topology_file, trace_file, node_file = parse_args()
    env = HPCsim(
        scheduler=selector,
        allocator=allocator,
        backfill_enable=backfill,
        topology_file=f"data/topology/{topology_file}",
        node_file=f"data/topology/{node_file}",
        trace_file=f"data/{trace_file}",
        random_job=False,
    )
    env.run()

    max_w, avg_w = env.evaluator.waiting_time()
    max_s, avg_s = env.evaluator.bounded_slowdown()
    avg_t = env.evaluator.average_turnaround()

    print(
        f"Maximum waiting:    {float(max_w):>12,.0f}s  |  Average waiting:    {float(avg_w):>10.2f}s"
    )
    print(
        f"Maximum slowdown:   {float(max_s):>12.4f}   |  Average slowdown:   {float(avg_s):>10.4f}"
    )
    print(f"Average turnaround: {float(avg_t):>12.2f}s")

    metrics = {
        "selector": selector,
        "allocator": allocator,
        "max_waiting": float(max_w),
        "avg_waiting": float(avg_w),
        "max_slowdown": float(max_s),
        "avg_slowdown": float(avg_s),
        "avg_turnaround": float(avg_t),
    }
    os.makedirs("result", exist_ok=True)
    metrics_path = f"result/{selector}+{allocator}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
