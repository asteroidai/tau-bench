# Copyright Sierra

import os
import json
import random
import argparse
import traceback
from math import comb
import multiprocessing
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from tau_bench.envs import get_env
from tau_bench.agents.base import Agent
from tau_bench.types import EnvRunResult
from litellm import provider_list
from tau_bench.envs.user import UserStrategy
from entropy_labs.supervision.config import supervision_config
from entropy_labs.api import (
    register_project,
    create_run,
    register_task,
    register_tools_and_supervisors,
    submit_run_status,
    update_run_result
)
from entropy_labs.sentinel_api_client.sentinel_api_client.models.status import Status

import pkgutil
import importlib
import inspect

from tau_bench.envs.tool import Tool

ENVIRONMENT = "retail" #"airline"


def run(
    args: argparse.Namespace,
    ckpt_path: str,
) -> List[EnvRunResult]:
    print(f"Loading user with strategy: {args.user_strategy}")
    env = get_env(
        args.env,
        user_strategy=args.user_strategy,
        user_model=args.user_model,
        user_provider=args.user_model_provider,
        task_split=args.task_split,
    )
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        args=args,
    )
    end_index = (
        len(env.tasks) if args.end_index == -1 else min(args.end_index, len(env.tasks))
    )
    results: List[EnvRunResult] = []
    lock = multiprocessing.Lock()
    if args.task_ids and len(args.task_ids) > 0:
        print(f"Running tasks {args.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(
            f"Running tasks {args.start_index} to {end_index} (checkpoint path: {ckpt_path})"
    )
    for i in range(args.num_trials):
        if args.task_ids and len(args.task_ids) > 0:
            idxs = args.task_ids
        else:
            idxs = list(range(args.start_index, end_index))
        if args.shuffle:
            random.shuffle(idxs)

        def _run(idx: int) -> EnvRunResult:
            isolated_env = get_env(
                args.env,
                user_strategy=args.user_strategy,
                user_model=args.user_model,
                task_split=args.task_split,
                user_provider=args.user_model_provider,
                task_index=idx,
            )

            print(f"Running task {idx}")
            # Create an execution for the task
            # Register the project and create a run
            project_id = register_project(f"Tau Bench", "http://localhost:8080")
            task_id = register_task(project_id=project_id, task_name=f"Tau Bench {ENVIRONMENT} Task {idx}")
            run_id = create_run(project_id=project_id, task_id=task_id, run_name=f"Tau Bench {ENVIRONMENT} Task {idx}")
            supervision_config.run_id = run_id

            # Register all tools and supervisors
            register_tools_and_supervisors(run_id)

            try:
                res = agent.solve(
                    env=isolated_env,
                    task_index=idx,
                )
                print(f"Result: Reward={res.reward} Info={res.info}")
                result = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=i,
                )
                # send the result to the sentinel API
                status = Status.COMPLETED
                submit_run_status(run_id, status)
                # send the result to the supervisor
                run_outcome = "passed" if res.reward == 1 else "failed"
                update_run_result(run_id, run_outcome, supervision_config.client)
            except Exception as e:
                result = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=i,
                )
            # TODO: End the run
            print(
                "✅" if result.reward == 1 else "❌",
                f"task_id={idx}",
                result.info,
            )
            print("-----")
            with lock:
                data = []
                if os.path.exists(ckpt_path):
                    with open(ckpt_path, "r") as f:
                        data = json.load(f)
                with open(ckpt_path, "w") as f:
                    json.dump(data + [result.model_dump()], f, indent=2)
            return result

        with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            res = list(executor.map(_run, idxs))
            results.extend(res)

    return results


def agent_factory(
    tools_info: List[Dict[str, Any]], wiki, args: argparse.Namespace
) -> Agent:
    if args.agent_strategy == "tool-calling":
        # native tool calling
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent

        return ToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=args.model,
            provider=args.model_provider,
            temperature=args.temperature,
        )
    elif args.agent_strategy == "act":
        # `act` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=args.model,
            provider=args.model_provider,
            use_reasoning=False,
            temperature=args.temperature,
        )
    elif args.agent_strategy == "react":
        # `react` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=args.model,
            provider=args.model_provider,
            use_reasoning=True,
            temperature=args.temperature,
        )
    else:
        raise ValueError(f"Unknown agent strategy: {args.agent_strategy}")


def display_metrics(results: List[EnvRunResult]) -> None:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r.trial for r in results]))
    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")


def find_all_tools():
    """
    Dynamically find all classes inheriting from Tool in the tau_bench.envs package.
    """
    tools = []
    package = importlib.import_module("tau_bench.envs")
    for importer, modname, ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        if not ispkg:
            module = importlib.import_module(modname)
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, Tool)
                    and obj != Tool
                ):
                    tools.append(obj)
    return tools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--env", type=str, choices=["retail", "airline"], default=ENVIRONMENT
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o", #TODO: change this to gpt-4o
        help="The model to use for the agent",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        choices=provider_list,
        default="openai",
        help="The model provider for the agent",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="gpt-4o", #TODO: change this to gpt-4o
        help="The model to use for the user simulator",
    )
    parser.add_argument(
        "--user-model-provider",
        type=str,
        choices=provider_list,
        default="openai", #TODO: change this to openai
        help="The model provider for the user simulator",
    )
    parser.add_argument(
        "--agent-strategy",
        type=str,
        default="tool-calling",
        choices=["tool-calling", "act", "react"],
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The sampling temperature for the action model",
    )
    parser.add_argument( # TODO: update
        "--task-split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
        help="The split of tasks to run (only applies to the retail domain for now",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1, help="Run all tasks if -1")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[54], help="(Optional) run only the tasks with the given IDs") #TODO: remove default , default=[3] [30, 31, 32, 33, 34, 35]
    parser.add_argument("--log-dir", type=str, default=f"results-{ENVIRONMENT}")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--user-strategy", type=str, default="llm", choices=[item.value for item in UserStrategy])

    args = parser.parse_args()
    print(args)
    random.seed(args.seed)

    time_str = datetime.now().strftime("%m%d%H%M%S")
    file_str = f"{args.log_dir}/{args.agent_strategy}-{args.model.split('/')[-1]}-{args.temperature}_range_{args.start_index}-{args.end_index}_user-{args.user_model}-{args.user_strategy}_{time_str}.json"

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    results = run(
        args=args,
        ckpt_path=file_str
    )

    display_metrics(results)

    with open(file_str, "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)
        print(f"\n📄 Results saved to {file_str}\n")


if __name__ == "__main__":
    main()
