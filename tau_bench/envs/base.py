# Copyright Sierra

import random
from hashlib import sha256
from tau_bench.envs.tool import Tool
from typing import Any, Callable, Dict, List, Type, Optional, Set, Union, Tuple

from tau_bench.envs.user import load_user, UserStrategy
from tau_bench.types import (
    Action,
    Task,
    EnvInfo,
    EnvResetResponse,
    EnvResponse,
    RewardResult,
    RewardOutputInfo,
    RewardActionInfo,
    RESPOND_ACTION_NAME,
)
from entropy_labs.supervision.decorators import supervise
from entropy_labs.supervision.supervisors import human_supervisor, llm_supervisor

MULTIPLE_REQUESTS_SYSTEM_PROMPT = """
Your goal is to ensure that all user requests are addressed and resolved. Do not finish the conversation before all requests are resolved. If not sure, ask the user if there's anything else they need help with. 

For each user request, ensure:
- You confirm the resolution of the request explicitly.
- If a request cannot be fulfilled, provide a clear explanation and offer alternatives if possible.
- In case the user asks for multiple things, make sure to address all of them. For example, if the user asks for a product refund and refund amount, make sure to respond with both information.

DECISION CRITERIA:
1. **Approve**:
   - The customer support agent is actively working on or has resolved all requests, including providing explanations for any unfulfilled requests.

2. **Reject**:
   - The agent attempts to conclude the conversation before all requests are resolved or fails to provide necessary explanations for unfulfilled requests.
   - Provide a detailed explanation.
"""
MULTIPLE_REQUESTS_INSTRUCTIONS = """
- Approve if all requests are being addressed or have been resolved, including explanations for any unfulfilled requests.
- Reject if the conversation is prematurely concluded or if explanations for unfulfilled requests are missing, providing a detailed explanation.
- **Do not modify the agent's actions yourself.**
"""

multiple_requests_supervisor = llm_supervisor(
    system_prompt=MULTIPLE_REQUESTS_SYSTEM_PROMPT,
    instructions=MULTIPLE_REQUESTS_INSTRUCTIONS,
    supervisor_name="Multiple Requests Supervisor",
    description="Supervisor that checks all user requests have been solved.",
    include_context=True
)

CORRECT_INFORMATION_PRESENTED_TO_USER_SYSTEM_PROMPT = """
Your goal is to verify that the customer support agent is providing accurate and complete information to the user. You need to ensure that all the user's requests have been addressed and that the information provided is correct.
**You should never modify the agent's actions yourself.**

DECISION CRITERIA:
1. **Approve**:
   - The information presented by the agent is accurate and complete.
   - All user requests have been addressed satisfactorily.

2. **Reject**:
   - The information presented by the agent is inaccurate or incomplete.
   - Provide a detailed explanation of what was incorrect or missing.
"""

CORRECT_INFORMATION_PRESENTED_TO_USER_INSTRUCTIONS = """
DECISION PROCESS:
1. Review the information provided by the customer support agent.
2. Check if all user requests have been addressed.
3. Verify the accuracy and completeness of the information.
4. **Do not modify the agent's actions yourself.**

OUTCOMES:
- **Approve**: If the information is correct and all requests are addressed.
- **Reject**: If any information is incorrect or requests are not fully addressed. Provide specific details on what is wrong.
"""

correct_information_presented_to_user_supervisor = llm_supervisor(
    system_prompt=CORRECT_INFORMATION_PRESENTED_TO_USER_SYSTEM_PROMPT,
    instructions=CORRECT_INFORMATION_PRESENTED_TO_USER_INSTRUCTIONS,
    supervisor_name="Correct Information Presented To User Supervisor",
    description="Supervisor that checks if the agent is presenting correct information to the user.",
    include_context=True
)

# Define a new supervisor for ambiguous item references
AMBIGUOUS_ITEM_SYSTEM_PROMPT = """
Your goal is to ensure that when a user mentions an item that could refer to multiple entries, the assistant requests clarification before proceeding. For example, if the user wants to return a "vacuum cleaner" and there are multiple vacuum cleaners in the order, the assistant should ask for clarification on which specific vacuum cleaner the user is referring to.

For any user request involving an item:
- If multiple items with the same name or description exist in the user's order, prompt the user to specify which one they mean by asking for the item ID or any distinguishing feature.
- Do not assume or proceed with an action involving an ambiguous item without clarification.
- After the user clarifies, continue assisting them with the specific item.

DECISION CRITERIA:
1. **Approve**:
   - The assistant correctly identifies the ambiguity and asks for clarification by referencing item IDs or unique features.
   - The assistant proceeds appropriately after receiving clarification.

2. **Reject**:
   - The assistant does not ask for clarification when multiple items with the same name or description are present.
   - The assistant assumes the wrong item or proceeds without confirming.
   - Provide a detailed explanation of the oversight.

"""

AMBIGUOUS_ITEM_INSTRUCTIONS = """
- **Approve** if the assistant seeks clarification when multiple items with the same name or description are present in the user's order.
- **Reject** if the assistant fails to request clarification and proceeds incorrectly.
- **Do not modify the assistant's actions yourself.**

"""

ambiguous_item_supervisor = llm_supervisor(
    system_prompt=AMBIGUOUS_ITEM_SYSTEM_PROMPT,
    instructions=AMBIGUOUS_ITEM_INSTRUCTIONS,
    supervisor_name="Ambiguous Item Supervisor",
    description="Supervisor that ensures assistants ask for clarification when multiple items match the user's description.",
    include_context=True
)


ToHashable = Union[
    str, int, float, Dict[str, "ToHashable"], List["ToHashable"], Set["ToHashable"]
]
Hashable = Union[str, int, float, Tuple["Hashable"], Tuple[Tuple[str, "Hashable"]]]


def to_hashable(item: ToHashable) -> Hashable:
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    elif isinstance(item, list):
        return tuple(to_hashable(element) for element in item)
    elif isinstance(item, set):
        return tuple(sorted(to_hashable(element) for element in item))
    else:
        return item

@supervise(
        supervision_functions=[
            # [human_supervisor()]
            [correct_information_presented_to_user_supervisor],
            [multiple_requests_supervisor],
            [ambiguous_item_supervisor]
             #, human_supervisor()]
        ],
        ignored_attributes=["self"]
    )
def respond_to_user(self, content):
    """
    Responds to the user.
    """
    return self.user.step(content)

def consistent_hash(
    value: Hashable,
) -> str:
    return sha256(str(value).encode("utf-8")).hexdigest()


class Env(object):
    def __init__(
        self,
        data_load_func: Callable[[], Dict[str, Any]],
        tools: List[Type[Tool]],
        tasks: List[Task],
        wiki: str,
        rules: List[str],
        user_strategy: Union[str, UserStrategy],
        user_model: str,
        user_provider: Optional[str] = None,
        task_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_load_func = data_load_func
        self.data = data_load_func()
        self.tools_map: Dict[str, Type[Tool]] = {
            tool.get_info()["function"]["name"]: tool for tool in tools
        }
        self.tools_info = [tool.get_info() for tool in tools]
        self.terminate_tools = []
        self.tasks = tasks
        if task_index is not None:
            self.task_index = task_index
        else:
            self.task_index = random.randint(0, len(tasks))
        self.task = tasks[self.task_index]
        self.wiki = wiki
        self.rules = rules
        self.user = load_user(
            user_strategy=user_strategy, model=user_model, provider=user_provider
        )
        self.actions: List[Action] = []

    def reset(self, task_index: Optional[int] = None) -> EnvResetResponse:
        if task_index is None:
            task_index = random.randint(0, len(self.tasks))
        self.task_index = task_index
        self.data = self.data_load_func()
        self.task = self.tasks[task_index]
        self.actions = []
        initial_observation = self.user.reset(instruction=self.task.instruction)
        return EnvResetResponse(
            observation=initial_observation, info=EnvInfo(task=self.task, source="user")
        )
        
    


    def step(self, action: Action) -> EnvResponse:
        self.actions.append(action)

        info = EnvInfo(task=self.task)
        reward = 0
        done = False

        if action.name == RESPOND_ACTION_NAME:
            observation = respond_to_user(self=self, content=action.kwargs["content"])
            print(f"Responding to user: {observation}")
            info.source = "user"
            done = "###STOP###" in observation
        elif action.name in self.tools_map:
            print(f"Invoking tool: {action.name}")
            try:
                observation = self.tools_map[action.name].invoke(
                    data=self.data, **action.kwargs
                )
                print(f"Observation: {observation}")
            except Exception as e:
                observation = f"Error: {e}"
                print(f"Error: {e}")
            info.source = action.name
            if action.name in self.terminate_tools:
                done = True
        else:
            observation = f"Unknown action {action.name}"
            print(f"Unknown action {action.name}")
            info.source = action.name

        if done:
            reward_res = self.calculate_reward()
            reward = reward_res.reward
            info.reward_info = reward_res
            info.user_cost = self.user.get_total_cost()

        return EnvResponse(observation=observation, reward=reward, done=done, info=info)

    def get_data_hash(self) -> str:
        return consistent_hash(to_hashable(self.data))

    def calculate_reward(self) -> RewardResult:
        data_hash = self.get_data_hash()
        reward = 1.0
        actions = [
            action for action in self.task.actions if action.name != RESPOND_ACTION_NAME
        ]

        if len(self.task.outputs) > 0:
            # check outputs
            r_outputs = 1.0
            outputs = {}
            for output in self.task.outputs:
                found = False
                for action in self.actions:
                    if (
                        action.name == RESPOND_ACTION_NAME
                        and output.lower()
                        in action.kwargs["content"].lower().replace(",", "")
                    ):
                        found = True
                        break
                outputs[output] = found
                if not found:
                    r_outputs = 0.0
                    reward = 0.0
            info = RewardOutputInfo(r_outputs=r_outputs, outputs=outputs)
        else:
            # check database change
            # TODO: cache gt_data_hash in tasks.py (low priority)
            self.data = self.data_load_func()
            for action in self.task.actions:
                if action.name not in self.terminate_tools:
                    self.step(action)
            gt_data_hash = self.get_data_hash()
            info = RewardActionInfo(
                r_actions=data_hash == gt_data_hash, gt_data_hash=gt_data_hash
            )
            if not info.r_actions:
                reward = 0.0
        return RewardResult(reward=reward, info=info, actions=actions)
