import re
from types import SimpleNamespace

from balrog.client import LLMResponse

ACT_INSTRUCTION = """
Look at your previous plan and observations, then choose exactly ONE action from the allowed actions listed previously.
Output no other text.
""".strip()


PLAN_INSTRUCTION = """
Review your previous observations and plan, then make a high-level plan for completing the task. Your plan can include reasoning about how to solve the task.
After this planning phase, you will be asked to take actions one at a time.

Output your plan strictly in the following format:
<plan>YOUR_PLAN</plan>
Replace YOUR_PLAN with your own thinking and plan.

After your plan, choose exactly ONE action from the allowed actions listed previously.
Output no other text.
""".strip()

MAYBE_PLAN_INSTRUCTION = """
Review your current plan and observations.  
• If you do not have a plan yet, create one.  
• If your plan is outdated or needs changes, create a new plan.

If you create a new plan, output it in the following format:

<plan>YOUR_NEW_PLAN</plan>

Replace YOUR_NEW_PLAN with your revised plan.

If your current plan is still valid, proceed without outputting it again.

After this evaluation (and any necessary replanning), output exactly ONE allowed action.

Output nothing else except an optional <plan>…</plan> block and that single action.
""".strip()

class BaseAgent:
    """Base class for agents using prompt-based interactions."""

    def __init__(self, client_factory, prompt_builder):
        """Initialize the agent with a client and prompt builder."""
        self.client = client_factory()
        self.prompt_builder = prompt_builder

    def act(self, obs):
        """Generate an action based on the observation."""
        raise NotImplementedError

    def update_prompt(self, observation, action):
        """Update the prompt with the observation and action."""
        self.prompt_builder.update_observation(observation)
        self.prompt_builder.update_action(action)

    def reset(self):
        """Reset the prompt builder."""
        self.prompt_builder.reset()


class BasePlanningAgent(BaseAgent):
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)

    def reset(self):
        super().reset()

    def _generate_output(self, messages, instruction):
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"].append({"type": "text", "text": instruction})
        elif messages and messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": [{"type": "text", "text": instruction}]})

        formatted_conversation = []
        for msg in messages:
            for content in msg["content"]:
                formatted_conversation.append(
                    SimpleNamespace(
                        role=msg["role"],
                        content=content["text"],
                        attachment=None
                    )
                )
        llm_response: LLMResponse = self.client.generate(formatted_conversation)
        return llm_response

    def _extract_plan(self, completion_text):
        match = re.search(r"<plan>(.*?)</plan>", completion_text, re.DOTALL)
        
        if match:
            plan = match.group(1).strip()
            action = completion_text.split("</plan>")[-1].strip()
            return plan, action, True
        else:
            return "", completion_text.strip(), False


class PlanEveryKStep(BasePlanningAgent):
    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder)
        self.plan_every_k = config.agent.plan_k
        self.time_to_plan = 0

    def reset(self):
        super().reset()
        self.time_to_plan = 0

    def act(self, obs, prev_action=None):
        try:
            self.prompt_builder.update_action(prev_action)
            self.prompt_builder.update_observation(obs)
        except Exception as e:
            print(f"Error updating action and observation in prompt builder: {e}")

        messages = self.prompt_builder.get_prompt()

        if self.time_to_plan == 0:
            self.time_to_plan = self.plan_every_k
                       
            plan_instruction = PLAN_INSTRUCTION

            llm_response = self._generate_output(messages, plan_instruction)
            plan, action, _ = self._extract_plan(llm_response.completion)
            
            self.prompt_builder.update_plan(plan)
        else:
            llm_response = self._generate_output(messages, ACT_INSTRUCTION)
            action = llm_response.completion
            plan = ""
        
        llm_response = llm_response._replace(completion=action)
        llm_response = llm_response._replace(reasoning=plan)

        self.time_to_plan -= 1
        return llm_response

    

ALWAYS_PLAN_INSTRUCTION = """
Review your previous observations and plan, then make a high-level plan for completing the task. Your plan can include reasoning about how to solve the task.
After this planning phase, you will be asked to take actions one at a time.

Output your plan strictly in the following format:
<plan>YOUR_PLAN</plan>
Replace YOUR_PLAN with your own thinking and plan.

After your plan, choose exactly ONE action from the allowed actions listed previously.
Output no other text.
""".strip()

class AlwaysPlan(BasePlanningAgent):
    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder)

    def act(self, obs, prev_action=None):
        try:
            self.prompt_builder.update_action(prev_action)
            self.prompt_builder.update_observation(obs)
        except Exception as e:
            print(f"Error updating action and observation in prompt builder: {e}")

        messages = self.prompt_builder.get_prompt()
        output, action_token_ids, prompt_token_ids, logprob = self._generate_output(messages, ALWAYS_PLAN_INSTRUCTION)
        plan, action, _ = self._extract_plan(output)
        self.prompt_builder.update_plan(plan)

        action = action.strip()
        return action, action_token_ids, prompt_token_ids, logprob
    
    
    
NEW_ACT_INSTRUCTION = """
Look at your previous observations, then choose exactly ONE action from the allowed actions listed previously.
Output no other text.
""".strip()

class NeverPlan(BasePlanningAgent):
    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder)

    def reset(self):
        super().reset()

    def act(self, obs, prev_action=None):
        try:
            self.prompt_builder.update_action(prev_action)
            self.prompt_builder.update_observation(obs)
        except Exception as e:
            print(f"Error updating action and observation in prompt builder: {e}")

        messages = self.prompt_builder.get_prompt()
        instruction = NEW_ACT_INSTRUCTION
        output, action_token_ids, prompt_token_ids, logprob = self._generate_output(messages, instruction)

        action = output

        action = action.strip()

        return action, action_token_ids, prompt_token_ids, logprob