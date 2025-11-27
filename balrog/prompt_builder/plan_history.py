import base64
from io import BytesIO
from collections import deque
from typing import List, Optional


def _process_image_openai(image):
    """Process an image for OpenAI API by converting it to base64."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
    }


class PlanHistoryPromptBuilder:
    """
    Builds a prompt with a history of observations and actions,
    distinguishing between a persistent master plan and ephemeral local plans.
    """

    def __init__(
        self,
        max_text_history: int = 16,
        max_image_history: int = 1,
        system_prompt: Optional[str] = None,
        max_cot_history: int = 0,
        remove_old_local_plans: bool = True,
    ):
        self.max_history = max_text_history
        self.max_image_history = min(max_image_history, max_text_history)
        self.remove_old_local_plans = remove_old_local_plans

        self._events = deque(maxlen=max_text_history)
        self.master_plan = None  # Store the persistent master plan
        self.current_local_plan = None # Store the latest local plan

        self.system_prompt = system_prompt

    def update_instruction_prompt(self, instruction: str):
        """Set the system-level instruction prompt."""
        self.system_prompt = instruction

    def update_plan(self, plan: str, plan_type: str = "local"):
        """Add plan as a new event. Master plans persist, local plans replace previous local plans."""
        if not plan:
            if plan_type == 'local':
                self.current_local_plan = None
            return

        plan_text = plan.strip()

        if plan_type == 'master':
            self.master_plan = plan_text
        elif plan_type == 'local':
            self.current_local_plan = plan_text

            # Conditionally remove previous local plan events from the history
            if self.remove_old_local_plans:
                new_events = deque(maxlen=self.max_history)
                for round_events in self._events:
                    # Remove all existing plan events from every round in self._events
                    filtered_round = [event for event in round_events if event.get("type") != "local_plan"]
                    if filtered_round: # Only add round if it's not empty after filtering
                        new_events.append(filtered_round)
                self._events = new_events
            # else: Keep old local plans in self._events

            # Add the new local plan event to the latest round
            if not self._events:
                self._events.append([])
            self._events[-1].append({
                "type": "local_plan",
                "text": plan_text,
            })
        else:
            raise ValueError(f"Unknown plan_type: {plan_type}")

    def update_observation(self, obs: dict):
        """
        Add an observation as a new round. Each time we get an observation,
        we create a new round (a new list in the deque).
        """
        self._events.append([])

        long_term_context = obs["text"].get("long_term_context", "")
        short_term_context = obs["text"].get("short_term_context", "")
        image = obs.get("image", None)

        self._events[-1].append({
            "type": "observation",
            "text": long_term_context,
            "short_term": short_term_context,
            "image": image,
        })

    def update_action(self, action: str):
        """
        Add an action to the current round, attach any previous plan if it exists.
        """
        if not action:
            return
        if not self._events:
            self._events.append([])

        self._events[-1].append({
            "type": "action",
            "action": action,
        })

    def pop_obs(self):
        if self._events:
            self._events.pop()

    def pop_action(self):
        if self._events:
            self._events[-1].pop()


    def reset(self):
        """Clear the event history and current plan."""
        self._events.clear()
        self.master_plan = None
        self.current_local_plan = None

    def get_prompt(self) -> List[dict]:
        """
        Generate a list of chat messages (OpenAI-style).
        """
        messages = []

        # 1) System prompt
        if self.system_prompt:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.system_prompt
                    }
                ]
            })

        # 1.5) Add Master Plan if it exists
        if self.master_plan:
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"<master_plan>{self.master_plan}</master_plan>"}
                ]
            })

        images_included = 0

        # Helper function to append text (and optional image) to the last user message
        def append_to_user_message(text: str, image=None):
            nonlocal messages, images_included

            # If there's a user message at the end, append there
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"].append({
                    "type": "text",
                    "text": text
                })
                if image is not None and images_included < self.max_image_history:
                    messages[-1]["content"].append(_process_image_openai(image))
                    images_included += 1
            else:
                new_content = [{"type": "text", "text": text}]
                if image is not None and images_included < self.max_image_history:
                    new_content.append(_process_image_openai(image))
                    images_included += 1

                messages.append({
                    "role": "user",
                    "content": new_content
                })

        # 2) Unroll each sub-event in each round:
        for idx, round_events in enumerate(self._events):
            for event in round_events:
                etype = event["type"]

                if etype == "observation":
                    # Construct text for observation
                    short_term_txt = event.get("short_term", "")
                    text_str = ""
                    if short_term_txt and idx == len(self._events) - 1:
                        text_str += f"Current Observation:\n{short_term_txt}\n" + event["text"] + "\n"
                    else:
                        text_str += "Observation:\n" + event["text"] + "\n"

                    img = event.get("image", None)
                    append_to_user_message(text_str, img)

                elif etype == "action":
                    # If the previous message was already assistant, append this action to the same message
                    if messages and messages[-1]["role"] == "assistant":
                        messages[-1]["content"][-1]["text"] += f"\n{event['action']}"
                    else:
                        messages.append({
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": event["action"]}
                            ]
                        })

                elif etype == "local_plan":
                     messages.append({
                         "role": "assistant",
                         "content": [
                            {"type": "text", "text": f"<local_plan>{event['text']}</local_plan>"}
                        ]
                    })

        return messages