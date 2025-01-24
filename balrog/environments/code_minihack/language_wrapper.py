import gymnasium as gym
from nle import nle_language_obsv
from nle.nethack import USEFUL_ACTIONS
from PIL import Image

from balrog.environments import Strings
from balrog.environments.nle.progress import get_progress_system
from balrog.environments.nle.render import tty_render_image
from balrog.environments.nle.render_rgb import rgb_render_image


class LanguageWrapper(gym.Wrapper):
    def __init__(self, env, vlm=False):
        super().__init__(env)
        self.nle_language = nle_language_obsv.NLELanguageObsv()
        self.language_action_space = self.create_action_space()
        self.vlm = vlm

        if not vlm:
            self.prompt_mode = "hybrid"
        else:
            self.prompt_mode = "language"

        self.progress = get_progress_system(self.env.unwrapped)
        self.max_steps = self.env.bot.max_strategy_steps

    def create_action_space(self):
        all_actions = [strat.__name__ for strat in self.env.bot.strategies]

        return Strings(all_actions)

    @property
    def default_action(self):
        return "noop"

    def reset(self, **kwargs):
        self.progress = get_progress_system(self.env.unwrapped)
        obsv, info = self.env.reset(**kwargs)

        return self.nle_process_obsv(obsv), info

    def step(self, action):
        action_idx = self.language_action_space._dict[action]
        obsv, reward, terminated, truncated, info = self.env.step(action_idx)
        done = terminated or truncated
        self.progress.update(obsv, reward, done, info)

        return self.nle_process_obsv(obsv), reward, terminated, truncated, info

    def nle_process_obsv(self, nle_obsv):
        img = Image.fromarray(self.render("tiles")).convert("RGB") if self.vlm else None
        text = self.nle_obsv_type(nle_obsv)

        return {
            "text": text,
            "image": img,
            "obs": nle_obsv,
        }

    def nle_obsv_type(self, nle_obsv):
        nle_obsv = self.nle_obsv_to_language(nle_obsv)
        if self.prompt_mode == "language":
            return self.render_text(nle_obsv)
        elif self.prompt_mode == "hybrid":
            return self.render_hybrid(nle_obsv)
        else:
            raise ValueError(f'"{self.prompt_mode}" is not a valid prompt mode.')

    def render(self, mode="human"):
        if mode == "tiles":
            obs = self.env.last_observation
            glyphs = obs[self.env._observation_keys.index("glyphs")]
            return rgb_render_image(glyphs)
        elif mode == "tty_image":
            obs = self.env.last_observation
            tty_chars = obs[self.env._observation_keys.index("tty_chars")]
            tty_colors = obs[self.env._observation_keys.index("tty_colors")]
            return tty_render_image(tty_chars, tty_colors)
        else:
            return super().render(mode)

    def get_stats(self):
        return self.progress.__dict__

    def ascii_render(self, chars):
        rows, cols = chars.shape
        result = ""
        for i in range(rows):
            for j in range(cols):
                entry = chr(chars[i, j])
                result += entry
            result += "\n"
        return result

    def nle_obsv_to_language(self, nle_obsv):
        """Translate NLE Observation into a language observation.
        Args:
            nle_obsv (dict): NLE observation from the base environment
        Returns:
            (dict): language observation
        """
        glyphs = nle_obsv["glyphs"]
        blstats = nle_obsv["blstats"]
        tty_cursor = nle_obsv["tty_cursor"]
        inv_strs = nle_obsv["inv_strs"]
        inv_letters = nle_obsv["inv_letters"]
        text_message = (
            (
                nle_obsv["text_message"]
                if "text_message" in nle_obsv
                else self.nle_language.text_message(nle_obsv["tty_chars"]).decode("latin-1")
            ),
        )

        return {
            "text_glyphs": self.nle_language.text_glyphs(glyphs, blstats).decode("latin-1"),
            "text_message": text_message,
            "text_blstats": self.nle_language.text_blstats(blstats).decode("latin-1"),
            "text_inventory": self.nle_language.text_inventory(inv_strs, inv_letters).decode("latin-1"),
            "text_cursor": self.nle_language.text_cursor(glyphs, blstats, tty_cursor).decode("latin-1"),
            "tty_chars": nle_obsv["tty_chars"],
            "tty_cursor": nle_obsv["tty_cursor"],
        }

    def render_text(self, nle_obsv):
        long_term_observations = [
            ("text_message", "message"),
            ("text_glyphs", "language observation"),
            ("text_cursor", "cursor"),
        ]

        short_term_observations = [
            ("text_blstats", "statistics"),
            ("text_inventory", "inventory"),
        ]

        long_term_context = "\n".join([f"{name}:\n{nle_obsv[key]}\n" for key, name in long_term_observations])
        short_term_context = "\n".join([f"{name}:\n{nle_obsv[key]}\n" for key, name in short_term_observations])

        return {
            "long_term_context": long_term_context,
            "short_term_context": short_term_context,
        }

    def render_hybrid(self, nle_obsv):
        ascii_map = self.ascii_render(nle_obsv["tty_chars"])
        cursor = nle_obsv["tty_cursor"]
        cursor = f"(x={cursor[1]}, y={cursor[0]})"
        ascii_map = "\n".join(ascii_map.split("\n")[1:])  # remove first line

        nle_obsv["map"] = ascii_map
        nle_obsv["text_cursor"] = nle_obsv["text_cursor"] + "\n" + cursor

        long_term_observations = [
            ("text_message", "message"),
            ("text_glyphs", "language observation"),
            ("text_cursor", "cursor"),
            ("map", "map"),
        ]
        short_term_observation = [
            ("text_inventory", "inventory"),
        ]

        long_term_context = "\n".join([f"{name}:\n{nle_obsv[key]}\n" for key, name in long_term_observations])
        short_term_context = "\n".join([f"{name}:\n{nle_obsv[key]}\n" for key, name in short_term_observation])

        return {
            "long_term_context": long_term_context,
            "short_term_context": short_term_context,
        }
