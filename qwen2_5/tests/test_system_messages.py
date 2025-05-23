import unittest
from importlib import resources

from transformers import AutoTokenizer


template_file = resources.files(__package__.split(".")[0]).joinpath(
    "chat_template.jinja"
)


class TestSystemMessages(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # NOTE: This may fail due to different default system message accross Qwen models.
    def test_default_system_message(self):
        messages = [
            {
                "role": "user",
                "content": "Hello, how are you?",
            }
        ]
        expected = self.tokenizer.apply_chat_template(messages, tokenize=False)

        with template_file.open("r", encoding="utf-8") as f:
            self.tokenizer.chat_template = f.read()
        actual = self.tokenizer.apply_chat_template(messages, tokenize=False)

        self.assertEqual(actual, expected)

    def test_custom_system_message(self):
        messages = [
            {
                "role": "system",
                "content": "You are not Qwen.",
            },
            {
                "role": "user",
                "content": "Hello, how are you?",
            },
        ]
        expected = self.tokenizer.apply_chat_template(messages, tokenize=False)

        with template_file.open("r", encoding="utf-8") as f:
            self.tokenizer.chat_template = f.read()
        actual = self.tokenizer.apply_chat_template(messages, tokenize=False)

        self.assertEqual(actual, expected)

    # Qwen's chat template does not support structured system message.
    def test_openai_content(self):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Qwen."}],
            },
        ]
        expected = """<|im_start|>system
You are Qwen.<|im_end|>
"""

        with template_file.open("r", encoding="utf-8") as f:
            self.tokenizer.chat_template = f.read()
        actual = self.tokenizer.apply_chat_template(messages, tokenize=False)

        self.assertEqual(actual, expected)

    def test_render_tools(self):
        def get_current_temperature(location: str, unit: str) -> float:
            """Get the current temperature at a location.

            Args:
                location: The location to get the temperature for, in the format "City, Country"
                unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
            Returns:
                The current temperature at the specified location in the specified units, as a float.
            """
            return 22.0  # A real function should probably actually get the temperature!

        tools = [get_current_temperature]
        messages = [
            {
                "role": "system",
                "content": "You are Qwen.",
            },
            {
                "role": "user",
                "content": "Hello, how are you?",
            },
        ]
        expected = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False
        )

        with template_file.open("r", encoding="utf-8") as f:
            self.tokenizer.chat_template = f.read()
        actual = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False
        )

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
