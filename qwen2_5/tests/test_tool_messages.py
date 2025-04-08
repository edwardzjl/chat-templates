import unittest
from importlib import resources

from transformers import AutoTokenizer


template_file = resources.files(__package__.split(".")[0]).joinpath(
    "chat_template.jinja"
)


class TestToolMessages(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    def test_single_tool_message(self):
        messages = [
            {
                "role": "system",
                "content": "You are Qwen.",
            },
            {
                "role": "tool",
                "content": "foo",
            },
        ]
        expected = self.tokenizer.apply_chat_template(messages, tokenize=False)

        with template_file.open("r", encoding="utf-8") as f:
            self.tokenizer.chat_template = f.read()
        actual = self.tokenizer.apply_chat_template(messages, tokenize=False)

        self.assertEqual(actual, expected)

    def test_multiple_tool_messages(self):
        messages = [
            {
                "role": "system",
                "content": "You are not Qwen.",
            },
            {
                "role": "tool",
                "content": "foo",
            },
            {
                "role": "tool",
                "content": "bar",
            },
        ]
        expected = self.tokenizer.apply_chat_template(messages, tokenize=False)

        with template_file.open("r", encoding="utf-8") as f:
            self.tokenizer.chat_template = f.read()
        actual = self.tokenizer.apply_chat_template(messages, tokenize=False)

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
