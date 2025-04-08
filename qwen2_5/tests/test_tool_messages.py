import unittest

from transformers import AutoTokenizer


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

        with open("./chat_template.jinja") as f:
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

        with open("./chat_template.jinja") as f:
            self.tokenizer.chat_template = f.read()
        actual = self.tokenizer.apply_chat_template(messages, tokenize=False)

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
