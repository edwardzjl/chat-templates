import unittest

from transformers import AutoTokenizer


class TestAssistantMessages(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    def test_simple_assistant_message(self):
        messages = [
            {
                "role": "system",
                "content": "You are Qwen.",
            },
            {
                "role": "user",
                "content": "Hello, how are you?",
            },
            {
                "role": "assistant",
                "content": "Fine thank you. And you?",
            },
        ]
        expected = self.tokenizer.apply_chat_template(messages, tokenize=False)

        with open("./chat_template.jinja") as f:
            self.tokenizer.chat_template = f.read()
        actual = self.tokenizer.apply_chat_template(messages, tokenize=False)

        self.assertEqual(actual, expected)

    def test_render_tool_call(self):
        tool_calls = [
            {
                "name": "get_current_temperature",
                "arguments": {"location": "Paris, France", "unit": "celsius"},
            }
        ]
        messages = [
            {
                "role": "system",
                "content": "You are Qwen.",
            },
            {
                "role": "user",
                "content": "What is the weather like in Paris?",
            },
            {"role": "assistant", "tool_calls": tool_calls},
        ]
        expected = self.tokenizer.apply_chat_template(messages, tokenize=False)

        with open("./chat_template.jinja") as f:
            self.tokenizer.chat_template = f.read()
        actual = self.tokenizer.apply_chat_template(messages, tokenize=False)

        self.assertEqual(actual, expected)

    def test_render_function_tool_call(self):
        tool_calls = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "arguments": {"location": "Paris, France", "unit": "celsius"},
                },
            }
        ]
        messages = [
            {
                "role": "system",
                "content": "You are Qwen.",
            },
            {
                "role": "user",
                "content": "What is the weather like in Paris?",
            },
            {"role": "assistant", "tool_calls": tool_calls},
        ]
        expected = self.tokenizer.apply_chat_template(messages, tokenize=False)

        with open("./chat_template.jinja") as f:
            self.tokenizer.chat_template = f.read()
        actual = self.tokenizer.apply_chat_template(messages, tokenize=False)

        self.assertEqual(actual, expected)

    def test_render_multiple_tool_calls(self):
        tool_calls = [
            {
                "name": "get_current_temperature",
                "arguments": {"location": "Paris, France", "unit": "celsius"},
            },
            {
                "name": "get_current_temperature",
                "arguments": {"location": "Hangzhou, China", "unit": "celsius"},
            },
        ]
        messages = [
            {
                "role": "system",
                "content": "You are Qwen.",
            },
            {
                "role": "user",
                "content": "What is the weather like in Paris?",
            },
            {"role": "assistant", "tool_calls": tool_calls},
        ]
        expected = self.tokenizer.apply_chat_template(messages, tokenize=False)

        with open("./chat_template.jinja") as f:
            self.tokenizer.chat_template = f.read()
        actual = self.tokenizer.apply_chat_template(messages, tokenize=False)

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
