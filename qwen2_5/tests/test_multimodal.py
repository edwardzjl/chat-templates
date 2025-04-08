import unittest
from importlib import resources

from transformers import AutoProcessor, AutoTokenizer


template_file = resources.files(__package__.split(".")[0]).joinpath(
    "chat_template.jinja"
)


class TestMultiModalMessages(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    def test_text_only_message(self):
        messages = [
            {
                "role": "system",
                "content": "You are Qwen.",
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hi"}],
            },
        ]
        expected = self.processor.apply_chat_template(messages, tokenize=False)

        with template_file.open("r", encoding="utf-8") as f:
            self.tokenizer.chat_template = f.read()
        actual = self.tokenizer.apply_chat_template(messages, tokenize=False)

        self.assertEqual(actual, expected)

    def test_text_only_message2(self):
        messages1 = [
            {
                "role": "system",
                "content": "You are Qwen.",
            },
            {
                "role": "user",
                "content": "Hi",
            },
        ]
        messages2 = [
            {
                "role": "system",
                "content": "You are Qwen.",
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hi"}],
            },
        ]

        with template_file.open("r", encoding="utf-8") as f:
            self.tokenizer.chat_template = f.read()

        expected = self.tokenizer.apply_chat_template(messages1, tokenize=False)
        actual = self.tokenizer.apply_chat_template(messages2, tokenize=False)

        self.assertEqual(actual, expected)

    def test_single_content_type(self):
        messages = [
            {
                "role": "system",
                "content": "You are Qwen.",
            },
            {
                "role": "user",
                "content": [{"type": "image", "image_url": "https://foo.com/bar.png"}],
            },
        ]
        expected = self.processor.apply_chat_template(messages, tokenize=False)

        with template_file.open("r", encoding="utf-8") as f:
            self.tokenizer.chat_template = f.read()

        actual = self.tokenizer.apply_chat_template(messages, tokenize=False)

        self.assertEqual(actual, expected)

    def test_multiple_content_type(self):
        messages = [
            {
                "role": "system",
                "content": "You are Qwen.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "image_url": "https://foo.com/bar.png"},
                    {"type": "video", "video_url": "https://foo.com/bar.mp4"},
                ],
            },
        ]
        expected = self.processor.apply_chat_template(messages, tokenize=False)

        with template_file.open("r", encoding="utf-8") as f:
            self.tokenizer.chat_template = f.read()

        actual = self.tokenizer.apply_chat_template(messages, tokenize=False)

        self.assertEqual(actual, expected)

    @unittest.skip(
        "Currently `add_vision_id` has no effect in processor. See <https://github.com/QwenLM/Qwen2.5-VL/issues/716>"
    )
    def test_add_vision_id(self):
        messages = [
            {
                "role": "system",
                "content": "You are Qwen.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "image_url": "https://foo.com/bar.png"},
                    {"type": "video", "video_url": "https://foo.com/bar.mp4"},
                ],
            },
        ]
        expected = self.processor.apply_chat_template(
            messages, tokenize=False, add_vision_id=True
        )

        with template_file.open("r", encoding="utf-8") as f:
            self.tokenizer.chat_template = f.read()

        actual = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_vision_id=True
        )

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
