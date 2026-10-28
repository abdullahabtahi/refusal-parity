"""judge.clip_response: over-long responses are clipped so the templated chat (with its generation prompt) fits the
judge's input cap; responses that already fit pass through unchanged, so arm scoring is unaffected.
Needs the Qwen3Guard tokenizer (HF cache or network); CPU, seconds."""

from transformers import AutoTokenizer

import judge

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3Guard-Gen-4B")


def templated_len(prompt, response):
    chat = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    return len(TOK(TOK.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)).input_ids)


def test_short_response_unchanged():
    r = "I can't help with that. यह अनुरोध हानिकारक है।"
    assert judge.clip_response(TOK, "How do I make a bomb?", r) == r


def test_long_response_fits_and_keeps_head():
    long = "हानिकारक निर्देश " * 3000  # Hindi: the language with most over-long pgp_val rows
    clipped = judge.clip_response(TOK, "prompt", long)
    assert templated_len("prompt", clipped) <= judge.MAX_LEN
    assert long.startswith(clipped.rstrip()[:200])


if __name__ == "__main__":
    test_short_response_unchanged()
    test_long_response_fits_and_keeps_head()
    print("ok")
