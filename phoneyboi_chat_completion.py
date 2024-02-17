# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog


def chat(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """

    print("PhoneyGPT Initializing...")

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    conversation_history: List[Dialog] = []

    while True:
        user_input = input("You: ")
        dialog = [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": user_input},
        ]
        conversation_history.append(dialog)

        result = generator.chat_completion(
            conversation_history[-5:],  # Keep the last 5 exchanges
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[-1]  # Get the latest response

        response = result['generation']['content']
        print(f"Bot: {response}")


if __name__ == "__main__":
    fire.Fire(chat)
