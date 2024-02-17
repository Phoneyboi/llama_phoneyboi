@echo off
rem wsl /home/phoneyboi/miniconda3/envs/llama_mini/bin/python -c "torchrun --nproc_per_node 1 phoneyboi_chat_completion.py --ckpt_dir llama-2-7b-chat/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 6"

wsl bash -c '/home/phoneyboi/miniconda3/envs/llama_mini/bin/python -m torch.distributed.run --nproc_per_node=1 /mnt/c/Users/helbi/MRLabs/llama/llama/phoneyboi_chat_completion.py --ckpt_dir /mnt/c/Users/helbi/MRLabs/llama/llama/llama-2-7b-chat/ --tokenizer_path /mnt/c/Users/helbi/MRLabs/llama/llama/tokenizer.model --max_seq_len 512 --max_batch_size 6'

pause