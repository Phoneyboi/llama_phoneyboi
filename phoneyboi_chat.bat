@echo off

REM Loop through variables in config.txt file
FOR /F "tokens=1,2 delims==" %%G IN (config.txt) DO (
    SET %%G=%%H
)

echo Model path is: %MODELS_PATH%
echo Environment path is: %ENV_PATH%
echo Python chat path is: %PY_CHAT_PATH%
echo Tokenizer path is: %TOKEN_PATH%

wsl bash -c '%ENV_PATH% -m torch.distributed.run --nproc_per_node=1 %PY_CHAT_PATH% --ckpt_dir %MODELS_PATH% --tokenizer_path %TOKEN_PATH% --max_seq_len 512 --max_batch_size 6'

pause