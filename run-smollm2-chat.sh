#!/bin/bash

PROMPT=$(cat <<'EOF'
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hello! How can I help you today?<|im_end|>
<|im_start|>user
Who was Isaac Newton?<|im_end|>
EOF
)

./bin/smollm2.exe -m "models/smollm2-135m-instruct/smollm2.bin" -n 1000 -p "$PROMPT"


