# llm-tokenwise-inference
Token-wise and real-time display Inference module for Llama2 and other LLMs.

# Getting Started

```powershell
cd llm-tokenwise-inference
pip install -r requirements.txt
```

- Run the following program in ipython or Jupyter.

```python
from llminferencepkg import TokenWiseLLM
model = TokenWiseLLM("path/to/model") # or HF repository
model.inference("Question")
```
![tokenwisellm](https://github.com/keisuke-okb/llm-tokenwise-inference/assets/70097451/1bc5c601-e241-4050-8d14-69b13bab696a)
