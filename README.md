# deception-prompt-hack

# Setup
```
conda create -n decept python==3.10
conda activate decept
pip install -r requirements.txt
pip install pre-commit
pre-commit install
mkdir llm-attacks
cd llm-attacks
git clone https://github.com/llm-attacks/llm-attacks.git
cd llm-attacks
pip install -e .
cd ../../
```


## Experiments

To run the knowledge retrievel experiments run the python file:

`reversed_knowledge_retrieval.py`
