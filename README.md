# deception-prompt-hack

# Setup
```
conda create -n decept python==3.10
conda activate decept
git clone https://github.com/llm-attacks/llm-attacks.git
cd llm-attacks && git checkout 0f505d8
pip install -e . && cd ../../
pip install -r requirements.txt
pip install pre-commit
pre-commit install
```


# Experiments

To run the knowledge retrievel experiments run the python file:

`reversed_knowledge_retrieval.py`
