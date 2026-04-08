clear
python make_dataset.py
python compute_similarities.py

python -m spacy download en_core_web_sm

python qa.py --model meta-llama/Llama-3.2-1B-Instruct
python qa.py --model google/gemma-3-1b-it
python qa.py --model Qwen/Qwen3-4B
python qa.py --model google/gemma-3-4b-it

python qa.py --model Qwen/Qwen3-8B
python qa.py --model meta-llama/Llama-3.1-8B-Instruct

python merge.py
tar -czf dataset_final.tar.gz dataset_final.json