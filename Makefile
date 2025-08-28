install:
	python3 -m venv venv  
	source venv/bin/activate && pip install -r requirements.txt

train:
	python3  src/train.py --lr 0.001 --optimizer adam --dropout 0.5 --epochs 1 --batch_size 64

train-sgd:
	python3  src/train.py --lr 0.01 --optimizer sgd --dropout 0.3 --epochs 1 --batch_size 128

test:
	PYTHONPATH=src pytest -v 
