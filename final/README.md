# Topic: Conversation in TV shows

## Import Package

| numpy | scipy | jieba | gensim | Keras | scikit-learn | word2vec |
|-------|-------|-------|--------|-------|--------------|----------|
|1.13.3 |0.19.1 | 0.39  | 3.1.0  | 2.0.8 |    0.19.1    |  0.9.2   |


## Constructure
final/README.md

`final/src/.` -> source code which can show kaggle score

`final/src/Joes` -> other methods we try

`final/src/Sun` -> other methods we try
     
## Usage (Kaggle Reproduced, do this part)
You need to change to directory to `src/` first.

1. testing:

	`bash test.sh <testing data file path> <output result file path>`

	For example:
	`bash test.sh ../provideData/testing_data.csv ../predict.csv`

2. training:

	`bash train.sh <training data directory>`

	For example:
	`bash train.sh ../provideData/training_data/`

## Usage (other implement method, for report and not for reproduced)
You need to change to directory to `src/Joe/` and put the `provideData/` in `src/Joe/` first.
You need to create `corpus.txt` from the 1. above. and put it in `src/Joe`.

1. tf-idf:
	
	`bash tf-idf.sh`

2. auto-encoder:

	`bash auto.sh`

3. average method (with word2vec):

	`bash average.sh`

You need to change to directory to `src/Sun/` and put the `provideData/` in `src/Sun/` first.

4. `python3 create_vector.py`

5. `python3 w2v_rnn.py` : this method should have enough memory.

