# Topic: Conversation in TV shows

## Import Package

`numpy` 1.13.3
`scipy` 0.19.1
`jieba` 0.39
`gensim` 3.1.0
`Keras` 2.0.8
`scikit-learn` 0.19.1
`word2vec` 0.9.2

## Constructure
final/README.md

final/src/. -> source code which can show kaggle score

final/src/Joes -> other methods we try
     
## Usage (Kaggle Reproduce)
You need to change to directory to `src/` first.

1. testing:

	`bash test.sh <training data directory>`

	For example:
	`bash test.sh ../provideData/training_data/`

2. training:

	`bash train.sh <testing data file path> <output result file path>`

	For example:
	`bash train.sh ../provideData/testing_data.csv ../predict.csv`


## Usage (other implement method)
You need to change to directory to `src/Joe/` and put the `provideData/` in `src/Joe/` first.

1. tf-idf:
	
	`bash tf-idf.sh`

2. auto-encoder:

	`bash auto.sh`

3. average method (with word2vec):

	`bash average.sh`


