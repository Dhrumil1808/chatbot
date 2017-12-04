# chatbot


Generate Data :


```
	python DeterministicGenerator.py test.gram 
  
```


Pre-processing (requires data generation) :


```
	python data_preprocessing.py
  
```

* navigate to CNN dir

Train CNN

```
    python train.py ./data/train_new.csv ./training_config.json

```

Predict CNN


```
    python predict.py ./trained_results_1512254446 ./data/small_samples_new.csv

```

Run on python 2.6/2.7 from here

Slackbot :

```
pip install slackclient
```

```
pip install git+https://github.com/mit-nlp/MITIE.git
```


```

cd slackbot

python starterbot.py

```







