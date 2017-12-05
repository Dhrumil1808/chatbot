# chatbot


Generate Data :


```
	python DeterministicGenerator.py test.gram 
  
```


Pre-processing (requires data generation) :


```
	python data_preprocessing.py
  
```

* navigate to sqs dir

Train CNN

```
    python train.py 

```

Predict CNN


```
    python predict.py 

```

Run on python 2.6/2.7 from here

Slackbot :

```
pip install slackclient
```

For NER:

```
pip install git+https://github.com/mit-nlp/MITIE.git
```

Also download :

https://www.dropbox.com/s/d4ncdbg88j4zzvs/new_ner_model.dat?dl=0
https://www.dropbox.com/s/3yhg2fm9qnzxu5y/total_word_feature_extractor.dat?dl=0

```

cd slackbot

python starterbot.py

```







