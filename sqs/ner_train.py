from mitie import *

trainer = ner_trainer("total_word_feature_extractor.dat")


time_sentences = []
time_sentences.append("what time is the class")
time_sentences.append("what time is the course")
time_sentences.append("what time is the CMPE297")

for sentence in time_sentences:

    print tokenize(sentence)
    sample1 = ner_training_instance(tokenize(sentence))
    sample1.add_entity(xrange(4,5), "class")
    trainer.add(sample1)

lab_proj_sentences = []

lab_proj_sentences.append("what do we do for lab one")
lab_proj_sentences.append("what do we do for assignment one")
lab_proj_sentences.append("what do we do for first lab")
lab_proj_sentences.append("what do we submit for lab one")
lab_proj_sentences.append("what do we submit for first lab")
lab_proj_sentences.append("what do we do for project")
lab_proj_sentences.append("what do we submit for project")
lab_proj_sentences.append("what do we submit for the project")
lab_proj_sentences.append("what do we submit for assignment")

for sentence in lab_proj_sentences:
    print tokenize(sentence)
    sample2 = ner_training_instance(tokenize(sentence))
    sample2.add_entity(xrange(5, 6), "lab or project")
    trainer.add(sample2)



person_sentences = []

person_sentences.append("who is Simon Shim")
person_sentences.append("who is Professor Shim")
person_sentences.append("who is Abhiram")
person_sentences.append("who is Srivatsa")

for sentence in person_sentences:
    print tokenize(sentence)
    sample3 = ner_training_instance(tokenize(sentence))
    sample3.add_entity(xrange(2, 3), "person")
    trainer.add(sample3)
'''
p_sentences = []

p_sentences.append("who is Simon Shim")
p_sentences.append("who is Professor Shim")
p_sentences.append("who is Abhiram")
p_sentences.append("who is Srivatsa")

for sentence in lab_proj_sentences:
    print tokenize(sentence)
    sample = ner_training_instance(tokenize(sentence))
    sample.add_entity(xrange(3, 3), "person")

'''

# The trainer can take advantage of a multi-core CPU.  So set the number of threads
# equal to the number of processing cores for maximum training speed.
trainer.num_threads = 4

# This function does the work of training.  Note that it can take a long time to run
# when using larger training datasets.  So be patient.
ner = trainer.train()

# Now that training is done we can save the ner object to disk like so.  This will
# allow you to load the model back in using a statement like:
#   ner = named_entity_extractor("new_ner_model.dat").
ner.save_to_disk("new_ner_model.dat")

# But now let's try out the ner object.  It was only trained on a small dataset but it
# has still learned a little.  So let's give it a whirl.  But first, print a list of
# possible tags.  In this case, it is just "person" and "org".
print ("tags:", ner.get_possible_ner_tags())