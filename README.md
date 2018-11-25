# MultiLabelClassification

MultiLabel intent classification is needed for a chatbot to simulate human-like conversational experience. We are using ClassifierChain from skmultilearn to classify a user utterance into multiple intents. Popular NLP APIs such as Dialogflow, Alexa/Lex, LUIS, etc don't support multilabel intent classification, hence this implementation fills that gap.

It uses Multinomial Naive Bayes as the estimator and WordNet lemmatizer instead of the default one. Datetime and numbers should be replaced with placeholders in the training data, however, due to a lack of a good datetime extraction package, we cannot clean up the data dynamically for now.

The output is a list of intents with probabilities. 
