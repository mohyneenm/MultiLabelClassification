# MultiLabelClassification

MultiLabel intent classification is needed for a chatbot to simulate human-like conversational experience. We are using ClassifierChain from skmultilearn to classify a user utterance into multiple intents. Popular NLP APIs such as Dialogflow, Alexa/Lex, LUIS, etc don't support multilabel intent classification, hence this implementation fills that gap.

The output is a list of intents with probabilities. 
