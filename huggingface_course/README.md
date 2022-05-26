# Huggingface Course
## Transformers, what they can do ?
### `pipline()` function
It connects the model with its necessary preprocessing and postprocessing steps, allowing us to directly input text and get an intelligible answer. 


```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```
output:
```
[{'label': 'POSITIVE', 'score': 0.9598047137260437}]
```
The pipeline selects a particular pretrained model that has been fine tuned for sentiment analysis in english. the model is downloaded and chaced when you create the classifier object. 
There are 3 main steps involved when you pass some text to a pipeline.
1. The text is preprocessed into a format the model can understand.
1. The preprocessed inputs are passed to the model.
1. The predictions of the model are postprocessed into a format that is more readable.
Some of the currently available pipelines are:

  - feature-extraction (get the vector representation of a text)
   - fill-mask
   - ner (named entity recognition)
   - question-answering
   - sentiment-analysis
   - summarization
   - text-generation
   - translation
   - zero-shot-classification