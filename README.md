# ChatNLP
###### Chat sentiment evaluation using VADER model and ranking ranking

A python module that takes a list of chat messages and a chat popularity ranking  
and evaluates a chat sentiment ranking based on the VADER nlp model

##### Please feel free to contact me for help, i'd be glad to support (:

###Requirements:
* numpy
* scipy
* nltk
* matplotlib
* textblob
* urllib2

###Usage:
To evaluate chat run:
```python
from chat_evaluator import evaluate_chat

evaluation = evaluate_chat([messages list as strings], chat ranking as natural number)
```
>
*Note you can set translate_input=True to get input translation*

To evaluate model run:
```python
from chat_evaluator import evaluate_model

evaluate_model()
```

###Model Evaluation:
Currently the model is based upon the formula below:  
![Model Formula](https://raw.githubusercontent.com/shakedlokits/ChatNLP/master/equation.png)
-----------------------------------
Can be visualized upon the VADER testing data as such:  
![Model Visualization](https://raw.githubusercontent.com/shakedlokits/ChatNLP/master/model_evaluation.png)
> "Model visualization based on sentiment, ranking and alpha trade-off"

###MIT License

Copyright (c) 2016 Shaked Lokits

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
