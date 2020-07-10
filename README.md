### Issue with data

```
wc -l data/*
  2000 test_source.txt
  2000 test_target.txt
  7260 train_source.txt
  5539 train_target.txt
 16799 total
```

as the train_source and train_target files had a different number of lines I chose to simply look at the first 5539 lines of train_source.txt after contacting dl_papago_mt_recruit@navercorp.com via email.

## Model

### Encoder

RNN Encoder using an embedding layer and gated recurrent units.

### Decoder

* **Attentional Mechanism** builds queries from output sequence produced so far and <key,value> pairs from the encoded input to build an indexing scheme of interesting information to look up.

* **Drop-out** picks random nodes to not be part of the graph to decrease the dependency on single paths and increase robustness

* **Teacher-forcing** makes the model sometimes use the target sequence during training to give the decoder a high quality input. Can lead to faster convergence.
## Evaluation Metric

At every step the decoder produces a tensor which for every token gives a (unnormalized) probability for this token to be the next in the output sequence. (logit)

This probability distribution is compared with the output symbol we expect according to ground truth using **negative log-likelihood**:

```math
  Loss = -log(p_i) 

  where p_i is softmax of ground truth token 
  over all possible tokens
```
## Experimental Results

source [33, 416, 437, 200, 35, 437, 68, 35, 140, 200, 157, 271, 68, 105, 157, 68, 140, 227, 437, 68, 416, 105, 157, 437, 68, 304, 584, 95, 68, 304, 584, 95, 437, 105, 402, 157, 437, 95, 35, 327, 659]
target [64, 211, 108, 68, 85, 620, 149, 68, 430, 68, 569, 68, 550, 430, 68, 241, 68, 241, 211, 565, 85, 659]
 model 64 211 108 68 550 68 68 68 68 68 68 68 68 68 68 68 68 68 68 68 <EOS>

source [29, 227, 437, 157, 68, 271, 584, 437, 35, 68, 311, 584, 342, 95, 68, 304, 416, 105, 402, 227, 140, 68, 416, 437, 200, 156, 437, 113, 659]
target [564, 68, 569, 108, 68, 5, 189, 68, 64, 550, 158, 359, 68, 211, 543, 659]
 model 564 68 68 68 68 68 68 68 68 68 68 68 68 68 <EOS>

source [271, 584, 68, 105, 68, 227, 200, 156, 437, 68, 140, 584, 68, 263, 437, 68, 227, 584, 35, 33, 105, 140, 200, 416, 105, 23, 437, 271, 113, 659]
target [370, 68, 607, 158, 68, 81, 68, 397, 68, 405, 68, 261, 85, 4, 116, 550, 158, 108, 149, 659]
 model 370 68 68 81 68 68 68 68 68 68 68 68 <EOS>

source [271, 105, 200, 95, 95, 227, 437, 200, 659]
target [89, 158, 189, 211, 189, 659]
 model 19 189 189 <EOS>

source [29, 227, 437, 95, 437, 68, 105, 35, 68, 140, 227, 437, 68, 271, 437, 33, 200, 95, 140, 78, 437, 157, 140, 68, 35, 140, 584, 95, 437, 113, 659]
target [503, 189, 68, 158, 108, 68, 569, 68, 19, 79, 359, 438, 359, 68, 85, 65, 189, 659]
 model 503 189 68 158 108 68 569 68 68 68 68 68 68 68 68 <EOS>

source [105, 453, 78, 68, 200, 68, 33, 437, 95, 35, 584, 157, 68, 584, 304, 68, 304, 437, 29, 68, 29, 584, 95, 271, 35, 327, 659]
target [607, 479, 68, 189, 68, 290, 625, 68, 640, 68, 187, 68, 170, 108, 659]
 model 607 479 68 642 211 68 68 68 68 68 68 68 68 68 68 108 68 <EOS>

source [402, 200, 78, 437, 659]
target [16, 479, 659]
 model 115 479 <EOS>

source [33, 416, 437, 200, 35, 437, 68, 402, 105, 304, 140, 513, 29, 95, 200, 33, 68, 105, 140, 327, 659]
target [64, 211, 108, 68, 514, 240, 359, 391, 68, 339, 659]
 model 64 211 108 68 68 68 68 68 68 68 68 <EOS>

source [227, 437, 68, 140, 95, 105, 437, 271, 68, 140, 584, 68, 35, 437, 109, 342, 200, 416, 416, 311, 68, 227, 200, 95, 200, 35, 35, 68, 78, 437, 327, 659]
target [153, 68, 264, 550, 158, 68, 397, 68, 549, 635, 471, 211, 68, 384, 435, 85, 68, 517, 659]
 model 87 68 68 149 68 68 68 68 68 68 68 68 68 68 68 <EOS>

source [29, 227, 200, 140, 68, 227, 437, 68, 416, 584, 584, 619, 35, 68, 416, 105, 619, 437, 659]
target [283, 68, 153, 68, 494, 85, 68, 550, 501, 659]
 model 261 86 68 68 68 68 <EOS>


## Task Description

The objective of this task is to create a model that approximates a mapping function from an input sequence of integers ("source") to an output sequence of integers ("target") using a training data set (`train_source.txt`, `train_target.txt`), and achieve best generalization performance on a held-out test set (`test_source.txt`, `test_target.txt`). Use any technique and framework you think is appropriate. 

Please submit a link to a GitHub repository, containing your code, and `README` which describes the following:

1. your experiment design (including baselines and models and/or data exploration results)
2. evaluation metrics
3. experimental results.
