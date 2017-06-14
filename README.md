# Recursive Application of Recurrent Neural Networks

A simple model for intent parsing that supports complex nested intents.

![](https://i.imgur.com/1MF5aLE.png)

## Model

The core model is a regular seq2seq/encoder-decoder model with attention. The attention model is from [Luong et al.'s "Effective Approaches to Attention-based Neural Machine Translation"](https://arxiv.org/abs/1508.04025) using dot-product based attention energies, with one important difference: there is no softmax layer, allowing attention to focus on multiple tokens at once. Instead a sigmoid layer is added to squeeze outputs between 0 and 1.

The encoder and decoder take one additional input `context` which represents the type of phrase, e.g. `%setLightState`. At the top level node the context is always `%`.

The encoder encodes the input sequence into a series of vectors using a bidirectional GRU. The decoder "translates" this into a sequence of phrase tokens, given the encoder outputs and current context, e.g. "turn off the office light" + `%setLightState` &rarr; `[$on_off, $light]`.

Once the decoder has chosen tokens and alignments, the phrase tokens and selection of inputs are used as the context and inputs of the next iteration. This recurs until no more phrase tokens are found.

## Data

Of course in order to parse a nested intent sructure, we need nested intent training data. Examples are generated with a natural language templating language called [Nalgene](https://github.com/spro/nalgene) which generates both a flat string and a parse tree. Templates define a number of `%phrases` and `$values` (leaf nodes) as well as filler `~synonyms` and the generator takes a random walk down the tree to build each example. Here's a snippet from the grammar file:

```
%if
    ~if %condition then %sequence

%sequence
    ~please? %action
    ~please? %action ~also ~please? %action

%getSwitchState
    the $switch_name state
   
%getTemperature
    the temperature in the $room_name
    the $room_name temperature

%getPrice
    price of $asset
    $asset price
```

## Author

Sean Robertson

```bibtex
@misc{Robertson2017,
    author = {Robertson, Sean},
    title = {Recursive Application of Recurrent Neural Networks},
    year = {2017},
    url = {https://github.com/spro/RARNN}
}
```

