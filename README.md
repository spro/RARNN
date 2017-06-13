# Recursive Application of Recurrent Neural Networks

A model for intent parsing that supports complex nested intents.

```
> If the temperature in Tokyo is equal to 56 then turn off the living room light.
```

The intent can be represented as a tree of phrases and values:

```
( %if
	( %condition
		( %getWeather
			( $key temperature )
			( $location Tokyo ) )
		( $operator equal to )
		( $number 56 ) )
	( %command
		( %setLightState
			( $on_off off )
			( $light_name living room light ) ) ) )
```

The goal of RARNN is to recursively identify phrases and their top-level components (sub-phrases and values), then recursively visit sub-phrases to find their components, until a full phrase tree is built.

## Example

For the above sentence,

1. Applied to the whole sentence, the model would recognize the overarching `%if` phrase.

2. Applied to the `%if` phrase (which happens to be the whole sentence), the model would recognize sub-phrases `%condition` and `%command` (marked with square brackets).
	```
	> If [the temperature in Tokyo is equal to 56] then [turn off the living room light].
	```

3. Applied to the `%condition` phrase, the model would recognize the `%getWeather` sub-phrase and `$operator` and `$number` values.
	```
	> [the temperature in Tokyo] is [equal to] [56]
	```

4. Applied to the `%getWeather` phrase, the model would recognize the `$temperature` and `$location` values.
	```
	> the [temperature] in [Tokyo]
	```

## Model

The core model is a regular seq2seq/encoder-decoder model with attention. The attention model is from [Luong et al.'s "Effective Approaches to Attention-based Neural Machine Translation"](https://arxiv.org/abs/1508.04025) using dot-product based attention energies, with one important difference: there is no softmax layer, allowing attention to focus on multiple tokens at once. Instead a sigmoid layer is added to squeeze outputs between 0 and 1.

The encoder and decoder take one additional input `context` which represents the type of phrase, e.g. `%setLightState`. At the top level node the context is always `%`.

The encoder encodes the input sequence into a series of vectors using a bidirectional GRU. The decoder "translates" this into a sequence of phrase tokens, given the encoder outputs and current context, e.g. "turn off the office light" + `%setLightState` &rarr; `[$on_off, $light]`.

Once the decoder has chosen tokens and alignments, the phrase tokens and selection of inputs are used as the context and inputs of the next iteration. This recurs until no more phrase tokens are found.

