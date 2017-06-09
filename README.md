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

