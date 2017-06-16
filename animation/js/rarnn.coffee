React = require 'react'
ReactDOM = require 'react-dom'
d3 = require 'd3'
fetch$ = require 'kefir-fetch'

color = d3.scaleOrdinal(d3.schemeCategory10)

FRAMERATE = 250

word_width = 55
level_height = 65
selection_height = 30
highlight_offset = -7

descend = (node, level) ->
    if level.length == 0
        return node
    else
        l = level[0]
        return descend node.children[l], level.slice(1)

Words = ({words, range}) ->
    <div className='words'>
        {words.slice(range[0], range[1] + 1).map (word, wi) ->
            <div style={transform: "translate(#{word_width * (wi + range[0])}px, 0px)", width: word_width} key=wi>
                <span style={width: word_width}>{word}</span>
                <span className='index' style={transform: "translate(0px, 15px)"}>{wi + range[0]}</span>
            </div>
        }
    </div>

Level = ({node, level_index, words}) ->
    # node = descend tree, level
    console.log '<Level> node=', node
    <div className='level' style={transform: "translate(0px, #{level_height * level_index + 20}px)"}>
        <Words range={node.position} words=words />
        <Highlight range={node.position} token=node.key />
        {if node.key[0] == '%'
            <UnderHighlight range={node.position} token=node.key />
        }
        <UnderToken range={node.position} token=node.key />
    </div>

Highlight = ({range, token}) ->
    x = range[0] * word_width
    y = highlight_offset
    <div
        style={
            border: "2px solid #{color(token)}"
            transform: "translate(#{x}px, #{y}px)"
            width: (range[1] - range[0] + 1) * word_width
            height: selection_height
            borderRadius: 4
        }
    />

UnderHighlight = ({range, token}) ->
    xoff = 0
    x = (range[0] - xoff) * word_width
    y = -7 + selection_height
    width = (range[1] - range[0] + 1) * word_width

    <div style={transform: "translate(#{x}px, #{y}px)"}>
        <div className='under' width=width>
            <div
                style={
                    x: 0
                    y: 0
                    width: width
                    height: selection_height
                    backgroundColor: color(token)
                    opacity: 0.1
                }
                rx=4
                ry=4
            />
        </div>
    </div>

UnderToken = ({range, token}) ->
    tx = (range[0] + (range[1] - range[0]) * 0.5) * word_width
    ty = selection_height
    <span
        className='token'
        style={
            width: word_width * 2
            transform: "translate(#{tx - word_width / 2}px, #{ty}px)"
            color: color(token)
        }>
        {token}
    </span>

Animation = React.createClass
    getInitialState: ->
        levels: [[]]

    componentDidMount: ->
        all_levels = []

        get_levels = (node, l=[]) ->
            console.log 'd', node.key, l
            all_levels.push l
            if node.children?
                for c in [0...node.children.length]
                    child = node.children[c]
                    if child.key?
                        l_ = l.slice(0)
                        l_.push c
                        get_levels child, l_

        get_levels @props.tree
        console.log 'all levels', all_levels
        @all_levels = all_levels

        @level_interval = setInterval @addLevel, FRAMERATE

    addLevel: ->
        {levels} = @state
        levels.push @all_levels[levels.length]
        if levels.length == @all_levels.length
            clearTimeout @level_interval
        @setState {levels}

    render: ->
        {tree, words} = @props
        <div className='svg' height='100%' width='100%' style={backgroundColor: '#fff'}>
            {@state.levels.map (level, li) ->
                node = descend tree, level
                level_index = level.length
                <Level node=node level_index=level_index words=words key=li />
            }
        </div>

type_str = 'hi maia if the price of bitcoin goes above 999 please turn the office light green and turn on the tea'

App = React.createClass
    getInitialState: ->
        type_i: 0
        input: ''

    componentDidMount: ->
        @refs.input.focus()
        setTimeout @fakeType, 500

    fakeType: ->
        {type_i} = @state
        console.log 'i', type_i
        input = type_str.slice(0, type_i)
        type_i += 1
        @setState {input, type_i}
        if @state.type_i <= type_str.length
            setTimeout @fakeType, Math.random() * 50
        else
            setTimeout @parse, 200

    changeInput: (e) ->
        input = e.target.value
        @setState {input}

    parse: (e) ->
        e?.preventDefault()
        fetch$('post', 'http://localhost:9922/parse.json', {body: {body: @state.input}})
            .onValue @didParse
            .onError (err) ->
                alert "error #{err}"

    didParse: ({words, parsed}) ->
        parsed.key = '%'
        console.log 'parsed', parsed
        @setState {words, parsed}

    reset: ->
        @setState {parsed: null}

    render: ->
        <div>
            {if @state.parsed?
                <Animation tree=@state.parsed words=@state.words />
            else
                <form onSubmit=@parse>
                    <input value=@state.input onChange=@changeInput ref='input' />
                </form>
            }
            {if @state.parsed? and false
                <button onClick=@reset>Reset</button>
            }
        </div>

ReactDOM.render <App />, document.getElementById 'app'

