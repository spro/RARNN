import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sconce
import somata

from nalgene.generate import *

USE_CUDA = False

parsed = parse_file('.', 'grammar.nlg')
parsed.map_leaves(tokenizeLeaf)

def descend(node, fn, child_type='phrase', returns=None):
    if returns is None: returns = []
    returned = fn(node)
    returns.append(returned)

    for child in node.children:
        if (child_type is None) or (child.type == child_type):
            descend(child, fn, child_type, returns)

    return returns

def ascend(node, fn):
    if node.parent is None:
        return fn(node)
    else:
        return ascend(node.parent, fn)

# Building input and output vocabularies

input_tokens = []

def get_input_tokens(node):
    if node.type == 'word':
        input_tokens.append(node.key)

descend(parsed, get_input_tokens, None)

input_tokens = list(set(input_tokens))
input_tokens = ['EOS'] + input_tokens
print(input_tokens)

output_tokens = [child.key for child in parsed.children if child.type in ['phrase', 'variable']]
output_tokens = ['EOS'] + output_tokens
print(output_tokens)

# Getting input and target data for nodes

def words_for_position(words, position):
    if position is None:
        return words
    start, end, length = position
    return words[start : end + 1]

def relative_position(node, parent):
    if parent.position is None:
        return node.position
    return node.position[0] - parent.position[0], node.position[1] - parent.position[0], node.position[2]

def data_for_node(flat, node):
    words = [child.key for child in flat.children]
    inputs = words_for_position(words, node.position)
    keys = [child.key for child in node.children]
    positions = [relative_position(child, node) for child in node.children]
    return node.key, inputs, list(zip(keys, positions))

walked_flat, walked_tree = walk_tree(parsed, parsed['%'], None)
data_for_node(walked_flat, walked_tree.children[0])

# Creating tensors for input and target data

def tokens_to_tensor(tokens, source_tokens, append_eos=True):
    indexes = []
    for token in tokens:
        indexes.append(source_tokens.index(token))
    if append_eos:
        indexes.append(0)
    return torch.LongTensor(indexes)

def ranges_to_tensor(ranges, seq_len):
    ranges_tensor = torch.zeros(len(ranges), seq_len)
    for r in range(len(ranges)):
        start, end, _ = ranges[r]
        ranges_tensor[r, start:end+1] = 1
    return ranges_tensor

ranges_to_tensor([(0, 2, 3), (4, 4, 1)], 5)

# Model

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=True)

    def forward(self, context_input, word_inputs):
        # TODO: Incorporate context input
        # TODO: Batching

        seq_len = word_inputs.size(0)
        batch_size = word_inputs.size(1)

        embedded = self.embedding(word_inputs.view(seq_len * batch_size, -1)) # Process seq x batch at once
        output = embedded.view(seq_len, batch_size, -1) # Resize back to seq x batch for RNN

        outputs, hidden = self.gru(output)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs

        return outputs, hidden

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attention_energies = Variable(torch.zeros(seq_len)) # B x 1 x S
        if USE_CUDA: attention_energies = attention_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attention_energies[i] = hidden.dot(encoder_outputs[i])

        # Squeeze to range 0 to 1, resize to 1 x 1 x seq_len
        return F.sigmoid(attention_energies).unsqueeze(0).unsqueeze(0)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.05):
        super(Decoder, self).__init__()

        # Keep parameters for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

        # Attention module
        self.attention = Attention()

    def forward(self, context_input, word_input, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        # TODO: Batching

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N

        # Combine context and embedded word, through RNN
        rnn_input = torch.cat((context_input.unsqueeze(0), word_embedded), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attention_weights = self.attention(rnn_output.squeeze(0), encoder_outputs)
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attention_weights

MAX_LENGTH = 50

class RARNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RARNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)

        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)

    def forward(self, context_input, word_inputs, word_targets=None):
        # Get embedding for context input
        context_embedded = self.embedding(context_input)

        input_len = word_inputs.size(0)
        target_len = word_targets.size(0) if word_targets is not None else MAX_LENGTH

        # Run through encoder
        encoder_outputs, encoder_hidden = self.encoder(context_embedded, word_inputs)
        decoder_hidden = encoder_hidden # Use encoder's last hidden state
        decoder_input = Variable(torch.LongTensor([0])) # EOS/SOS token
        if USE_CUDA:
            decoder_input = decoder_input.cuda()

        # Variables to store decoder and attention outputs
        decoder_outputs = Variable(torch.zeros(target_len, output_size))
        decoder_attentions = Variable(torch.zeros(target_len, input_len))
        if USE_CUDA:
            decoder_outputs = decoder_outputs.cuda()
            decoder_attentions = decoder_attentions.cuda()

        # Run through decoder
        for i in range(target_len):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(context_embedded, decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs[i] = decoder_output
            decoder_attentions[i] = decoder_attention

            # Teacher forcing with known targets, if provided
            if word_targets is not None:
                decoder_input = word_targets[i]

            # Sample with last outputs
            else:
                max_index = decoder_output.topk(1)[1].data[0][0]
                decoder_input = Variable(torch.LongTensor([max_index]))
                if USE_CUDA:
                    decoder_input = decoder_input.cuda()

                if max_index == 0: break # EOS

        # Slice outputs
        if word_targets is None:
            decoder_outputs = decoder_outputs[:i]
            decoder_attentions = decoder_attentions[:i]
        else:
            decoder_attentions = decoder_attentions[:-1] # Ignore attentions on EOS

        return decoder_outputs, decoder_attentions

# Training

input_size = len(input_tokens)
output_size = len(output_tokens)
hidden_size = 100

learning_rate = 1e-4
weight_decay = 1e-6

rarnn = RARNN(input_size, output_size, hidden_size)
optimizer = torch.optim.Adam(rarnn.parameters(), lr=learning_rate, weight_decay=weight_decay)

decoder_criterion = nn.NLLLoss()
attention_criterion = nn.MSELoss(size_average=False)

def train(flat, node):
    context, inputs, targets = data_for_node(flat, node)

    # Turn inputs into tensors
    context_var = tokens_to_tensor([context], output_tokens, False)
    context_var = Variable(context_var)
    inputs_var = tokens_to_tensor(inputs, input_tokens).view(-1, 1, 1) # seq x batch x size
    inputs_var = Variable(inputs_var)
    target_tokens = [target_token for target_token, _ in targets]
    target_ranges = [target_range for _, target_range in targets]
    target_tokens_var = tokens_to_tensor(target_tokens, output_tokens)
    target_tokens_var = Variable(target_tokens_var)
    target_ranges_var = ranges_to_tensor(target_ranges, len(inputs) + 1)
    target_ranges_var = Variable(target_ranges_var)

    # Run through model
    decoder_outputs, attention_outputs = rarnn(context_var, inputs_var, target_tokens_var)

    # Loss calculation and backprop
    optimizer.zero_grad()
    decoder_loss = decoder_criterion(decoder_outputs, target_tokens_var)
    attention_loss = attention_criterion(attention_outputs, target_ranges_var)
    total_loss = decoder_loss + attention_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.data[0]

job = sconce.Job('rarnn')
job.plot_every = 20
job.log_every = 100

n_epochs = 5000

for i in range(n_epochs):
    walked_flat, walked_tree = walk_tree(parsed, parsed['%'], None)
    def _train(node): return train(walked_flat, node)
    ds = descend(walked_tree, _train)
    d = sum(ds) / len(ds)
    job.record(i, d)

# Evaluating

def evaluate(context, inputs, node=None):
    if node == None:
        node = Node('parsed')

    # Turn data into tensors
    context_var = tokens_to_tensor([context], output_tokens, False)
    context_var = Variable(context_var)
    inputs_var = tokens_to_tensor(inputs, input_tokens).view(-1, 1, 1) # seq x batch x size
    inputs_var = Variable(inputs_var)

    # Run through RARNN
    decoder_outputs, attention_outputs = rarnn(context_var, inputs_var)

    # Given the decoder and attention outputs, gather contexts and inputs for sub-phrases
    # Use attention values > 0.5 to select words for next input sequence

    next_contexts = []
    next_inputs = []

    for i in range(len(decoder_outputs)):
        max_value, max_index = decoder_outputs[i].topk(1)
        max_index = max_index.data[0]
        next_contexts.append(output_tokens[max_index]) # Get decoder output token
        a = attention_outputs[i]
        next_input = []
        for t in range(len(a) - 1):
            at = a[t].data[0]
            if at > 0.5:
                next_input.append(inputs[t])
        next_inputs.append(next_input)

    evaluated = list(zip(next_contexts, next_inputs))

    # Print decoded outputs
    print('\n(evaluate) %s %s -> %s' % (context, ' '.join(inputs), next_contexts))

    for context, inputs in evaluated:
        # Add a node for parsed sub-phrases and values
        sub_node = Node(context)
        node.add(sub_node)

        # Recursively evaluate sub-phrases
        if context[0] == '%':
            if len(inputs) > 0:
                evaluate(context, inputs, sub_node)
            else:
                print("WARNING: Empty inputs")

        # Or add words directly to value node
        elif context[0] == '$':
            sub_node.add(' '.join(inputs))

    return node

def evaluate_and_print(context, inputs):
    evaluated = evaluate(context, inputs)
    print(' '.join(inputs))
    print(evaluated)
    return evaluated

walked_flat, walked_tree = walk_tree(parsed, parsed['%'], None)
inputs = [child.key for child in walked_flat.children]
print(inputs, walked_tree)
evaluate_and_print('%', inputs)

evaluate_and_print('%', "hey maia if the ethereum price is less than 2 0 then turn the living room light on".split(' '))
evaluate_and_print('%', "hey maia what's the ethereum price".split(' '))
evaluate_and_print('%', "hey maia play some Skrillex please and then turn the office light off".split(' '))
evaluate_and_print('%', "turn the office light up and also could you please turn off the living room light and make the temperature of the bedroom to 6 thank you maia".split(' '))
evaluate_and_print('%', "turn the living room light off and turn the bedroom light up and also turn the volume up".split(' '))

def node_to_list(node):
    l = []
    for n in node:
        if n.is_leaf:
            l.append(n.key)
        else:
            l += [n.key, node_to_list(n)]
    return l

def parse(s, cb):
    evaluated = evaluate_and_print('%', s.split(' '))
    cb({'evaluated': node_to_list(evaluated)})

parse('hey maia if the price of bitcoin is greater than 3 0 0 0 turn the office light off', lambda r: print("response", r))

torch.save(rarnn, 'rarnn.pt')

service = somata.Service('maia:parser', {'parse': parse}, {'bind_port': 8855})
