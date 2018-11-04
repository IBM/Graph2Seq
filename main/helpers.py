import numpy as np

def batch(inputs, max_sequence_length=None):
    """
        Args:
            inputs:
                list of sentences (integer lists)
            max_sequence_length:
                integer specifying how large should `max_time` dimension be.
                If None, maximum sequence length would be used

        Outputs:
            inputs_time_major:
                input sentences transformed into time-major matrix
                (shape [max_time, batch_size]) padded with 0s
            sequence_lengths:
                batch-sized list of integers specifying amount of active
                time steps in each input sequence
    """
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    loss_weights = []
    for _ in range(len(inputs)):
        weights = []
        for __ in range(len(inputs[_])+1):
            weights.append(1)
        for __ in range(max_sequence_length-len(inputs[_])):
            weights.append(0)
        loss_weights.append(weights)

    return inputs_batch_major, sequence_lengths, loss_weights