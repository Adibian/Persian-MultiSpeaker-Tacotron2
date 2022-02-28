from synthesizer.persian_utils.symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, cleaner_names):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

      The text can optionally have ARPAbet sequences enclosed in curly braces embedded
      in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

      Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through

      Returns:
        List of integers corresponding to the symbols in the text
    """
    # print("######")
    # print(cleaner_names)
    if cleaner_names != ['persian_cleaners']:
        return 'cleaner is not persian!'
    sequence = []
    for phoneme in text.split():
        sequence.append(_symbol_to_id[phoneme])
    # print(sequence)
    # print("************")
    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = []
    for symbol_id in sequence:
        result.append(_id_to_symbol[symbol_id])
    ' '.join(result)
    return result
