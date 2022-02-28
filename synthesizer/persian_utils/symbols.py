"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
"""
_pad        = "_"
_eos        = "~"
_characters = ['U', 'Q', 'G', 'AA', 'V', 'N', 'CH', 'R', 'KH', 'B', 'Z', 'SH', 'O', 'A', 'E', 'ZH', 'H', 'SIL', 'AH', 'S', 'D', 'J', 'L', 'F', 'K', 'I', 'T', 'P', 'M', 'Y']
_other_symbols = ['!', '(', ')', '-', '.', ':', ';', '?']

# Export all symbols:
symbols = [_pad, _eos] + _characters + _other_symbols
