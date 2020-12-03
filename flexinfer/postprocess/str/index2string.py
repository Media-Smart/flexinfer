from flexinfer.misc import registry
from ..base import Base


@registry.register_module('postprocess')
class IndexToString(Base):
    """
    Current is designed for CTC decoder.
    TODO: Attn-based, FC-based.
    """

    def __init__(self, character, **kwargs):
        super(IndexToString, self).__init__(**kwargs)
        self.character = ['[blank]'] + list(character)

    def postprocess(self, results):
        data = results[self.key]

        texts = []
        batch_size = data.shape[0]
        length = data.shape[1]
        for i in range(batch_size):
            t = data[i]
            char_list = []
            for idx in range(length):
                if t[idx] != 0 and (not (idx > 0 and t[idx - 1] == t[idx])):
                    char_list.append(self.character[t[idx]])
            text = ''.join(char_list)
            texts.append(text)

        results[self.out] = texts
