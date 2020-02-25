from random import randint, seed
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from text_generation_with_spacy import text_seq


def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):
    output_text = []
    input_text = seed_text

    for i in range(num_gen_words):
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')

        pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
        pred_word = tokenizer.index_word[pred_word_ind]

        input_text += ' ' + pred_word
        output_text.append(pred_word)

        return ' '.join(output_text)


# print()
seed(101)
random_pick = randint(0, len(text_seq))
random_seed_text = text_seq[random_pick]
seed_text = ' '.join(random_seed_text)
# print(seed_text)
print('-------------------')
# print(generate_text(model, tokenizer, seq_length, seed_text=seed_text, num_gen_words=10))

# generate_text(model, tokenizer, seq_length, seed_text=seed_text, num_gen_words=50)

# full_text = read_file('moby_dick_four_chapters.txt.txt')

# txtx = "moby_dick_four_chapters.txt"
# full_text = open(txtx, 'r', encoding='utf-8').read()
#
# for i, word in enumerate(full_text.split()):
#     if word == 'inkling':
#         print(' '.join(full_text.split()[i - 20:i + 20]))
#         print('\n')

model = load_model('epochBIG.h5')
tokenizer = load(open('epochBIG', 'rb'))

print( generate_text(model, tokenizer, seq_length, seed_text, num_gen_words=20) )