# 0.31 F1 score, 0.78 Accuracy
import csv
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf
import tensorflow_hub as hub

lemmatizer = WordNetLemmatizer()
english_stopwords = stopwords.words('english')


def clean_sentence(sentence):
    sentence = sentence.lower()
    for char in sentence:
        if (not char.isalpha()) and char != ' ':
            sentence = sentence.replace(char, ' ')
    tokens = sentence.split()
    good_tokens = []
    for token in tokens:
        if token not in english_stopwords:
            good_tokens.append(lemmatizer.lemmatize(token))
    sentence = ' '.join(good_tokens)
    return sentence


def read_file_rows(file_path, delimiter=","):
    rows = []
    tsv_file = open(file_path)
    tsv_reader = csv.reader(tsv_file, delimiter=delimiter)

    for row in tsv_reader:
        rows.append(row)
    return rows


def build_examples():
    pcl_rows = read_file_rows('./train_subset.csv')[1:]
    pcl_rows_test = read_file_rows('./validation_subset.csv')[1:]
    categories_rows = read_file_rows('./dontpatronizeme_categories.tsv', '\t')[5:]

    positive_examples = []
    test_positive_examples = []
    negative_examples = []
    test_data = []
    test_labels = []

    for curr_row in pcl_rows:
        if int(curr_row[-1]) == 0:
            negative_examples.append(clean_sentence(curr_row[-2]))
        else:
            for category_row in categories_rows:
                if curr_row[0] == category_row[0]:
                    positive_examples.append(clean_sentence(category_row[-3]))

    for curr_row in pcl_rows_test[1:]:
        test_data.append(clean_sentence(curr_row[-2]))
        test_labels.append(int(curr_row[-1]))
        if int(curr_row[-1]) == 0:
            pass
        else:
            for category_row in categories_rows:
                if curr_row[0] == category_row[0]:
                    test_positive_examples.append(clean_sentence(category_row[-3]))

    # ['disabled', 'immigrant', 'homeless', 'poor-families', 'in-need', 'migrant', 'women', 'vulnerable', 'refugee', 'hopeless']

    return positive_examples, negative_examples, test_positive_examples, test_data, test_labels


(positive_examples, negative_examples, test_positive_examples, test_data, test_labels) = build_examples()

negative_examples = negative_examples[0: len(positive_examples)]

avg_pos_len = 0
max_len = 0

for positive_example in positive_examples:
    avg_pos_len += len(positive_example)
    max_len = max(max_len, len(positive_example))

avg_pos_len = int(avg_pos_len / len(positive_examples))


train_examples = []
train_labels = []
for negative_example in negative_examples:
    train_examples.append(negative_example)
    train_labels.append(0)


for positive_example in positive_examples:
    train_examples.append(positive_example)
    train_labels.append(1)

def get_model():
    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    VOCAB_SIZE = 1000
    encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, output_mode="int")
    encoder.adapt(train_dataset.map(lambda text, label: text))
    hub_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", input_shape=[],
                               output_shape=[512, 16],
                               dtype=tf.string, trainable=True)
    model = tf.keras.models.Sequential([
        hub_layer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2)
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

train_examples = np.array(train_examples)
train_labels = np.array(train_labels).astype('float32')

model = get_model()
model.fit(train_examples, train_labels, batch_size=64, epochs=10)
predicted_labels = model.predict(test_positive_examples)
predicted_labels = np.array([predicted_label.argmax() for predicted_label in predicted_labels])
good = 0.0
total = 0.0
for predicted_label in predicted_labels:
    good += predicted_label
    total += 1.0
print(f'Shards accuracy {good / total}')


predicted_labels = []
window = avg_pos_len * 2
for q in range(0, len(test_data)):
    print(f"Example {q}")
    test_example = test_data[q]
    if len(test_example) < 10:
        predicted_labels.append(0)
        continue
    predicted_label = model.predict([test_example])[0].argmax()
    for i in range(0, len(test_example) - window, window):
        test_example_shard = test_example[i:(i + window)]
        shard_label = model.predict([test_example_shard])[0].argmax()
        if shard_label == 1:
            predicted_label = 1
            break
    predicted_labels.append(int(predicted_label))
print(f'F1 score for validation is {f1_score(test_labels, predicted_labels)}')
print(f'Accuracy score for validation is {accuracy_score(test_labels, predicted_labels)}')
