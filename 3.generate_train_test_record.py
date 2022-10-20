import generate_tfrecord

test = generate_tfrecord.generate_tfrecord('csv/test.csv', 'records/test.record')
test.generate()

train = generate_tfrecord.generate_tfrecord('csv/train.csv', 'records/train.record')
train.generate()