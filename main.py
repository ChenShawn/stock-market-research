import tensorflow as tf
import argparse
import os

from models import burn_in_lstm
from data.dataset import CSVSequentialDataset

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to be used')
parser.add_argument('--label-name', default='label', type=str, help='label name')
parser.add_argument('--csv-dir', default='./data/stocks', type=str, help='csv path')
parser.add_argument('--logdir', default='./tensorboard', type=str, help='tf logs path')
parser.add_argument('--save-dir', default='./train', type=str, help='csv path')
parser.add_argument('--batch-size', default=256, type=int, help='batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--num-epochs', default=40, type=int, help='epoch number')
parser.add_argument('--memory-growth', action="store_true", default=False)
args = parser.parse_args()


def train():
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # build dataset seperately for train and eval
    csv_files = os.listdir(args.csv_dir)
    csv_files = [os.path.join(args.csv_dir, fn) for fn in csv_files]
    dataset_generator = CSVSequentialDataset(csv_files, batch_size=args.batch_size)
    dataset = tf.data.Dataset.from_generator(dataset_generator, 
        output_types=((tf.float32, tf.float32, tf.int32), tf.int32))
    # data_eval = ds.create_dataset_from_file(
    #     csv_path=os.path.join(args.csv_dir, 'single_line_test.csv'), 
    #     batch_size=args.batch_size, num_epochs=1, 
    #     shuffle=False, label_name=args.label_name)

    model = burn_in_lstm.SimpleSequentialLSTM()
    # model = burn_in_lstm.BurnInStateLSTM()
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    adam = tf.keras.optimizers.Adam(lr=args.lr)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='acc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    model.compile(loss=loss, optimizer=adam, metrics=metrics)

    # Can only use `model.save_weights` instead of `model.save`
    # because the latter is not supported for user-defined `tf.keras.Model`
    # NOTE: Uncomment the following lines if training on pretrained model is needed
    if '{}.h5'.format(args.label_name) in os.listdir(args.save_dir):
        try:
            model.load(os.path.join(args.save_dir, args.label_name + '.h5'))
        except:
            model.load_weights(os.path.join(args.save_dir, args.label_name + '.h5'))
        print(' [*] Loaded pretrained model {}.h5'.format(args.label_name))

    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=args.logdir)]
    model.fit(dataset, 
        epochs=args.num_epochs, 
        #validation_data=data_eval,
        callbacks=callbacks)
    try:
        model.save(os.path.join(args.save_dir, args.label_name + '.h5'))
    except:
        model.save_weights(os.path.join(args.save_dir, args.label_name + '.h5'))

    # evaluation
    # results = model.evaluate(data_eval)
    # print('\n\n [*] Evaluation: ', results)


def main():
    # Set GPU configuration
    if tf.test.is_built_with_cuda():
        if args.memory_growth:
            gpu_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        if len(args.gpu) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    train()
    print(' [*] Done!!')


if __name__ == '__main__':
    main()
