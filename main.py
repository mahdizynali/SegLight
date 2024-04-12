from config import *
from inference import *
from network import MazeNet
from data_provider import getData

model = MazeNet()

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return 1 - dice
# loss_function = dice_loss

# loss_function = SparseCategoricalCrossentropy(from_logits=False)
loss_function = CategoricalCrossentropy(from_logits=False)

optimizer = Adam(learning_rate=LEARNING_RATE)
mean_iou = MeanIoU(num_classes=NUMBER_OF_CLASSES)
mean_loss = Mean()

# writer = tf.summary.create_file_writer('logs/graphs')
# writer.set_as_default()
# tf.summary.trace_on(graph=True, profiler=True)

# with writer.as_default():
#     tf.summary.trace_export(
#         name="model_trace",
#         step=0,
#         profiler_outdir='logs/graphs'
#     )

# tf.profiler.experimental.start(logdir='logs/graphs')
# tf.profiler.experimental.stop()


def train_one_epoch(data):

    mean_loss.reset_states()
    mean_iou.reset_states()

    pbar = tqdm(data, desc="Training", unit="batch")

    for images, labels in pbar:
        with tf.GradientTape() as tape:

            predictions = model.call(images, training=True)

            loss = loss_function(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        predictions_argmax = tf.argmax(predictions, axis=-1)

        labels_reshaped = tf.argmax(labels, axis=-1)

        mean_loss.update_state(loss)
        mean_iou.update_state(labels_reshaped, predictions_argmax)

    return mean_loss.result(), mean_iou.result()


def evaluate_one_epoch(data):

    mean_loss.reset_states()
    mean_iou.reset_states()

    pbar = tqdm(data, desc="Evaluation", unit="batch")
    for images, labels in pbar:

        predictions = model(images)

        loss = loss_function(labels, predictions)

        predictions_argmax = tf.argmax(predictions, axis=-1)

        labels_reshaped = tf.argmax(labels, axis=-1)

        mean_loss.update_state(loss)
        mean_iou.update_state(labels_reshaped, predictions_argmax)

    return mean_loss.result(), mean_iou.result()

if __name__ == '__main__':
    
    train_set, test_set = getData()

    # session = tf.compat.v1.Session()

    for epoch in range(EPOCH_NUMBER):

        train_loss, train_iou = train_one_epoch(train_set)
        print(f'Epoch {epoch + 1}/{EPOCH_NUMBER}')
        print(f'Training loss: {train_loss:.4f}, Training mean IoU: {train_iou:.4f}')

        eval_loss, eval_iou = evaluate_one_epoch(test_set)
        print(f'Evaluation loss: {eval_loss:.4f}, Evaluation mean IoU: {eval_iou:.4f}')
        print("\n==================================================================\n")

    model.save("./model", save_format='tf')
    print("\nNew Model has been save !\n")

    loaded_model = tf.keras.models.load_model("./model")

    # inference_on_image(loaded_model, test_set, num_samples=5)
    real_time_inference(loaded_model)