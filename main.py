from config import *
from network import Network
from data_provider import getData

net = Network()
model = net.model(tf.zeros((1, INPUT_HEIGHT, INPUT_WIDTH, 3)))
model = tf.identity(model, name="maze-model")

loss_function = SparseCategoricalCrossentropy(from_logits=True)
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
    # Reset metrics
    mean_loss.reset_states()
    mean_iou.reset_states()

    # Loop through the dataset
    for images, labels in data:
        with tf.GradientTape() as tape:

            predictions = model(images, training=True)
            
            # Compute loss
            loss = loss_function(labels, predictions)

        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)

        # Update model weights
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Update metrics
        mean_loss.update_state(loss)
        mean_iou.update_state(labels, tf.argmax(predictions, axis=-1))

    # Return mean loss and mean IoU
    return mean_loss.result(), mean_iou.result()

# Function to evaluate one epoch
def evaluate_one_epoch(data):
    # Reset metrics
    mean_loss.reset_states()
    mean_iou.reset_states()

    # Loop through the dataset
    for images, labels in data:
        # Forward pass
        predictions = model(images)

        # Compute loss
        loss = loss_function(labels, predictions)

        # Update metrics
        mean_loss.update_state(loss)
        mean_iou.update_state(labels, tf.argmax(predictions, axis=-1))

    # Return mean loss and mean IoU
    return mean_loss.result(), mean_iou.result()



if __name__ == '__main__':
    
    train_set, test_set = getData()

    for epoch in range(EPOCH_NUMBER):
        print(f'Epoch {epoch + 1}/{EPOCH_NUMBER}')

        train_loss, train_iou = train_one_epoch(train_set)
        print(f'Training loss: {train_loss:.4f}, Training mean IoU: {train_iou:.4f}')

        eval_loss, eval_iou = evaluate_one_epoch(test_set)
        print(f'Evaluation loss: {eval_loss:.4f}, Evaluation mean IoU: {eval_iou:.4f}')

    model.save('my_model')
