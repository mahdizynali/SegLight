from config import *
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


# loss_function = SparseCategoricalCrossentropy(from_logits=False)
# loss_function = CategoricalCrossentropy(from_logits=False)
loss_function = dice_loss
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

            predictions = model.call(images)

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


def display_predictions_opencv(model, test_dataset, num_samples=5):
    """Display predictions on some test samples using OpenCV."""
    for i, (image, label) in enumerate(test_dataset.take(num_samples)):
        # Perform prediction using the trained model
        prediction = model(image)
        
        # Convert prediction and label tensors to numpy arrays
        prediction_np = prediction.numpy()
        label_np = label.numpy()
        
        # Get the predicted classes using argmax (assuming one-hot encoding)
        predicted_classes = np.argmax(prediction_np, axis=-1)
        true_classes = np.argmax(label_np, axis=-1)
        
        # Convert image from TensorFlow tensor to numpy array and scale it
        image_np = (image[0].numpy() * 255).astype(np.uint8)
        
        # Create color lookup table for labels
        color_lookup_bgr = np.zeros((len(COLOR_MAP), 3), dtype=np.uint8)
        for idx, (class_name, color) in enumerate(COLOR_MAP.items()):
            color_bgr = [color[2], color[1], color[0]]
            color_lookup_bgr[idx] = np.array(color_bgr, dtype=np.uint8)
        
        # Color the true and predicted classes using the color lookup table
        true_colored = color_lookup_bgr[true_classes[0]]
        predicted_colored = color_lookup_bgr[predicted_classes[0]]
        
        # Display the images using OpenCV
        cv2.imshow(f'Original Image {i + 1}', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        cv2.imshow(f'True Label {i + 1}', true_colored)
        cv2.imshow(f'Predicted Label {i + 1}', predicted_colored)
        
        # Wait for a key press to move to the next sample
        cv2.waitKey(0)
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()

def save_model(model, path):
    print("here \n\n")
    os.makedirs(path, exist_ok=True)

    tf.saved_model.save(model, path)

if __name__ == '__main__':
    
    train_set, test_set = getData()

    for epoch in range(EPOCH_NUMBER):

        train_loss, train_iou = train_one_epoch(train_set)
        print(f'Epoch {epoch + 1}/{EPOCH_NUMBER}')
        print(f'Training loss: {train_loss:.4f}, Training mean IoU: {train_iou:.4f}')

        eval_loss, eval_iou = evaluate_one_epoch(test_set)
        print(f'Evaluation loss: {eval_loss:.4f}, Evaluation mean IoU: {eval_iou:.4f}')
        print("\n==================================================================\n")

    # save_model(model, "./")
    # display_predictions_opencv(model, test_set, num_samples=5)
