from config import *

def inference_on_image(model, test_dataset, num_samples=10):
    
    for i, (image, label) in enumerate(test_dataset.take(num_samples)):

        prediction = model(image)
        
        prediction_np = prediction.numpy()
        label_np = label.numpy()
        
        predicted_classes = np.argmax(prediction_np, axis=-1)
        true_classes = np.argmax(label_np, axis=-1)
        
        image_np = (image[0].numpy() * 255).astype(np.uint8)
        
        color_lookup_bgr = np.zeros((len(COLOR_MAP), 3), dtype=np.uint8)
        for idx, (class_name, color) in enumerate(COLOR_MAP.items()):
            color_bgr = [color[2], color[1], color[0]]
            color_lookup_bgr[idx] = np.array(color_bgr, dtype=np.uint8)
        
        true_colored = color_lookup_bgr[true_classes[0]]
        predicted_colored = color_lookup_bgr[predicted_classes[0]]
        
        cv2.imshow(f'Original Image {i + 1}', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        cv2.imshow(f'True Label {i + 1}', true_colored)
        cv2.imshow(f'Predicted Label {i + 1}', predicted_colored)
        
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()

def real_time_inference(model):

    photo = cv2.imread("/home/mahdi/Desktop/hslSegment/SegLight/dataset/images/262.png")
    lb = cv2.imread("/home/mahdi/Desktop/hslSegment/SegLight/dataset/labels/262.png")

    cap = cv2.VideoCapture(0) 

    COLOR_MAP2 = {
        0: (0, 0, 255),    # Ball: Blue (BGR format)
        1: (0, 255, 0),    # Field: Green (BGR format)
        2: (255, 255, 255),    # Line: White (BGR format)
        3: (0, 0, 0) # Background: Black (BGR format)
    }

    color_lookup_bgr = np.zeros((len(COLOR_MAP2), 3), dtype=np.uint8)
    for idx, (class_name, color) in enumerate(COLOR_MAP2.items()):
        # Convert RGB color to BGR for OpenCV
        color_bgr = [color[2], color[1], color[0]]
        color_lookup_bgr[idx] = np.array(color_bgr, dtype=np.uint8)
    
    while True:
        # ret, frame = cap.read()
        
        # if not ret:
        #     break
        
        frame = photo
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_rgb = cv2.resize(frame_rgb, (320,240), interpolation=cv2.INTER_LINEAR)
        
        frame_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.float32)
        frame_tensor = tf.expand_dims(frame_tensor, axis=0)
        
        frame_tensor = frame_tensor / 255.0
        
        prediction = model.call(frame_tensor,training=False)
        
        prediction_np = prediction[0]
        
        predicted_classes = np.argmax(prediction_np, axis=-1)
        
        predicted_colored = color_lookup_bgr[predicted_classes]
        
        # predicted_colored_bgr = cv2.cvtColor(predicted_colored, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('real', frame)
        cv2.imshow('pred', predicted_colored)
        cv2.imshow('true', lb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

def load_model(path):
    model = tf.keras.models.load_model(path)
    return model