from config import *
from network import Network
import data_provider

if __name__ == '__main__':
    net = Network()
    dummy_input = tf.zeros((1, INPUT_HEIGHT, INPUT_WIDTH, 3))
    model = net.model(dummy_input)
    model = tf.identity(model, name="maze-model")
    
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
