import numpy as np
import utils.createDataAndPlot as cp

NETWORK_SHAPE = [2, 4, 4, 2]
BATCH_SIZE = 32
PATTERN = "circle"  # square, circle, triangle
BACKPROPAGATION = "adam"  # mbsgd, adam, rmsprop, adagrad

# ----------------------------------
def activation_ReLU(inputs):
    return np.maximum(0, inputs)


def activation_sigmoid(inputs):
    return 1 / (1 + np.exp(-inputs))


def activation_softmax(inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


# ----------------------------------
def classify(probabilities):
    classification = np.rint(probabilities[:, 1])
    return classification


def normalize(inputs):
    max_number = np.max(np.absolute(inputs), axis=1, keepdims=True)
    scale_rate = np.where(max_number == 0, 1, 1 / max_number)
    return inputs * scale_rate


# ----------------------------------
def create_weights(n_inputs, n_neurons):
    return np.random.randn(n_inputs, n_neurons)


def create_biases(n_neurons):
    return np.random.randn(n_neurons)


# ----------------------------------
def precise_loss_function(predicted, real):
    epsilon = 1e-12
    predicted = np.clip(predicted, epsilon, 1.0 - epsilon)
    ce_loss = -np.sum(
        real * np.log(predicted) + (1 - real) * np.log(1 - predicted), axis=1
    )
    return np.mean(ce_loss)


def get_final_layer_preAct_demands(predicted_values, target_vector):
    return predicted_values - target_vector


# ----------------------------------
class Layer:
    def __init__(
        self,
        n_inputs,
        n_neurons,
        activation,
    ):
        self.weights = create_weights(n_inputs, n_neurons)
        self.biases = create_biases(n_neurons)
        self.activation = activation

        # 初始化用于不同优化器的参数
        self.v = np.zeros_like(self.weights)  # RMSprop ,Adagrad
        self.m = np.zeros_like(self.weights)  # Adam
        self.t = 1  # Adam

    def layer_forward(self, inputs):
        self.sum = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation(self.sum)
        return self.output

    def get_weight_adjust_matrix(self, preWeights_values, afterWeights_demands):
        weights_adjust_matrix = (
            np.dot(preWeights_values.T, afterWeights_demands)
            / preWeights_values.shape[0]
        )
        return weights_adjust_matrix

    def layer_backward_mbsgd(
        self, preWeights_values, afterWeights_demands, learning_rate
    ):
        weights_adjust_matrix = self.get_weight_adjust_matrix(
            preWeights_values, afterWeights_demands
        )
        self.weights -= learning_rate * weights_adjust_matrix
        self.biases -= learning_rate * np.mean(afterWeights_demands, axis=0)

    def layer_backward_adam(
        self,
        preWeights_values,
        afterWeights_demands,
        learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        weights_adjust_matrix = self.get_weight_adjust_matrix(
            preWeights_values, afterWeights_demands
        )

        self.m = beta1 * self.m + (1 - beta1) * weights_adjust_matrix
        self.v = beta2 * self.v + (1 - beta2) * weights_adjust_matrix**2

        # 进行偏差修正
        m_hat = self.m / (1 - beta1**self.t)
        v_hat = self.v / (1 - beta2**self.t)

        self.weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        self.biases -= learning_rate * np.mean(afterWeights_demands, axis=0)

        self.t += 1  # 训练步长增加

    def layer_backward_rmsprop(
        self,
        preWeights_values,
        afterWeights_demands,
        learning_rate,
        beta1=0.9,
        epsilon=1e-8,
    ):
        weights_adjust_matrix = self.get_weight_adjust_matrix(
            preWeights_values, afterWeights_demands
        )
        self.v = beta1 * self.v + (1 - beta1) * weights_adjust_matrix**2
        self.weights -= (
            learning_rate * weights_adjust_matrix / (np.sqrt(self.v) + epsilon)
        )
        self.biases -= learning_rate * np.mean(afterWeights_demands, axis=0)

    def layer_backward_adagrad(
        self, preWeights_values, afterWeights_demands, learning_rate, epsilon=1e-8
    ):
        weights_adjust_matrix = self.get_weight_adjust_matrix(
            preWeights_values, afterWeights_demands
        )
        self.v += weights_adjust_matrix**2
        self.weights -= (
            learning_rate * weights_adjust_matrix / (np.sqrt(self.v) + epsilon)
        )
        self.biases -= learning_rate * np.mean(afterWeights_demands, axis=0)

    backpropagation_dict = {
        "mbsgd": layer_backward_mbsgd,
        "adam": layer_backward_adam,
        "rmsprop": layer_backward_rmsprop,
        "adagrad": layer_backward_adagrad,
    }

    def layer_backward(self, preWeights_values, afterWeights_demands, learning_rate):
        Layer.backpropagation_dict[BACKPROPAGATION](
            self, preWeights_values, afterWeights_demands, learning_rate
        )


# ---------------------  -------------
class Network:
    def __init__(self, network_shape, activation):
        self.shape = network_shape
        self.layers = []
        final_activation = activation
        for i in range(len(network_shape) - 1):
            if i == len(network_shape) - 2:
                activation = final_activation
            else:
                activation = activation_ReLU
            layer = Layer(network_shape[i], network_shape[i + 1], activation)
            self.layers.append(layer)

    def network_forward(self, inputs):
        outputs = [inputs]
        for layer in self.layers:
            layer_output = layer.layer_forward(inputs)
            outputs.append(layer_output)
            inputs = layer_output
        return outputs

    def network_backward(self, outputs, targets, learning_rate):
        demands = get_final_layer_preAct_demands(outputs[-1], targets)
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            preWeights_values = outputs[i]
            layer.layer_backward(preWeights_values, demands, learning_rate)
            if i > 0:
                demands = np.dot(demands, layer.weights.T) * (outputs[i] > 0)

    def train(self, inputs, targets, learning_rate, epochs):
        num_samples = inputs.shape[0]
        num_batches = num_samples // BATCH_SIZE

        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            inputs_shuffled = inputs[indices]
            targets_shuffled = targets[indices]

            for batch in range(num_batches):
                start_index = batch * BATCH_SIZE
                end_index = (batch + 1) * BATCH_SIZE
                batch_inputs = inputs_shuffled[start_index:end_index]
                batch_targets = targets_shuffled[start_index:end_index]

                outputs = self.network_forward(batch_inputs)
                self.network_backward(outputs, batch_targets, learning_rate)

            if epoch % 100 == 0:
                # 计算整个数据集的损失
                full_outputs = self.network_forward(inputs)
                loss = precise_loss_function(full_outputs[-1], targets)
                print(f"Epoch {epoch}, Loss: {loss}")


# ----------------------------------
def test():
    data = cp.create_data(2048, PATTERN)
    data1 = np.copy(data)
    inputs = data[:, (0, 1)]
    targets = np.column_stack((1 - data[:, 2], data[:, 2]))
    network = Network(NETWORK_SHAPE, activation_softmax)
    network.train(inputs, targets, learning_rate=0.01, epochs=1000)
    outputs = network.network_forward(inputs)
    classification = classify(outputs[-1])
    data[:, 2] = classification
    data2 = np.copy(data)
    print("Classification:", classification)
    print("Targets:", targets[:, 1])
    print("Loss:", precise_loss_function(outputs[-1], targets))
    cp.plot_data(data1, data2, "training")


if __name__ == "__main__":
    test()
