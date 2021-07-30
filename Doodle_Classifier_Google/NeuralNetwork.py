# we would be having four hidden layers,
# first would have X number of perceptrons, second 2X,
# third 2X, fourth X, this excludes the input and output layer;
import HelperFunctions as HF
import numpy as np
from matplotlib import pyplot as plt


class NeuralNetwork():
    def __init__(self, number_of_perceptrons, test_X, test_Y, train_X, train_Y,
                 dims, M, M_test):
        self.number_of_perceptrons = number_of_perceptrons
        self.test_X = test_X
        self.test_Y = test_Y
        self.train_X = train_X
        self.train_Y = train_Y
        self.M_test = M_test
        self.dims = dims

        # first layer parameters;
        self.first_layer_weights = HF.random_initialization(number_of_perceptrons,
                                                            dims) * np.sqrt(1/(dims))

        self.first_layer_bias = HF.zeros(1,
                                         number_of_perceptrons)

        self.first_layer_v_w = HF.zeros(number_of_perceptrons,
                                        dims)
        self.first_layer_s_w = HF.zeros(number_of_perceptrons,
                                        dims)
        self.first_layer_v_b = HF.zeros(1,
                                        number_of_perceptrons)
        self.first_layer_s_b = HF.zeros(1,
                                        number_of_perceptrons)

        # second layer parameters;
        self.second_layer_weights = HF.random_initialization(2 * number_of_perceptrons,
                                                             number_of_perceptrons) * np.sqrt(1/(number_of_perceptrons))

        self.second_layer_bias = HF.zeros(1,
                                          2*number_of_perceptrons)

        self.second_layer_v_w = HF.zeros(2 * number_of_perceptrons,
                                         number_of_perceptrons)
        self.second_layer_s_w = HF.zeros(2 * number_of_perceptrons,
                                         number_of_perceptrons)

        self.second_layer_v_b = HF.zeros(1,
                                         2*number_of_perceptrons)
        self.second_layer_s_b = HF.zeros(1,
                                         2*number_of_perceptrons)

        # third layer parameters;
        self.third_layer_weights = HF.random_initialization(2*number_of_perceptrons,
                                                            2*number_of_perceptrons) * np.sqrt(1/(2*number_of_perceptrons))
        self.third_layer_bias = HF.zeros(1,
                                         2*number_of_perceptrons)

        self.third_layer_v_w = HF.zeros(2*number_of_perceptrons,
                                        2*number_of_perceptrons)
        self.third_layer_s_w = HF.zeros(2*number_of_perceptrons,
                                        2*number_of_perceptrons)

        self.third_layer_v_b = HF.zeros(1,
                                        2*number_of_perceptrons)
        self.third_layer_s_b = HF.zeros(1,
                                        2*number_of_perceptrons)

        # fourth layer parameters;
        self.fourth_layer_weights = HF.random_initialization(
            number_of_perceptrons, 2*number_of_perceptrons) * np.sqrt(1/(2*number_of_perceptrons))

        self.fourth_layer_bias = HF.zeros(1,
                                          number_of_perceptrons)

        self.fourth_layer_v_w = HF.zeros(
            number_of_perceptrons, 2*number_of_perceptrons)
        self.fourth_layer_s_w = HF.zeros(
            number_of_perceptrons, 2*number_of_perceptrons)

        self.fourth_layer_v_b = HF.zeros(1,
                                         number_of_perceptrons)

        self.fourth_layer_s_b = HF.zeros(1,
                                         number_of_perceptrons)
        # output layer parameters;
        self.output_layer = HF.random_initialization(
            4, number_of_perceptrons) * np.sqrt(2/(number_of_perceptrons))

        self.cost_values = []

        self.M = M

    def forward_prop_train(self):
        Z1 = np.matmul(self.train_X, self.first_layer_weights.T) + \
            self.first_layer_bias
        A1 = np.tanh(Z1)

        Z2 = np.matmul(A1, self.second_layer_weights.T) + \
            self.second_layer_bias
        A2 = np.tanh(Z2)

        Z3 = np.matmul(A2, self.third_layer_weights.T) + self.third_layer_bias
        A3 = np.tanh(Z3)

        Z4 = np.matmul(A3, self.fourth_layer_weights.T) + \
            self.fourth_layer_bias
        A4 = np.tanh(Z4)

        Z_output = np.matmul(A4, self.output_layer.T)
        A_output = HF.sigmoid(Z_output)

        values = {
            'A1': A1,
            'A2': A2,
            'A3': A3,
            'A4': A4,
            'A_output': A_output
        }

        return values

    def forward_prop_test(self):
        Z1 = np.matmul(self.test_X, self.first_layer_weights.T) + \
            self.first_layer_bias
        A1 = np.tanh(Z1)

        Z2 = np.matmul(A1, self.second_layer_weights.T) + \
            self.second_layer_bias
        A2 = np.tanh(Z2)

        Z3 = np.matmul(A2, self.third_layer_weights.T) + self.third_layer_bias
        A3 = np.tanh(Z3)

        Z4 = np.matmul(A3, self.fourth_layer_weights.T) + \
            self.fourth_layer_bias
        A4 = np.tanh(Z4)

        Z_output = np.matmul(A4, self.output_layer.T)
        temp = HF.sigmoid(Z_output)
        print(temp)
        A_output = np.argmax(temp, axis=1)

        A_output = A_output.reshape(A_output.shape[0], 1)
        return A_output

    def plot_graph(self):

        X_values = np.arange(len(self.cost_values))
        Y_values = self.cost_values
        plt.plot(X_values, Y_values)
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.show()
        return

    def test(self):
        result = self.forward_prop_test()
        return result

    def train(self, number_of_iterations=2000):
        for _ in range(number_of_iterations):
            values = self.forward_prop_train()
            self.backward_prop_gradient_descent(values, iteration=_+1)
        self.plot_graph()
        self.cost_values = []

    def train_adam(self, number_of_iterations=2000):
        for _ in range(number_of_iterations):
            values = self.forward_prop_train()
            self.backward_prop_adam_optimizer(values, iteration=_+1)
        self.plot_graph()
        self.cost_values = []

    def calculate_cost(self, value, iteration):
        total_cost = np.squeeze(np.sum(np.power(value, 2)))
        print(f"Iteration : {iteration} | {total_cost}")
        self.cost_values.append(total_cost)

    def backward_prop_adam_optimizer(self, values, iteration, beta1=0.9, beta2=0.999, epsilon=1e-8, gamma=0.001):
        A1 = values['A1']
        A2 = values['A2']
        A3 = values['A3']
        A4 = values['A4']
        A_output = values['A_output']

        assert(A_output.shape == self.train_Y.shape)

        output_layer_error = A_output - self.train_Y
        self.calculate_cost(output_layer_error, iteration)
        # doing calculations for the output layer;
        dW_output = np.matmul(output_layer_error.T, A4)

        assert(dW_output.shape == (4, self.number_of_perceptrons))
        self.output_layer = self.output_layer + \
            (1/self.M) * gamma * dW_output

        # doing calculations for the fourth layer;
        error__fourth_layer = np.matmul(
            output_layer_error, self.output_layer) * HF.get_tanh_derivative(A4)
        dW_fourth_layer = np.matmul(error__fourth_layer.T, A3)
        assert(dW_fourth_layer.shape == self.fourth_layer_weights.shape)

        self.fourth_layer_v_w = beta1 * \
            self.fourth_layer_v_w - (1-beta1) * dW_fourth_layer
        self.fourth_layer_s_w = beta2 * self.fourth_layer_s_w - \
            (1-beta2) * np.power(dW_fourth_layer, 2)

        temp = np.sum(error__fourth_layer, axis=0, keepdims=True)

        self.fourth_layer_v_b = beta1 * self.fourth_layer_v_b - \
            (1-beta2) * temp
        self.fourth_layer_s_b = beta2 * self.fourth_layer_s_b - \
            (1-beta2) * np.power(temp, 2)

        print(self.fourth_layer_s_w)

        self.fourth_layer_weights = self.fourth_layer_weights - \
            (gamma) * (self.fourth_layer_v_w /
                       (np.sqrt(self.fourth_layer_s_w) + epsilon))
        self.fourth_layer_bias = self.fourth_layer_bias - \
            (gamma) * (self.fourth_layer_v_b /
                       (np.sqrt(self.fourth_layer_s_b) + epsilon))

        # doing calculations for the third layer;
        error_third_layer = np.matmul(
            error__fourth_layer, self.fourth_layer_weights
        ) * HF.get_tanh_derivative(A3)
        dW_third_layer = np.matmul(error_third_layer.T, A2)
        assert(dW_third_layer.shape == self.third_layer_weights.shape)

        self.third_layer_v_w = beta1 * \
            self.third_layer_v_w - (1-beta1) * dW_third_layer
        self.third_layer_s_w = beta2 * self.third_layer_s_w - \
            (1-beta2) * np.power(dW_third_layer, 2)

        temp = np.sum(error_third_layer, axis=0, keepdims=True)
        self.third_layer_v_b = beta1 * self.third_layer_v_b - \
            (1-beta2) * temp
        self.third_layer_s_b = beta2 * self.third_layer_s_b - \
            (1-beta2) * np.power(temp, 2)


        self.third_layer_weights = self.third_layer_weights - \
            (gamma) * (self.third_layer_v_w /
                       (np.sqrt(self.third_layer_s_w) + epsilon))


        self.third_layer_bias = self.third_layer_bias - \
            (gamma) * (self.third_layer_v_b /
                       (np.sqrt(self.third_layer_s_b) + epsilon))

        # doing calculations for the second layer;
        error_second_layer = np.matmul(
            error_third_layer, self.third_layer_weights) * HF.get_tanh_derivative(A2)
        dW_second_layer = np.matmul(error_second_layer.T, A1)
        assert(dW_second_layer.shape == self.second_layer_weights.shape)

        self.second_layer_v_w = beta1 * \
            self.second_layer_v_w - (1-beta1) * dW_second_layer
        self.second_layer_s_w = beta2 * self.second_layer_s_w - \
            (1-beta2) * np.power(dW_second_layer, 2)


        temp = np.sum(error_second_layer, axis=0, keepdims=True)
        self.second_layer_v_b = beta1 * self.second_layer_v_b - \
            (1-beta1)*(temp)

        self.second_layer_s_b = beta2 * self.second_layer_s_b - \
            (1-beta2) * (np.power(temp, 2))

        self.second_layer_weights = self.second_layer_weights - \
            (gamma) * (self.second_layer_v_w /
                       (np.sqrt(self.second_layer_s_w) + epsilon))

        self.second_layer_bias = self.second_layer_bias - \
            (gamma) * (self.second_layer_v_b /
                       (np.sqrt(self.second_layer_s_b) + epsilon))

        # doing calculations for the first layer
        error_first_layer = np.matmul(
            error_second_layer, self.second_layer_weights) * HF.get_tanh_derivative(A1)
        dW_first_layer = np.matmul(error_first_layer.T, self.train_X)
        assert(dW_first_layer.shape == self.first_layer_weights.shape)

        self.first_layer_v_w = beta1 * \
            self.first_layer_v_w - (1-beta1) * dW_first_layer
        self.first_layer_s_w = beta2 * self.first_layer_s_w - \
            (1-beta2) * np.power(dW_first_layer, 2)
        temp = np.sum(error_first_layer, axis=0, keepdims=True)
        self.first_layer_v_b = beta1 * self.first_layer_v_b - \
            (1-beta1)*(temp)
        self.first_layer_s_b = beta2 * self.first_layer_s_b - \
            (1-beta2) * (np.power(temp, 2))

        self.first_layer_weights = self.first_layer_weights - \
            (gamma) * (self.first_layer_v_w /
                       (np.sqrt(self.first_layer_s_w) + epsilon))
        self.first_layer_bias = self.first_layer_bias - \
            (gamma) * (self.first_layer_v_b /
                       (np.sqrt(self.first_layer_s_b) + epsilon))

    def backward_prop_gradient_descent(self, values, iteration, learning_rate=0.003):
        A1 = values['A1']
        A2 = values['A2']
        A3 = values['A3']
        A4 = values['A4']
        A_output = values['A_output']

        assert(A_output.shape == self.train_Y.shape)

        output_layer_error = A_output - self.train_Y
        self.calculate_cost(output_layer_error, iteration)
        # doing calculations for the output layer;
        dW_output = np.matmul(output_layer_error.T, A4)

        assert(dW_output.shape == (4, self.number_of_perceptrons))

        self.output_layer = self.output_layer + \
            (1/self.M) * learning_rate * dW_output

        # doing calculations for the fourth layer;
        error__fourth_layer = np.matmul(
            output_layer_error, self.output_layer) * HF.get_tanh_derivative(A4)
        dW_fourth_layer = np.matmul(error__fourth_layer.T, A3)
        assert(dW_fourth_layer.shape == self.fourth_layer_weights.shape)

        self.fourth_layer_weights = self.fourth_layer_weights - \
            (1/self.M) * learning_rate * dW_fourth_layer

        self.fourth_layer_bias = self.fourth_layer_bias - \
            (1/self.M) * learning_rate * \
            np.sum(error__fourth_layer, axis=0, keepdims=True)

        # doing calculations for the third layer;
        error_third_layer = np.matmul(
            error__fourth_layer, self.fourth_layer_weights
        ) * HF.get_tanh_derivative(A3)
        dW_third_layer = np.matmul(error_third_layer.T, A2)
        assert(dW_third_layer.shape == self.third_layer_weights.shape)

        self.third_layer_weights = self.third_layer_weights - \
            (1/self.M) * learning_rate * dW_third_layer

        self.third_layer_bias = self.third_layer_bias - \
            (1/self.M) * learning_rate * \
            np.sum(error_third_layer, axis=0, keepdims=True)

        # doing calculations for the second layer;
        error_second_layer = np.matmul(
            error_third_layer, self.third_layer_weights) * HF.get_tanh_derivative(A2)
        dW_second_layer = np.matmul(error_second_layer.T, A1)
        assert(dW_second_layer.shape == self.second_layer_weights.shape)

        self.second_layer_weights = self.second_layer_weights - \
            (1/self.M) * learning_rate * dW_second_layer
        self.second_layer_bias = self.second_layer_bias - \
            (1/self.M) * learning_rate * \
            np.sum(error_second_layer, axis=0, keepdims=True)

        # doing calculations for the first layer
        error_first_layer = np.matmul(
            error_second_layer, self.second_layer_weights) * HF.get_tanh_derivative(A1)
        dW_first_layer = np.matmul(error_first_layer.T, self.train_X)
        assert(dW_first_layer.shape == self.first_layer_weights.shape)

        self.first_layer_weights = self.first_layer_weights - \
            (1/self.M) * learning_rate * dW_first_layer
        self.first_layer_bias = self.first_layer_bias - \
            (1/self.M) * learning_rate * \
            np.sum(error_first_layer, axis=0, keepdims=True)
