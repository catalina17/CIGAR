import ast
import numpy


if __name__ == '__main__':

    num_genres = 6
    num_test_examples = 60
    num_runs = 28
    num_genres_str = 'six'

    average_confusion_matrix = numpy.zeros((num_genres, num_genres))
    mean_error = 0.0
    z_value = 1.96

    for i in range(num_runs):
        # Read the results for the i-th run
        f = open('eval_runs/' + str(num_genres) + 'genres/' + num_genres_str +
                 '_class_run' + str(i + 1) + '.txt')
        text_data = f.read()
        f.close()

        # Parse the test error
        test_data = text_data[text_data.find('\'test\''): text_data.find(',')]
        error = float(test_data[test_data.find(' ') + 1:])
        mean_error += error

        if num_genres > 2:
            # Parse the confusion matrix
            text_confusion_matrix = text_data[text_data.find('array(') + 6:
                                              text_data.find('), \'train_loss\'')]
            confusion_matrix = numpy.array(ast.literal_eval(text_confusion_matrix))
            average_confusion_matrix += confusion_matrix

    if num_genres > 2:
        # Compute average confusion matrix
        average_confusion_matrix /= num_runs
    # Estimate the classification error (unbiased estimator)
    mean_error /= num_runs
    # As num_test_examples > 30, we can approximate the Binomial distribution
    # with the Normal distribution; we can therefore derive a 95% CI for the error
    half_range = z_value * numpy.sqrt(mean_error * (1.0 - mean_error) /
                                      (num_test_examples * num_runs))

    # 95% CI
    print str(mean_error) + " +- " + str(half_range)
    if num_genres > 2:
        # Averaged confusion matrix
        print numpy.array(average_confusion_matrix * 1000.0, dtype=int) / 100.0
