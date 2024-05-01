<head>
</head>
<body>
    <header>
        <h1>Neural Networks Project Overview</h1>
    </header>
    <main>
      <p>
        This project involves designing and evaluating a neural network using the MLPClassifier from Scikit-learn. It focuses on applying machine learning techniques to recognize patterns in the MNIST dataset of handwritten digits.
      </p>
      <h1>
          Important Warning!
      </h1>
      <p>
          The GridSearch functionality in the 'Neural_Network_Model' file is configured with <code>n_jobs=7</code> by default. This setting indicates that it will utilize 7 CPU cores to perform hyperparameter optimization. Please adjust this setting according to your system's capabilities to avoid system overloads or crashes.
      </p>
      <h2>Getting Started</h2>
      <p>
        Begin by cloning the repository to your local machine. Once cloned, navigate to the project directory, unzip the MNIST csv files, and run the main script. The script prompts you to select the MNIST csv files; choose the training dataset first, followed by the test dataset.
      </p>
      <p>
        You can adjust various parameters, including:
      </p>
      <ul>
        <li><strong>Data:</strong> Training and Testing Datasets (Primarily uses MNIST).</li>
        <li><strong>Split:</strong> Percentage of the training data to be used as a validation set.</li>
        <li><strong>Base Model:</strong> Parameters like hidden layers, activation function, solver, iterations, and random state can be modified.</li>
        <li><strong>Optimal Model:</strong> Parameters tested to determine the optimal model are configurable in the 'param_grid'.</li>
      </ul>
      <h2>Project Implementation</h2>
      <p>The project is implemented in several key phases:</p>
      <ul>
        <li>Preprocessing the data and dividing it into training and test sets.</li>
        <li>Utilizing the MLPClassifier to classify digits.</li>
        <li>Evaluating model performance using metrics such as accuracy, precision, and F1 scores.</li>
        <li>Optimizing hyperparameters to improve the effectiveness of the model.</li>
      </ul>
    </main>
    <footer>
        <p>&copy; 2024 Dewald Oosthuizen 38336529. All rights reserved.</p>
    </footer>

</body>
</html>
