import numpy as np
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Dataset: time taken for each rep


def failureRep(rep_times):
# Prepare the data
    X = np.arange(1, len(rep_times) + 1).reshape(-1, 1)  # Repetition numbers (1, 2, 3, ...)
    y = np.array(rep_times)  # Corresponding times

    # Use polynomial features (degree 2) to model the data
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    # Train the model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Function to predict the time for a specific repetition number
    def predict_time(repetition_number):
        X_input = np.array([[repetition_number]])  # Create the input array
        X_input_poly = poly.transform(X_input)  # Transform the input using polynomial features
        predicted_time = model.predict(X_input_poly)  # Predict the time
        return predicted_time[0]


    # Plot the actual data and the predicted curve
    """X_plot = np.linspace(1, len(rep_times) + 3, 100).reshape(-1, 1)  # Plotting range
    X_plot_poly = poly.transform(X_plot)  # Transform plot range using polynomial features
    y_plot = model.predict(X_plot_poly)  # Predicted values for plotting

    plt.scatter(X, y, color='blue', label='Actual Data')
    plt.plot(X_plot, y_plot, color='red', linestyle='--', label='Predicted Curve')
    plt.title('Time Taken for Each Repetition (with Polynomial Regression)')
    plt.xlabel('Repetition Number')
    plt.ylabel('Time (seconds)')
    plt.xticks(np.arange(1, len(rep_times) + 4))  # Set x ticks for the plot
    plt.grid(True)
    plt.legend()
    plt.show() """

    n=0

    while(1):

        t = predict_time(n)
        
        if(t>3.5):
            print(f"\nMuscular failure occurs at rep number {n}"  )
            return (n)
        
            
        n=n+1

