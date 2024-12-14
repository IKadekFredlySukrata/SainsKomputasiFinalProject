import matplotlib
import numpy
import scipy
import sklearn.metrics
from sympy import symbols, diff, lambdify, sympify
import matplotlib.pyplot as plt

# Input Matrix
def matrixInput():
    rows = int(input("Rows  : "))
    cols = int(input("Cols  : "))

    elements = list(map(float, input(f"Enter {rows * cols} elements (space-separated): ").split()))

    if len(elements) != rows * cols:
        print(f"Please input the matrix correctly with {rows * cols} elements.")
        return None

    else:
        matrix = numpy.array(elements).reshape(rows, cols)
        print("Matrix:")
        print(matrix)
        return matrix

# Input Vektor
def input_vektor(n):
    print(f"Masukkan elemen vektor ukuran {n}:")
    return numpy.array([float(input(f"Elemen {i + 1}: ")) for i in range(n)])

# Matrix Basic Operation
def matrixSum():
    print("\nSum Matrix\n")
    print("\nMatrix A:")
    A = matrixInput()
    if A is None:
        return

    print("\nMatrix B:")
    B = matrixInput()
    if B is None:
        return

    if A.shape != B.shape:
        print("The matrices cannot be summed because their dimensions do not match.")
        return

    print("\nResult:\n", A + B)

def matrixSubstraction():
    print("\nMatrix Substraction\n")
    print("\nMatrix A:")
    A = matrixInput()
    if A is None:
        return

    print("\nMatrix B:")
    B = matrixInput()
    if B is None:
        return

    if A.shape != B.shape:
        print("The matrices cannot be summed because their dimensions do not match.")
        return

    print("\nResult:\n", A - B)

def matrixScalarMultiplication():
    print("\nMatrix Skalar Multiplication\n")
    print("\nMatrix: ")
    A = matrixInput()
    if A is None:
        return
    
    scalar = float(input("Scalar: "))

    print("\nResult:\n", A * scalar)

def matrixMultiplication():
    print("\nMatrix Multiplication\n")
    print("\nMatrix: ")
    A = matrixInput()
    if A is None:
        return
    
    print("\nMatrix B:")
    B = matrixInput()
    if B is None:
        return
    
    if A.shape[1] != B.shape[0]:
        print("These Matrices don't meet the requirement from this operation")
    
    print("\nResult:\n", numpy.dot(A, B))

def inversMatrix():
    print("\nInvers Matrix\n")
    print("\nMatrix: ")
    A = matrixInput()
    if A.shape[0] != A.shape[1]:
        print("The invers of this matrix can't be solve")
        return
    try:
        print("\nResult:\n", numpy.linalg.inv(A))

    except numpy.linalg.LinAlgError:
        print("The invers can't be solve, because the determinant value of the matrix is zero")

def determinantMatrix():
    print("\nDeterminant Matrix\n")
    print("\nMatrix: ")
    A = matrixInput()
    if A.shape[0] != A.shape[1]:
        print("The determinant of this matrix can't be solve")
        return
    print("\nResult:\n", numpy.linalg.det(A).astype(int))

def transposeMatrix():
    print("\nTranspose Matrix\n")
    print("\nMatrix: ")
    A = matrixInput()
    print("\nResult:\n", A.T)

# LU Decomposition
def matrixLUDecomposition():
    print("\nLU Decomposition\nAx = B\n")
    print("\nMatrix A: ")
    A = matrixInput()

    print("\nMatriks B: ")
    B = matrixInput()

    P, L, U = scipy.linalg.lu(A)
    print("\nOriginalMatrix\n", A)
    print("\nPermutation Matrix\n", P)
    print("\nLower Triangular Matrix\n", L)
    print("\nUpper Triangular Matrix\n", U)

    lu, piv = scipy.linalg.lu_factor(A)

    x = scipy.linalg.lu_solve((lu, piv), B)

    print("\nSolution X:\n", x)

# Iteration
# Jacobi Iteration
def jacobiIteration():
    print("\nJacobi Iteration\n")
    print("Matrix:\n")
    A = matrixInput()
    B = input_vektor(len(A))
    X = numpy.zeros_like(B)
    n = len(A)
    maxIteration = int(input("\nMax Iteration: "))
    resultTolerance = float(input("\nResult Tolerance: "))
    
    for _ in range(maxIteration):
        X_new = numpy.copy(X)
        for i in range(A.shape[0]):
            sum_ax = numpy.dot(A[i, :i], X_new[:i]) + numpy.dot(A[i, i + 1:], X[i + 1:])
            X_new[i] = (B[i] - sum_ax) / A[i, i]
        if numpy.linalg.norm(X_new - X, ord=numpy.inf) < resultTolerance:
            print("Result: ", X_new)
            return
        X = X_new
    print("Max iteration reached.\nResult:", X_new)
    
def seidelIteration():
    print("\nSeidel Iteration\n")
    print("Matrix Input:\n")
    A = matrixInput()
    if A is None:
        return
    B = input_vektor(len(A))
    X = numpy.zeros_like(B, dtype=float)
    maxIteration = int(input("Max Iteration: "))
    resultTolerance = float(input("Result Tolerance: "))
    
    print("\nStarting Iterations...\n")
    
    for iteration in range(maxIteration):
        X_new = numpy.copy(X)
        for i in range(A.shape[0]):
            sum_ax = numpy.dot(A[i, :i], X_new[:i]) + numpy.dot(A[i, i + 1:], X[i + 1:])
            X_new[i] = (B[i] - sum_ax) / A[i, i]
        
        diff = numpy.linalg.norm(X_new - X, ord=numpy.inf)
        if diff < resultTolerance:
            print(f"Converged in {iteration + 1} iterations.")
            print("Result:", X_new)
            return
        
        X = X_new
    
    print("Max Iteration reached.")
    print("Result:", X)

# Interpolation
def interpolation():
    print("\nInterpolation\n")
    print("1. Linear")
    print("2. Non-Linear")
    choice = int(input("Operation: "))
    
    if choice == 1:
        x0, y0 = map(float, input("First Point  (x0 y0): ").split())
        x1, y1 = map(float, input("Second Point (x1 y1): ").split())
        x = float(input("Target Position: "))
        
        if x1 == x0:
            raise ValueError("x0 and x1 can't be the same value")
        
        y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        print(f"Result:\nf({x}) = {y}")
    
    elif choice == 2:
        n = int(input("Number of Points: "))
        x_points = []
        y_points = []
        
        print("Enter the points (x y):")
        for _ in range(n):
            x, y = map(float, input().split())
            x_points.append(x)
            y_points.append(y)
        
        x = float(input("Target Position: "))
        
        def lagrange(x_points, y_points, x):
            n = len(x_points)
            y_estimated = 0
            
            for i in range(n):
                L_i = 1
                for j in range(n):
                    if i != j:
                        L_i *= (x - x_points[j]) / (x_points[i] - x_points[j])
                y_estimated += y_points[i] * L_i
            
            return y_estimated
        
        y = lagrange(x_points, y_points, x)
        print(f"Result:\nf({x}) = {y}")
    
    else:
        print("Invalid choice. Please choose 1 or 2.")

# Linear
# Table Method

def tableMethod():
    print("\nTable Method\n")
    
    f = sympify(input("f(x): "))
    x = symbols('x')
    f_lambdified = lambdify(x, f)

    a = float(input("Lower Limit: "))
    b = float(input("Upper Limit: "))
    N = int(input("Patition Quantity: "))
    
    h = (b - a) / N
    print(f"\nStep size: {h:.4f}")
    

    print("\nxi       | f(xi)")
    print("-" * 20)
    x_prev, f_prev = None, None
    results = []
    
    for i in range(N + 1):
        xi = a + i * h
        fi = f_lambdified(xi)
        print(f"{xi:.4f} | {fi:.6f}")
        
        if i > 0:
            if f_prev * fi < 0:
                results.append(f"Root found between x = {x_prev:.4f} and x = {xi:.4f}")
            elif abs(fi) < 1e-6:
                results.append(f"Root found at x = {xi:.4f}")
        
        x_prev, f_prev = xi, fi
    
    print("\nResults:")
    if results:
        for res in results:
            print(res)
    else:
        print("No roots found in the given range.")

# Bisection Method
def bisectionMethod():
    print("\nBisection Method\n")
    f = sympify(input("f(x): "))
    x = symbols('x')
    f_lambdified = lambdify(x, f)
    
    a = float(input("Upper Limit: "))
    b = float(input("Lower Limit: "))
    maxIteration = int(input("Max Iteration: "))
    resultTolerance = float(input("Result Tolerance: "))

    if f_lambdified(a) * f_lambdified(b) >= 0:
        print("There is no root in the interval [{a}, {b}]")
        return
    
    iteration = 0
    while abs(b - a) > resultTolerance or iteration > maxIteration:
        c = (a + b) / 2
        if f_lambdified(c) == 0:
            break
        elif f_lambdified(a) * f_lambdified(c) < 0:
            b = c
        else:
            a = c
    print(f"Result: {c:.4f}")

# Regula Falsi Method
def regulaFalsiMethod():
    print("\nRegula Falsi Method\n")
    f = sympify(input("f(x): "))
    x = symbols('x')
    f_lambdified = lambdify(x, f)
    
    a = float(input("Upper Limit: "))
    b = float(input("Lower Limit: "))
    maxIteration = int(input("Max Iteration: "))
    resultTolerance = float(input("Result Tolerance: "))

    if f_lambdified(a) * f_lambdified(b) >= 0:
        print("There is no root in the interval [{a}, {b}]")
        return

    x_old = None
    for _ in range(maxIteration):
        x = (a * f_lambdified(b) - b * f_lambdified(a)) / (f_lambdified(b) - f_lambdified(a))
        if x_old and abs(x - x_old) < resultTolerance:
            break
        elif f_lambdified(x) == 0:
            break
        elif f_lambdified(a) * f_lambdified(x) < 0:
            b = x
        else:
            a = x
            x_old = x
    print(f"Result: {x:.4f}")

# Newton Raphson Method
def newtonRaphsonMethod():
    print("\nNewton-Raphson Method\n")
    f = sympify(input("f(x): "))
    x = symbols('x')
    f_lambdified = lambdify(x, f)
    f_derivative_lambdified = lambdify(x, diff(f, x))

    maxIteration = int(input("Max Iteration: "))
    resultTolerance = float(input("Result Tolerance: "))
    initialGuess = float(input("Initial Guess: "))

    for _ in range(maxIteration):
        f_initialGuess = f_lambdified(initialGuess)
        f_derivative_initialGuess = f_derivative_lambdified(initialGuess)

        if abs(f_derivative_initialGuess) < 1e-12:
            print("Derivative is too small")
            return
            
        nextGuess = initialGuess - f_initialGuess / f_derivative_initialGuess

        if abs(nextGuess - initialGuess) < resultTolerance:
            print(f"Result: {nextGuess:.4f}")
            return

        initialGuess = nextGuess
    
    print(f"Maximum iteration reached.\nResult: {initialGuess:.4f}")

# Secant Method
def secantMethod():
    print("\nSecant Method\n")
    f = sympify(input("f(x): "))
    x = symbols('x')
    f_lambdified = lambdify(x, f)
    
    initialGuess1 = float(input("Initial Guess 1: "))
    initialGuess2 = float(input("Initial Guess 2: "))
    maxIteration = int(input("Max Iteration: "))
    resultTolerance = float(input("Result Tolerance: "))

    for _ in range(maxIteration):
        f_x1 = f_lambdified(initialGuess1)
        f_x2 = f_lambdified(initialGuess2)

        if abs(f_x2 - f_x1) < 1e-12:
            print("Function values are too close")
            return

        nextGuess = initialGuess2 - f_x2 * (initialGuess2 - initialGuess1) / (f_x2 -f_x1)

        if abs(nextGuess - initialGuess2) < resultTolerance:
            print(f"Result: {nextGuess:.4f}")
            return

        initialGuess1, initialGuess2 = initialGuess2, nextGuess

# Gauss Elimination
def gaussElimination():
    print("\nGauss Elimination Method\n")
    
    n = int(input("Enter the number of variables: "))
    
    print("Enter the augmented matrix row by row (including constants):")
    augmented_matrix = []
    for _ in range(n):
        row = list(map(float, input().split()))
        augmented_matrix.append(row)
    augmented_matrix = numpy.array(augmented_matrix, dtype=float)
    
    for i in range(n):
        if augmented_matrix[i, i] == 0:
            for k in range(i + 1, n):
                if augmented_matrix[k, i] != 0:
                    augmented_matrix[[i, k]] = augmented_matrix[[k, i]]  # Swap rows
                    break
            else:
                print("No unique solution exists.")
                return
        
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]
        
        for k in range(i + 1, n):
            factor = augmented_matrix[k, i]
            augmented_matrix[k] -= factor * augmented_matrix[i]
    
    x = numpy.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented_matrix[i, -1] - numpy.sum(augmented_matrix[i, i+1:n] * x[i+1:n])
    
    print("Solution:")
    for i in range(n):
        print(f"x{i + 1} = {x[i]:.4f}")

# Simulation
# Monte Carlo
def monteCarlo():
    print("\nMonte Carlo Integration\n")
    
    from sympy import sympify, symbols, lambdify
    
    x = symbols('x')
    f = sympify(input("Enter the function f(x): "))
    f_lambdified = lambdify(x, f)
    
    a = float(input("Enter lower limit (a): "))
    b = float(input("Enter upper limit (b): "))
    N = int(input("Enter number of random samples (N): "))
    
    random_samples = numpy.random.uniform(a, b, N)
    function_values = f_lambdified(random_samples)
    
    result = (b - a) * numpy.mean(function_values)
    
    print(f"Approximate integral value: {result:.4f}")

    x_plot = numpy.linspace(a, b, 1000)
    y_plot = f_lambdified(x_plot)
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, label="f(x)", color="blue")
    plt.scatter(random_samples, function_values, color="red", s=10, label="Random Points")
    plt.title("Monte Carlo Integration")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()

# Markov Chain
def markovChain():
    print("\nMarkov Chain Simulation\n")
    
    # Input the states and transition matrix
    num_states = int(input("Enter the number of states: "))
    states = [input(f"State {i+1}: ") for i in range(num_states)]
    
    print("\nEnter the transition matrix (row by row):")
    transition_matrix = []
    for i in range(num_states):
        row = list(map(float, input(f"Row {i+1}: ").split()))
        if not numpy.isclose(sum(row), 1):
            print("Error: Each row must sum to 1. Please re-enter the matrix.")
            return
        transition_matrix.append(row)
    transition_matrix = numpy.array(transition_matrix)
    
    print("\nEnter the initial state probabilities:")
    initial_state = list(map(float, input().split()))
    if not numpy.isclose(sum(initial_state), 1):
        print("Error: Initial state probabilities must sum to 1.")
        return
    initial_state = numpy.array(initial_state)
    
    # Input number of steps
    steps = int(input("\nEnter the number of steps to simulate: "))
    
    # Simulate the Markov Chain
    current_state = initial_state
    print("\nState distributions over steps:")
    print(f"Step 0: {dict(zip(states, current_state))}")
    
    for step in range(1, steps + 1):
        current_state = numpy.dot(current_state, transition_matrix)
        print(f"Step {step}: {dict(zip(states, current_state))}")

def curveFitting():
    # Define models
    def linear(x, a, b):
        return a * x + b

    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c

    def exponential(x, a, b):
        return a * numpy.exp(b * x)

    # Generate base dataset (quadratic with noise)
    numpy.random.seed(42)
    x_data = numpy.linspace(0, 10, 100)
    y_data = 2 * x_data**2 - 3 * x_data + 5 + numpy.random.normal(0, 5, len(x_data))

    # Monte Carlo parameters
    n_simulations = 100  # Number of simulations
    noise_std = 3        # Standard deviation of noise

    # Store results for analysis
    monte_carlo_results = {model_name: [] for model_name in ["Linear", "Quadratic", "Exponential"]}

    # Perform Monte Carlo simulation
    for _ in range(n_simulations):
        # Add random noise to the original data
        y_noisy = y_data + numpy.random.normal(0, noise_std, size=y_data.shape)

        # Fit each model to the noisy data
        for name, model in {"Linear": linear, "Quadratic": quadratic, "Exponential": exponential}.items():
            try:
                params, _ = scipy.optimize.curve_fit(model, x_data, y_noisy, maxfev=10000)
                y_pred = model(x_data, *params)
                avg_distance = sklearn.metrics.mean_absolute_error(y_noisy, y_pred)
                monte_carlo_results[name].append(avg_distance)
            except RuntimeError:
                monte_carlo_results[name].append(numpy.inf)

    # Analyze the results
    average_distances = {name: numpy.mean(distances) for name, distances in monte_carlo_results.items()}
    best_model_name = min(average_distances, key=average_distances.get)

    # Fit the best model to the original data
    best_model = {"Linear": linear, "Quadratic": quadratic, "Exponential": exponential}[best_model_name]
    params, _ = scipy.optimize.curve_fit(best_model, x_data, y_data, maxfev=10000)
    y_best_fit = best_model(x_data, *params)

    # Generate the equation string
    if best_model_name == "Linear":
        equation = f"y = {params[0]:.4f}x + {params[1]:.4f}"
    elif best_model_name == "Quadratic":
        equation = f"y = {params[0]:.4f}x^2 + {params[1]:.4f}x + {params[2]:.4f}"
    elif best_model_name == "Exponential":
        equation = f"y = {params[0]:.4f} * exp({params[1]:.4f}x)"

    # Plot the original data and best-fit curve
    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.scatter(x_data, y_data, label='Original Data', color='black', s=15)
    matplotlib.pyplot.plot(x_data, y_best_fit, label=f'{best_model_name} Fit (Best)', color='red', linewidth=2)
    matplotlib.pyplot.title("Best Fit Curve with Monte Carlo Simulation")
    matplotlib.pyplot.xlabel("X")
    matplotlib.pyplot.ylabel("Y")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.text(0.6, 0.1, f"Best Fit Equation:\n{equation}", transform=matplotlib.pyplot.gca().transAxes,
                       fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    matplotlib.pyplot.show()

    # Print the Monte Carlo results
    print("Monte Carlo Simulation Results:")
    for name, avg_dist in average_distances.items():
        print(f"Model: {name}, Average Distance: {avg_dist:.4f}")
    print(f"Best Model Based on Monte Carlo: {best_model_name}")
    print(f"Best Fit Equation: {equation}")

# Menu
def menu():
    print("\n=== Operation Menu ===")
    print("\n=== Matrix Basic Operation ===")
    print("1. Matrix Sum")
    print("2. Matrix Substraction")
    print("3. Matrix Scalar Multiplication")
    print("4. Matrix Multiplication")
    print("5. Inverse Matrix")
    print("6. Determinant Matrix")
    print("7. Transpose Matrix")
    print("\n=== Finding Solution Operation ===")
    print("8. LU Decomposition")
    print("9. Jacobi Iteration")
    print("10. Seidel Iteration")
    print("11. Interpolation")
    print("12. Table Method")
    print("13. Bisection Method")
    print("14. Regula Falsi Method")
    print("15. Newton-Raphson Method")
    print("16. Secant Method")
    print("17. Gauss Elimination")
    print("\n=== Simulation ===")
    print("18. Monte Carlo")
    print("19. Markov Chain")
    print("\n=== Implementation ===")
    print("20. Curve Fitting")
    print("0. Exit")

# Main
def main():
    while True:
        menu()
        choice = int(input("Operation: "))
        
        actions = {
            1: matrixSum,
            2: matrixSubstraction,
            3: matrixScalarMultiplication,
            4: matrixMultiplication,
            5: inversMatrix,
            6: determinantMatrix,
            7: transposeMatrix,
            8: matrixLUDecomposition,
            9: jacobiIteration,
            10: seidelIteration,
            11: interpolation,
            12: tableMethod,
            13: bisectionMethod,
            14: regulaFalsiMethod,
            15: newtonRaphsonMethod,
            16: secantMethod,
            17: gaussElimination,
            18: monteCarlo,
            19: markovChain,
            20: curveFitting,
            0: lambda: print("Exiting...") or exit()
        }
        
        action = actions.get(choice, lambda: print("Invalid choice. Please try again."))
        action()

# Start
main()
