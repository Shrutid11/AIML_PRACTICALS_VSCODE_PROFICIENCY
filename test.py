libraries = [
    "numpy", "pandas", "seaborn", "sklearn", "tensorflow", "keras",
    "scipy", "os", "sys", "random", "math", "re", "matplotlib"
]

missing_libraries = []

for lib in libraries:
    try:
        __import__(lib)
    except ImportError:
        missing_libraries.append(lib)

if missing_libraries:
    print("The following libraries are missing or not installed:")
    print("\n".join(missing_libraries))
else:
    print("All required libraries are installed.")