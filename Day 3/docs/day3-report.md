Day 3 Report – AI/ML Developer Track
Industry Immersion Program
 1. Setup Status
VS Code / PyCharm environment successfully configured
Python interpreter and virtual environment set up
Required libraries installed:
NumPy
Pandas
Git and GitHub integration completed
Project folder structured properly
2. Task Inventory
Created and executed basics.py for Python fundamentals
Implemented:
Data containers (list, string, integer)
Processing loop for data validation
Analytical function for statistical analysis
Learned and implemented NumPy operations:
Array creation
Reshaping (1D → 2D matrix)
Vectorized operations (scaling data)
Created data.csv dataset for analysis
Performed Pandas operations:
Loaded dataset using pd.read_csv()
Displayed top records using df.head()
Displayed bottom records using df.tail()
Checked dataset shape using df.shape
Viewed column names using df.columns
Inspected data types using df.dtypes
Generated statistical summary using df.describe()
Calculated mean of Score column using df['Score'].mean()
Found maximum value using df['Score'].max()
Found minimum value using df['Score'].min()
Filtered data based on condition (e.g., Score > 80)
Sorted dataset using df.sort_values()
Organized project into structured folders:
/scripts
/data
/docs
/notebooks
3. Debugging Log
Issue 1: NumPy Installation Error
Error: externally-managed-environment
Cause: Python environment managed by system (uv), restricting direct package installation
Solution: Created a virtual environment (venv) in PyCharm and installed NumPy successfully

Issue 2: File Path Error in Pandas
Error: invalid escape sequence \A
Cause: Incorrect use of backslashes in file path
Solution: Fixed using:
Raw string (r"path")
OR double backslashes (\\)
OR forward slashes (/)
4. Key Insights (Aha Moment)
Understood how loops iterate over data for validation, similar to real-world data preprocessing
Learned that functions act as reusable data processors in AI pipelines
Realized the power of NumPy vectorization, which performs operations on entire datasets without loops, improving performance
Discovered how Pandas simplifies data analysis, making it easy to extract insights from structured data
5. Learning Outcome
Today marked a transition from basic coding to data handling and analysis, which is a core requirement in AI/ML development. I gained hands-on experience in building a small data pipeline involving data storage, processing, and analysis using Python, NumPy, and Pandas.
