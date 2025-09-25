# Install dependencies
install:
	python3 -m pip install -r requirements.txt

test:
	pytest -v

# Run the data analysis script
run:
	python3 analyzedata.py

# Clean cache
clean:
	rm -rf __pycache__ .ipynb_checkpoints

# Lint the code
lint:
	flake8 --max-line-length=90 *.py


# Format the code
format:
	black analyzedata.py test_analyzedata.py
