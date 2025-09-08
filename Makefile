# Install dependencies
install:
	python3 -m pip install -r requirements.txt

# Run the data analysis script
run:
	python3 analyzedata.py

# Clean cache
clean:
	rm -rf __pycache__ .ipynb_checkpoints
