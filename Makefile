.PHONY: clean

clean:
	find . -name __pycache__ | xargs rm -rf 
	find . -name .ipynb_checkpoints| xargs rm -rf

