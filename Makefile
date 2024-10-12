VENV_DIR = venv 
PYTHON = $(VENV_DIR)/bin/python3
PIP = $(VENV_DIR)/bin/pip

#create venv if not existing 
$(VENV_DIR)/bin/activate:
	python3 -m venv $(VENV_DIR)
	$(PIP) install -r requirements.txt

run : $(VENV_DIR)/bin/activate
	$(PYTHON) first_neuron.py


clean_venv :
	rm -rf $(VENV_DIR)