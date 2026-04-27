## Environment setup and run

1. Create a virtual environment (first time only):

```bash
python -m venv .venv
```

2. Activate the environment:

- PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

- Git Bash:

```bash
source .venv/Scripts/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the simulation:

```bash
python SIR_model.py
```

5. Examine the model outputs in the "output" folder

## Additional: macOS/Linux

1. Create a virtual environment (first time only):

```bash
python3 -m venv .venv
```

2. Activate the environment:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the simulation:

```bash
python3 SIR_model.py
```