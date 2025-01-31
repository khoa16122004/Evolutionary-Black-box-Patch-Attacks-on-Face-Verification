import subprocess

script_path = "results.sh"

try:
    with open(script_path, "r") as file:
        for line in file:
            command = line.strip()  
            if command:  
                print(f"Running: {command}")
                result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
                print("Output:", result.stdout)
except subprocess.CalledProcessError as e:
    print("Error:", e.stderr)
except FileNotFoundError:
    print(f"File '{script_path}' không tồn tại.")
