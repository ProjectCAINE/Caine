
import subprocess
import threading

def run_training():
    subprocess.run(["python", "train.py"])

def run_eyes():
    subprocess.run(["python", "eyes.py"])

if __name__ == "__main__":
    t1 = threading.Thread(target=run_training)
    t2 = threading.Thread(target=run_eyes)
    
    t1.start()
    t2.start()
