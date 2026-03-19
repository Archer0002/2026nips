import subprocess
import time
import sys
import os
import signal

# 子进程列表
subprocesses = []

def close_subprocesses(signum, frame):
    print("Received interrupt signal. Closing subprocesses...")
    for proc in subprocesses:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

    sys.exit(0)

if __name__ == '__main__':
    num_processes = 50

    signal.signal(signal.SIGINT, close_subprocesses)

    for i in range(num_processes):
        exp = i + 205

        command = f'taskset -c {exp} python main_synthetic.py --num_exp {i}' 
        # command = f'taskset -c {exp} python main_criteo.py --num_exp {i}' 

        proc = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
        subprocesses.append(proc)
    
    print('all subprocesses are running...')
    
    for proc in subprocesses:
        proc.wait()  # 等待子进程结束

    print('all subprocesses have finished.')

    sys.exit(0)
