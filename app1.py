import subprocess

def run_scripts():
    # Run the first script
    subprocess.Popen(['python', 'exercise.py'])
    
    # Run the second script
    subprocess.Popen(['python', 'videofeed.py'])

if __name__ == '__main__':
    run_scripts()
